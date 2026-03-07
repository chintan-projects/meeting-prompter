"""Session manager — bridges dual audio pipelines to WebSocket consumers.

Dual-stream architecture: captures microphone (you) and system audio
(others) simultaneously. Speaker identity is deterministic from the
audio source — no ML-based speaker attribution needed.

Pipeline flow:
    Mic Audio   → ASR → source="mic"    → TranscriptBuffer → "You"
    System Audio → ASR → source="system" → TranscriptBuffer → "Others"
    Both streams → on_silence → turn boundaries
    Both streams → ConversationBuffer → trigger engine → prompts
"""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from lib.config import AppConfig, load_config
from lib.conversation.meeting_context import MeetingContext, load_meeting_context
from lib.generation.types import GenerationResult
from lib.text_refiner import TextRefiner
from lib.triggers.types import Trigger, TriggerType

from .transcript_buffer import TranscriptBuffer, Turn
from .transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

# Output directory for session recordings
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


class Session:
    """Managed meeting session with dual-stream audio capture.

    Two AudioCapture instances (mic + system audio) feed a shared
    TranscriptBuffer. Each turn is tagged with its source ("mic" or
    "system"), and speaker labels are set deterministically:
    - source="mic" → speaker="You"
    - source="system" → speaker="Others"

    Transcript turns and trigger results are pushed into asyncio queues
    that WebSocket handlers consume.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.transcript = TranscriptStore()
        self.meeting_context: Optional[MeetingContext] = None

        # Turn-based buffer: accumulates raw chunks into speech turns
        self._transcript_buffer = TranscriptBuffer(
            turn_pause=2.0,
            max_turn_duration=30.0,
            on_update=self._on_turn_update,
            on_final=self._on_turn_final,
        )

        # Async queues for WebSocket consumers
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._prompt_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._orchestrator: Optional[object] = None
        self._mic_capture: Optional[object] = None
        self._text_refiner: Optional[TextRefiner] = None
        self._thread: Optional[threading.Thread] = None
        self._mic_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_time: float = 0.0
        self._loading = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def elapsed_seconds(self) -> float:
        if not self._start_time:
            return 0.0
        return time.time() - self._start_time

    def load_context(self, path: Path) -> Optional[MeetingContext]:
        """Load meeting context from YAML."""
        self.meeting_context = load_meeting_context(path)
        return self.meeting_context

    def start(
        self,
        audio_device: str = "BlackHole 2ch",
        mic_device: str = "",
    ) -> None:
        """Start the dual audio pipeline in background threads.

        Args:
            audio_device: System audio device (e.g. "BlackHole 2ch").
            mic_device: Microphone device (e.g. "MacBook Pro Microphone").
                If empty, falls back to config default.
        """
        if self._running or self._loading:
            logger.warning("Session already running or loading")
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self._loading = True
        self._running = True
        self._start_time = time.time()
        self._audio_device = audio_device
        self._mic_device = mic_device or self.config.audio.device_mic
        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()
        logger.info(
            "Session starting — system: %s, mic: %s",
            audio_device,
            self._mic_device,
        )

    def stop(self) -> None:
        """Stop both audio pipelines, finalize active turn, save recording."""
        self._running = False
        if self._orchestrator:
            self._orchestrator._running = False

        # Stop mic capture
        if self._mic_capture:
            self._mic_capture.stop()

        # Flush any in-progress turn
        self._transcript_buffer.flush()

        if self._thread:
            self._thread.join(timeout=10.0)
        if self._mic_thread:
            self._mic_thread.join(timeout=10.0)

        # Save session recording (system audio — primary stream)
        if self._orchestrator and hasattr(self._orchestrator, "audio"):
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            recording_path = _OUTPUT_DIR / f"session_{ts}.wav"
            self._orchestrator.audio.save_recording(recording_path)

        logger.info("Session stopped after %.0fs", self.elapsed_seconds)

    def get_status(self) -> dict:
        """Get current session status."""
        audio_health: dict = {}
        if self._orchestrator and hasattr(self._orchestrator, "audio"):
            audio_health = self._orchestrator.audio.get_audio_health()
        return {
            "running": self._running,
            "loading": self._loading,
            "elapsed_seconds": self.elapsed_seconds,
            "segment_count": self.transcript.segment_count,
            "meeting_title": (
                self.meeting_context.title if self.meeting_context else ""
            ),
            "audio_health": audio_health,
        }

    # --- Orchestrator callbacks (called from system audio pipeline thread) ---

    def _on_transcription(self, text: str, timestamp: float) -> None:
        """Called when valid speech is transcribed from system audio.

        Tags the chunk with source="system" (remote participants).
        """
        logger.debug("[system] Valid speech: %r", text[:80])
        self._transcript_buffer.add_chunk(text, timestamp, source="system")

    def _on_mic_transcription(self, text: str, timestamp: float) -> None:
        """Called when valid speech is transcribed from microphone.

        Tags the chunk with source="mic" (local user).
        """
        logger.debug("[mic] Valid speech: %r", text[:80])
        self._transcript_buffer.add_chunk(text, timestamp, source="mic")

    def _on_silence_detected(self, timestamp: float) -> None:
        """Called when ASR returns empty/hallucination (silence).

        Notifies the TranscriptBuffer so it can finalize the active turn
        if the silence gap exceeds the turn_pause threshold.
        """
        self._transcript_buffer.on_silence(timestamp)

    def _on_trigger_result(self, trigger: Trigger, result: GenerationResult) -> None:
        """Called when a trigger fires and produces a generation result."""
        logger.info(
            "[trigger] %s → %s (conf=%.2f, %dms)",
            trigger.type.value,
            result.method,
            result.confidence,
            result.latency_ms,
        )

        if trigger.type == TriggerType.QUESTION:
            self._orchestrator.buffer.add_qa_pair(trigger.text, result.answer)

        self._thread_safe_put(self._prompt_queue, {
            "type": "prompt",
            "trigger_type": trigger.type.value,
            "trigger_text": trigger.text,
            "answer": result.answer,
            "confidence": result.confidence,
            "method": result.method,
            "latency_ms": result.latency_ms,
            "source": result.source,
        })

    # --- Turn callbacks (called from TranscriptBuffer in pipeline thread) ---

    def _on_turn_update(self, turn: Turn) -> None:
        """Called when a turn is updated (new chunk added)."""
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=False,
        )
        self._thread_safe_put(self._transcript_queue, {
            "type": "transcript_update",
            **turn.to_dict(),
        })

    def _on_turn_final(self, turn: Turn) -> None:
        """Called when a turn is finalized (pause detected).

        Speaker label is set deterministically from source:
        mic → "You", system → "Others".
        """
        # Source-based speaker attribution — deterministic, no ML
        if turn.source == "mic":
            turn.speaker = "You"
        elif turn.source == "system":
            turn.speaker = "Others"

        # Emit raw finalization immediately
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=True,
        )
        self._thread_safe_put(self._transcript_queue, {
            "type": "transcript_final",
            **turn.to_dict(),
        })

        # Polish with text refiner (runs in pipeline thread during silence)
        if self._text_refiner:
            polished = self._text_refiner.refine(turn.text)
            if polished and polished != turn.text:
                self.transcript.upsert(
                    seg_id=turn.id,
                    text=polished,
                    timestamp=turn.start_timestamp,
                    end_timestamp=turn.end_timestamp,
                    is_final=True,
                )
                self._thread_safe_put(self._transcript_queue, {
                    "type": "transcript_polished",
                    "id": turn.id,
                    "text": polished,
                    "timestamp": turn.start_timestamp,
                    "end_timestamp": turn.end_timestamp,
                    "is_final": True,
                    "speaker": turn.speaker,
                    "source": turn.source,
                })

    # --- Pipeline threads ---

    def _run_pipeline(self) -> None:
        """Background thread: load models, start system audio + mic capture."""
        try:
            from lib.audio_capture import AudioCapture
            from lib.orchestrator import MeetingOrchestrator

            logger.info("Loading models in background thread...")
            self._orchestrator = MeetingOrchestrator(
                config=self.config,
                audio_device=self._audio_device,
                meeting_context_path=None,
                headless=True,
            )

            # Inject meeting context if loaded
            if self.meeting_context and self._orchestrator:
                orch = self._orchestrator
                orch.meeting_context = self.meeting_context
                if self.meeting_context.watch_words:
                    orch.trigger_engine.set_watch_words(self.meeting_context.watch_words)

            # Wire orchestrator callbacks for system audio stream
            self._orchestrator.on_transcription = self._on_transcription
            self._orchestrator.on_silence_detected = self._on_silence_detected
            self._orchestrator.on_trigger_result = self._on_trigger_result

            # Create text refiner (shares Llama instance with generator)
            refiner_config = getattr(self.config, "refiner", None)
            refiner_enabled = (
                getattr(refiner_config, "enabled", True) if refiner_config else True
            )
            if refiner_enabled and hasattr(self._orchestrator.generator, "_generator"):
                self._text_refiner = TextRefiner(
                    self._orchestrator.generator._generator,
                    min_words_to_refine=(
                        getattr(refiner_config, "min_words_to_refine", 5)
                        if refiner_config
                        else 5
                    ),
                )
                logger.info("Text refiner enabled (sharing LFM2.5-Instruct instance)")

            # --- Mic capture: separate AudioCapture + ASR pipeline ---
            self._mic_capture = AudioCapture(device=self._mic_device)
            self._mic_thread = threading.Thread(
                target=self._run_mic_pipeline,
                daemon=True,
                name="mic-pipeline",
            )
            self._mic_thread.start()
            logger.info("Mic pipeline started on: %s", self._mic_device)

            self._loading = False
            logger.info("Models loaded, starting system audio capture...")

            # System audio runs on this thread (blocking)
            self._orchestrator.run()
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self._loading = False
            self._running = False

    def _run_mic_pipeline(self) -> None:
        """Mic pipeline thread: capture mic audio, transcribe, tag source.

        Shares the LFM2 ASR model with the system audio pipeline via the
        orchestrator. Uses the same filter chain but tags output as "mic".
        """
        try:
            from lib.filters import is_hallucination_only, is_noise

            if not self._orchestrator:
                logger.error("Mic pipeline: orchestrator not ready")
                return

            lfm2 = self._orchestrator.lfm2

            def process_mic_chunk(audio_path: Path, timestamp: float) -> None:
                """Process a mic audio chunk through ASR + filters."""
                try:
                    text = lfm2.transcribe(audio_path)

                    if not text or text.startswith("["):
                        self._on_silence_detected(timestamp)
                        return

                    if is_hallucination_only(text):
                        self._on_silence_detected(timestamp)
                        return

                    # Valid speech from mic
                    self._on_mic_transcription(text, timestamp)

                    # Feed trigger pipeline (same as system audio)
                    if not is_noise(text):
                        triggers = self._orchestrator.buffer.add_chunk(text, timestamp)
                        self._handle_mic_triggers(triggers)
                except Exception as e:
                    logger.error("Mic chunk error: %s", e)

            self._mic_capture.start_stream(process_mic_chunk)
        except Exception as e:
            logger.error("Mic pipeline error: %s", e, exc_info=True)

    def _handle_mic_triggers(self, triggers: list) -> None:
        """Handle triggers from mic audio (same as system audio triggers)."""
        from lib.triggers.types import Trigger as TriggerType_

        for trigger in triggers:
            result = self._orchestrator._process_trigger(trigger)
            if result and result.answer and not result.answer.startswith("["):
                if self._orchestrator.on_trigger_result:
                    self._orchestrator.on_trigger_result(trigger, result)

    def _thread_safe_put(self, queue: asyncio.Queue, item: dict) -> None:
        """Put an item on an asyncio.Queue from a background thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
        else:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
