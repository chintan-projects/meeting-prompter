"""Session manager — bridges the audio pipeline to WebSocket consumers.

Uses orchestrator callbacks (on_transcription, on_silence_detected,
on_trigger_result) instead of monkey-patching process_chunk. This
keeps the Session's role clear: it wires pipeline events to the
transcript buffer, transcript store, text refiner, and async queues.

Pipeline flow:
    Audio → ASR → Orchestrator.process_chunk() → callbacks
    on_transcription  → TranscriptBuffer → turns → TextRefiner → WebSocket
    on_silence        → TranscriptBuffer → turn boundaries
    on_trigger_result → prompt queue → WebSocket
"""
import asyncio
import logging
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
    """Managed meeting session that bridges audio pipeline to async API.

    The audio pipeline runs in a background thread. Transcript turns
    and trigger results are pushed into asyncio queues that WebSocket
    handlers consume. Orchestrator callbacks bridge the pipeline thread
    to the Session cleanly — no monkey-patching.
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
        self._text_refiner: Optional[TextRefiner] = None
        self._thread: Optional["threading.Thread"] = None
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

    def start(self, audio_device: str = "BlackHole 2ch") -> None:
        """Start the audio pipeline in a background thread."""
        import threading

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
        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()
        logger.info("Session starting on %s (loading models in background)", audio_device)

    def stop(self) -> None:
        """Stop the audio pipeline, finalize active turn, save recording."""
        self._running = False
        if self._orchestrator:
            self._orchestrator._running = False

        # Flush any in-progress turn
        self._transcript_buffer.flush()

        if self._thread:
            self._thread.join(timeout=10.0)

        # Save session recording
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

    # --- Orchestrator callbacks (called from pipeline background thread) ---

    def _on_transcription(self, text: str, timestamp: float) -> None:
        """Called when valid speech is transcribed by the ASR model.

        Feeds the text into the TranscriptBuffer (which handles turn
        assembly and fires _on_turn_update / _on_turn_final).
        """
        logger.info("[pipeline] Valid speech: %r", text[:80])
        self._transcript_buffer.add_chunk(text, timestamp)

    def _on_silence_detected(self, timestamp: float) -> None:
        """Called when ASR returns empty/hallucination (silence).

        Notifies the TranscriptBuffer so it can finalize the active turn
        if the silence gap exceeds the turn_pause threshold.
        """
        self._transcript_buffer.on_silence(timestamp)

    def _on_trigger_result(self, trigger: Trigger, result: GenerationResult) -> None:
        """Called when a trigger fires and produces a generation result.

        Pushes the result to the prompt queue for the WebSocket consumer.
        """
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

        Emits transcript_final (raw text) immediately, then runs the
        text refiner and emits transcript_polished if the text changed.
        """
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
                })

    # --- Pipeline thread ---

    def _run_pipeline(self) -> None:
        """Background thread: load models then run the audio capture loop."""
        try:
            from lib.orchestrator import MeetingOrchestrator

            logger.info("Loading models in background thread...")
            self._orchestrator = MeetingOrchestrator(
                config=self.config,
                audio_device=self._audio_device,
                meeting_context_path=None,
            )

            # Inject meeting context if loaded
            if self.meeting_context and self._orchestrator:
                orch = self._orchestrator
                orch.meeting_context = self.meeting_context
                if self.meeting_context.watch_words:
                    orch.trigger_engine.set_watch_words(self.meeting_context.watch_words)
                if self.meeting_context.title:
                    orch.dashboard.set_meeting_title(self.meeting_context.title)

            # Wire orchestrator callbacks — clean pipeline observation
            self._orchestrator.on_transcription = self._on_transcription
            self._orchestrator.on_silence_detected = self._on_silence_detected
            self._orchestrator.on_trigger_result = self._on_trigger_result

            # Create text refiner (shares Llama instance with generator)
            refiner_config = getattr(self.config, "refiner", None)
            refiner_enabled = getattr(refiner_config, "enabled", True) if refiner_config else True
            if refiner_enabled and hasattr(self._orchestrator.generator, "_generator"):
                self._text_refiner = TextRefiner(
                    self._orchestrator.generator._generator,
                    min_words_to_refine=getattr(refiner_config, "min_words_to_refine", 5)
                    if refiner_config
                    else 5,
                )
                logger.info("Text refiner enabled (sharing LFM2.5-Instruct instance)")

            self._loading = False
            logger.info("Models loaded, starting audio capture...")

            self._orchestrator.run()
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self._loading = False
            self._running = False

    def _thread_safe_put(self, queue: asyncio.Queue, item: dict) -> None:
        """Put an item on an asyncio.Queue from a background thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
        else:
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass
