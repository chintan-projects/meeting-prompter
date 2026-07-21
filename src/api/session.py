"""Session manager — bridges dual audio pipelines to WebSocket consumers.

Dual-stream architecture: captures microphone (you) and system audio
(others) simultaneously. Tier 1 speaker attribution is deterministic
from the audio source. Tier 2 adds neural diarization on system audio
to distinguish individual remote speakers.

Pipeline flow:
    Mic Audio   → ASR → source="mic"    → TranscriptBuffer → "You"
    System Audio → ASR → source="system" → TranscriptBuffer → "Others"
    System Audio → diarization (parallel) → relabel "Others" → "Speaker B"
    Both streams → on_silence → turn boundaries
    Both streams → ConversationBuffer → trigger engine → prompts
"""

import asyncio
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

from lib.config import AppConfig, load_config
from lib.attribution import AttributionResolver, Regime
from lib.conversation.meeting_context import MeetingContext, load_meeting_context
from lib.diarization import SpeakerDiarizer
from lib.generation.types import GenerationResult
from lib.stream_dedup import StreamDeduplicator
from lib.text_refiner import TextRefiner
from lib.triggers.types import Trigger

from .transcript_buffer import TranscriptBuffer, Turn
from .transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

# Output directory for session recordings (resolved via lib.paths)
_OUTPUT_DIR = None  # lazy-resolved to avoid import-time side effects


def _get_output_dir() -> Path:
    """Lazy-resolve output directory on first use."""
    global _OUTPUT_DIR  # noqa: PLW0603
    if _OUTPUT_DIR is None:
        from lib.paths import get_output_dir

        _OUTPUT_DIR = get_output_dir()
    return _OUTPUT_DIR


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
            turn_pause=self.config.buffer.turn_pause,
            max_turn_duration=self.config.buffer.max_turn_duration,
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
        self._diarizer: Optional[SpeakerDiarizer] = None
        self._thread: Optional[threading.Thread] = None
        self._mic_thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._start_time: float = 0.0
        self._loading = False
        self._total_pause_time: float = 0.0
        self._pause_start: float = 0.0
        self._single_device_mode = False  # True when mic == system audio device

        # Accumulated trigger results for post-meeting summary (bounded for long sessions)
        self._trigger_history: deque[dict] = deque(maxlen=1000)

        # Speaker name mapping: diarizer label → custom name (e.g. "Speaker A" → "Alice")
        self._speaker_names: dict[str, str] = {}

        # Attribution hierarchy (F-601): single place that decides a turn's
        # speaker label — L1 channel, L3 acoustic, L4 roster, regime-aware.
        self._resolver = AttributionResolver()

        # Cross-stream echo suppression: detects when mic and system audio
        # both capture the same speech (acoustic coupling without headphones)
        self._deduplicator = StreamDeduplicator(self.config.dual_stream)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def trigger_history(self) -> list[dict]:
        """Accumulated trigger results for post-meeting summary."""
        return list(self._trigger_history)

    def get_rag_engine(self) -> Optional[object]:
        """Public accessor for the RAG engine (if models are loaded)."""
        if self._orchestrator and hasattr(self._orchestrator, "rag"):
            return self._orchestrator.rag
        return None

    @property
    def elapsed_seconds(self) -> float:
        if not self._start_time:
            return 0.0
        total = time.time() - self._start_time
        pause_adjustment = self._total_pause_time
        if self._paused and self._pause_start:
            pause_adjustment += time.time() - self._pause_start
        return total - pause_adjustment

    def load_context(self, path: Path) -> Optional[MeetingContext]:
        """Load meeting context from YAML."""
        self.meeting_context = load_meeting_context(path)
        # L4 roster: seed the attribution resolver with expected participants.
        if self.meeting_context and self.meeting_context.participants:
            self._resolver.set_roster(self.meeting_context.participants)
        # F-606: conference-room regime → honest degradation to a flagged bucket.
        if self.meeting_context and self.meeting_context.conference_room:
            self._resolver.set_regime(Regime.CONFERENCE_ROOM)
        return self.meeting_context

    def start(
        self,
        audio_device: str = "BlackHole 2ch",
        mic_device: str = "",
        system_audio_pid: int = 0,
        system_audio_app: str = "",
    ) -> None:
        """Start the dual audio pipeline in background threads.

        Args:
            audio_device: System audio device (e.g. "BlackHole 2ch").
            mic_device: Microphone device (e.g. "MacBook Pro Microphone").
                If empty, falls back to config default.
            system_audio_pid: PID of app to capture via ScreenCaptureKit.
                When > 0, bypasses BlackHole and uses per-app capture.
            system_audio_app: Display name of the captured app (for logging).
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
        self._system_audio_pid = system_audio_pid
        self._system_audio_app = system_audio_app
        self._thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._thread.start()
        capture_desc = (
            f"app-tap: {system_audio_app or system_audio_pid}"
            if system_audio_pid > 0
            else f"device: {audio_device}"
        )
        logger.info(
            "Session starting — system: %s, mic: %s",
            capture_desc,
            self._mic_device,
        )

    def stop(self) -> None:
        """Stop both audio pipelines and finalize active turn.

        Audio recording is NOT saved automatically — call ``save_audio()``
        after obtaining user consent via the post-meeting dialog.
        """
        self._running = False
        if self._orchestrator:
            self._orchestrator._running = False

        # Stop mic capture
        if self._mic_capture:
            self._mic_capture.stop()

        # Flush any in-progress turn and reset dedup state
        self._transcript_buffer.flush()
        self._deduplicator.reset()

        if self._thread:
            self._thread.join(timeout=10.0)
        if self._mic_thread:
            self._mic_thread.join(timeout=10.0)

        logger.info("Session stopped after %.0fs", self.elapsed_seconds)

    def save_audio(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Save the session's audio recording to WAV.

        Call after ``stop()`` and before the session is replaced by a new
        ``start()``.  The audio buffer persists in the ``AudioCapture``
        object until then.

        Returns:
            Path to the saved WAV file, or *None* if no audio is available.
        """
        if not self._orchestrator or not hasattr(self._orchestrator, "audio"):
            return None
        out_dir = output_dir or _get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        recording_path = out_dir / f"session_{ts}.wav"
        saved = self._orchestrator.audio.save_recording(recording_path)
        if saved:
            logger.info("Audio saved to %s", recording_path)
            return recording_path
        return None

    @property
    def has_audio(self) -> bool:
        """True if audio data is available for saving."""
        if not self._orchestrator or not hasattr(self._orchestrator, "audio"):
            return False
        return hasattr(self._orchestrator.audio, "save_recording")

    def pause(self) -> None:
        """Pause audio capture. Models stay loaded, timer pauses."""
        if not self._running or self._paused:
            return
        self._paused = True
        self._pause_start = time.time()
        # Flush active turn so it finalizes cleanly
        self._transcript_buffer.flush()
        # Pause both audio captures
        if self._orchestrator and hasattr(self._orchestrator, "audio"):
            self._orchestrator.audio.pause()
        if self._mic_capture and hasattr(self._mic_capture, "pause"):
            self._mic_capture.pause()
        logger.info("Session paused at %.0fs", self.elapsed_seconds)

    def resume(self) -> None:
        """Resume audio capture after pause."""
        if not self._running or not self._paused:
            return
        self._paused = False
        if self._pause_start:
            self._total_pause_time += time.time() - self._pause_start
            self._pause_start = 0.0
        # Resume both audio captures
        if self._orchestrator and hasattr(self._orchestrator, "audio"):
            self._orchestrator.audio.resume()
        if self._mic_capture and hasattr(self._mic_capture, "resume"):
            self._mic_capture.resume()
        logger.info("Session resumed at %.0fs", self.elapsed_seconds)

    def get_status(self) -> dict:
        """Get current session status."""
        audio_health: dict = {}
        if self._orchestrator and hasattr(self._orchestrator, "audio"):
            audio_health = self._orchestrator.audio.get_audio_health()
        if self._single_device_mode:
            capture_mode = "single_device"
        elif getattr(self, "_system_audio_pid", 0) > 0:
            capture_mode = "app_tap"
        else:
            capture_mode = "device"
        return {
            "running": self._running,
            "paused": self._paused,
            "loading": self._loading,
            "elapsed_seconds": self.elapsed_seconds,
            "segment_count": self.transcript.segment_count,
            "meeting_title": (self.meeting_context.title if self.meeting_context else ""),
            "audio_health": audio_health,
            "capture_mode": capture_mode,
        }

    # --- Orchestrator callbacks (called from system audio pipeline thread) ---

    def _on_transcription(self, text: str, timestamp: float) -> None:
        """Called when valid speech is transcribed from system audio.

        Tags the chunk with source="system" (remote participants), unless
        running in single-device mode where all speech is source="mic" (You).
        Cross-stream echo suppression runs before buffering.
        """
        source = "mic" if self._single_device_mode else "system"
        dedup = self._deduplicator.check(text, source, timestamp)
        if dedup.action == "suppress":
            logger.info(
                "[dedup] Suppressed %s echo: %r (%.0f%% match with %s: %r)",
                source,
                text[:60],
                dedup.similarity * 100,
                dedup.matched_source,
                dedup.matched_text[:60],
            )
            self._on_silence_detected(timestamp, source=source)
            return
        logger.debug("[%s] Valid speech: %r", source, text[:80])
        self._transcript_buffer.add_chunk(text, timestamp, source=source)

    def _on_mic_transcription(self, text: str, timestamp: float) -> None:
        """Called when valid speech is transcribed from microphone.

        Tags the chunk with source="mic" (local user).
        Cross-stream echo suppression runs before buffering.
        """
        dedup = self._deduplicator.check(text, "mic", timestamp)
        if dedup.action == "suppress":
            logger.info(
                "[dedup] Suppressed mic echo: %r (%.0f%% match with %s: %r)",
                text[:60],
                dedup.similarity * 100,
                dedup.matched_source,
                dedup.matched_text[:60],
            )
            self._on_silence_detected(timestamp, source="mic")
            return
        logger.debug("[mic] Valid speech: %r", text[:80])
        self._transcript_buffer.add_chunk(text, timestamp, source="mic")

    def _on_silence_detected(self, timestamp: float, source: str = "") -> None:
        """Called when ASR returns empty/hallucination (silence).

        Notifies the TranscriptBuffer with the source so that system
        audio silence doesn't prematurely finalize mic turns (and vice
        versa). Source is "system" or "mic".
        """
        self._transcript_buffer.on_silence(timestamp, source=source)

    def _on_trigger_result(self, trigger: Trigger, result: GenerationResult) -> None:
        """Called when a trigger fires and produces a generation result.

        F-202: Suppresses dead-end responses — never show empty answers or
        "no_match"/"no_context"/"suppressed" results to the user.

        Note: Q&A memory is handled by the orchestrator's _handle_triggers —
        no duplicate add_qa_pair here.
        """
        # F-202: Suppress dead-end responses — silence beats "I can't help"
        _dead_end_methods = {"no_match", "no_context", "suppressed"}
        if not result.answer or result.method in _dead_end_methods:
            logger.debug(
                "[trigger] %s suppressed (method=%s, answer_len=%d)",
                trigger.type.value,
                result.method,
                len(result.answer) if result.answer else 0,
            )
            return

        logger.info(
            "[trigger] %s → %s (conf=%.2f, %dms)",
            trigger.type.value,
            result.method,
            result.confidence,
            result.latency_ms,
        )

        # Store for post-meeting summary
        self._trigger_history.append(
            {
                "trigger_type": trigger.type.value,
                "trigger_text": trigger.text,
                "answer": result.answer,
                "confidence": result.confidence,
                "timestamp": time.time(),
            }
        )

        # Resolve auto-dismiss duration from config
        _dismiss_ms_map = {
            "persistent": self.config.triggers.dismiss_persistent_ms,
            "standard": self.config.triggers.dismiss_standard_ms,
            "ephemeral": self.config.triggers.dismiss_ephemeral_ms,
        }
        persistence = trigger.type.persistence
        dismiss_ms = _dismiss_ms_map.get(persistence, 90_000)

        self._thread_safe_put(
            self._prompt_queue,
            {
                "type": "prompt",
                "trigger_type": trigger.type.value,
                "trigger_text": trigger.text,
                "answer": result.answer,
                "confidence": result.confidence,
                "method": result.method,
                "latency_ms": result.latency_ms,
                "source": result.source,
                "persistence": persistence,
                "dismiss_ms": dismiss_ms,
                "display_label": trigger.type.label,
                "display_emoji": trigger.type.emoji,
            },
        )

    # --- Turn callbacks (called from TranscriptBuffer in pipeline thread) ---

    def _on_turn_update(self, turn: Turn) -> None:
        """Called when a turn is updated (new chunk added)."""
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=False,
            source=turn.source,
        )
        self._thread_safe_put(
            self._transcript_queue,
            {
                "type": "transcript_update",
                **turn.to_dict(),
            },
        )

    def _on_turn_final(self, turn: Turn) -> None:
        """Called when a turn is finalized (pause detected).

        Speaker label is set deterministically from source (Tier 1):
        mic → "You", system → "Others". If Tier 2 diarization is enabled,
        system turns are then relabeled with neural speaker embeddings.
        """
        # L1 channel attribution (deterministic) via the resolver. Unknown
        # sources yield an empty label → keep the turn's existing speaker.
        channel = self._resolver.resolve_channel(turn.source)
        if channel.speaker:
            turn.speaker = channel.speaker

        # Emit raw finalization immediately
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=True,
            speaker=turn.speaker,
            source=turn.source,
        )
        self._thread_safe_put(
            self._transcript_queue,
            {
                "type": "transcript_final",
                **turn.to_dict(),
            },
        )

        # Polish with text refiner (thread-safe via RAGAnswerGenerator.generate_text())
        if self._text_refiner:
            t0 = time.time()
            polished = self._text_refiner.refine(turn.text)
            refine_ms = (time.time() - t0) * 1000
            if polished and polished != turn.text:
                logger.info("Turn %s refined in %.0fms", turn.id, refine_ms)
                turn.text = polished
                self.transcript.upsert(
                    seg_id=turn.id,
                    text=polished,
                    timestamp=turn.start_timestamp,
                    end_timestamp=turn.end_timestamp,
                    is_final=True,
                    speaker=turn.speaker,
                    source=turn.source,
                )
                self._thread_safe_put(
                    self._transcript_queue,
                    {
                        "type": "transcript_polished",
                        "id": turn.id,
                        "text": polished,
                        "timestamp": turn.start_timestamp,
                        "end_timestamp": turn.end_timestamp,
                        "is_final": True,
                        "speaker": turn.speaker,
                        "source": turn.source,
                    },
                )

        # Tier 2: neural speaker diarization (system audio only)
        if self._diarizer and turn.source == "system":
            self._relabel_speaker(turn)

    def _relabel_speaker(self, turn: Turn) -> None:
        """Retroactively relabel a system-audio turn via neural diarization.

        Extracts the turn's raw audio segment, computes a speaker embedding,
        and assigns a speaker label via online clustering. Updates the
        transcript store and pushes a relabeled message to the UI.
        """
        try:
            audio_segment = self._get_turn_audio(turn)
            if audio_segment is None:
                return

            diar_label = self._diarizer.process_turn(
                audio_segment,
                self.config.audio.sample_rate,
            )
            # L3/L4 + regime via the resolver (roster names, honest degradation).
            attribution = self._resolver.resolve_acoustic(diar_label, names=self._speaker_names)
            speaker = attribution.speaker
            if speaker and speaker != turn.speaker:
                turn.speaker = speaker
                turn.low_confidence = attribution.low_confidence
                self.transcript.upsert(
                    seg_id=turn.id,
                    text=turn.text,
                    timestamp=turn.start_timestamp,
                    end_timestamp=turn.end_timestamp,
                    is_final=True,
                    speaker=speaker,
                    source=turn.source,
                    low_confidence=attribution.low_confidence,
                )
                self._thread_safe_put(
                    self._transcript_queue,
                    {
                        "type": "transcript_relabeled",
                        "id": turn.id,
                        "text": turn.text,
                        "timestamp": turn.start_timestamp,
                        "end_timestamp": turn.end_timestamp,
                        "is_final": True,
                        "speaker": speaker,
                        "source": turn.source,
                        "low_confidence": attribution.low_confidence,
                    },
                )
                logger.info(
                    "Turn %s relabeled: Others → %s%s",
                    turn.id,
                    speaker,
                    " (low confidence)" if attribution.low_confidence else "",
                )
        except Exception as e:
            logger.warning("Speaker relabeling failed for %s: %s", turn.id, e)

    def rename_speaker(self, old_name: str, new_name: str) -> None:
        """Rename a speaker across all transcript segments.

        Updates the name mapping so future diarizer results also resolve
        to the custom name. Emits transcript_relabeled for each affected
        segment so the UI updates in real time.
        """
        # Update mapping: find original diarizer label that maps to old_name
        original_label = None
        for orig, custom in self._speaker_names.items():
            if custom == old_name:
                original_label = orig
                break
        if original_label:
            self._speaker_names[original_label] = new_name
        else:
            self._speaker_names[old_name] = new_name

        # Bulk rename in store
        affected_ids = self.transcript.rename_speaker(old_name, new_name)
        if not affected_ids:
            return

        # Emit relabeled messages for each affected segment
        for seg_id in affected_ids:
            idx = self.transcript._index.get(seg_id)
            if idx is None:
                continue
            seg = self.transcript._segments[idx]
            self._thread_safe_put(
                self._transcript_queue,
                {
                    "type": "transcript_relabeled",
                    "id": seg.id,
                    "text": seg.text,
                    "timestamp": seg.timestamp,
                    "end_timestamp": seg.end_timestamp or seg.timestamp,
                    "is_final": seg.is_final,
                    "speaker": new_name,
                    "source": seg.source,
                },
            )

        logger.info(
            "Renamed speaker '%s' → '%s' (%d segments)", old_name, new_name, len(affected_ids)
        )

    def _get_turn_audio(self, turn: Turn) -> "Optional[np.ndarray]":
        """Retrieve raw audio for a turn's time range from the system AudioCapture."""
        if not self._orchestrator or not hasattr(self._orchestrator, "audio"):
            return None

        audio = self._orchestrator.audio.get_audio_segment(
            turn.start_timestamp,
            turn.end_timestamp,
        )
        if audio is None or len(audio) == 0:
            return None
        return audio

    # --- Pipeline threads ---

    def _run_pipeline(self) -> None:
        """Background thread: load models, start system audio + mic capture."""
        try:
            from lib.audio_capture import AudioCapture
            from lib.orchestrator import MeetingOrchestrator

            # Build per-app capture if PID was provided
            system_capture = None
            if self._system_audio_pid > 0:
                from lib.system_audio_capture import SystemAudioCapture

                system_capture = SystemAudioCapture(
                    pid=self._system_audio_pid,
                    app_name=self._system_audio_app,
                    sample_rate=self.config.audio.sample_rate,
                )
                logger.info(
                    "Using per-app capture: %s (PID %d)",
                    self._system_audio_app,
                    self._system_audio_pid,
                )

            logger.info("Loading models in background thread...")
            self._orchestrator = MeetingOrchestrator(
                config=self.config,
                audio_device=self._audio_device,
                meeting_context_path=None,
                headless=True,
                audio_capture=system_capture,
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

            # Create text refiner (shares RAGAnswerGenerator — thread-safe via generate_text())
            refiner_config = getattr(self.config, "refiner", None)
            refiner_enabled = getattr(refiner_config, "enabled", True) if refiner_config else True
            if refiner_enabled and hasattr(self._orchestrator.generator, "_generator"):
                self._text_refiner = TextRefiner(
                    self._orchestrator.generator._generator,
                    min_words_to_refine=(
                        getattr(refiner_config, "min_words_to_refine", 5) if refiner_config else 5
                    ),
                )
                logger.info("Text refiner enabled (sharing LFM2.5-Instruct instance)")

            # --- Tier 2: speaker diarization on system audio ---
            if self.config.diarization.enabled:
                try:
                    self._diarizer = SpeakerDiarizer(self.config.diarization)
                    if self._diarizer.available:
                        logger.info("Tier 2 speaker diarization enabled")
                    else:
                        logger.warning("Diarization model unavailable — Tier 1 fallback")
                        self._diarizer = None
                except Exception as e:
                    logger.warning("Diarization init failed: %s — Tier 1 fallback", e)
                    self._diarizer = None

            # --- Mic capture: separate AudioCapture + ASR pipeline ---
            # Skip mic pipeline if same device as system audio (single-device mode).
            # In single-device mode, all speech comes from one source → tagged "mic" → "You".
            # App-tap mode is never single-device (mic and app tap are different sources).
            is_same_device = (
                self._system_audio_pid <= 0
                and self._audio_device.lower() == self._mic_device.lower()
            )
            if is_same_device:
                self._single_device_mode = True
                logger.info(
                    "Single-device mode: mic == system audio (%s). " "All speech tagged as 'You'.",
                    self._audio_device,
                )
            else:
                self._single_device_mode = False
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
            # Ensure mic capture is stopped if pipeline thread exits unexpectedly
            if self._mic_capture:
                try:
                    self._mic_capture.stop()
                except Exception:
                    pass

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
                        self._on_silence_detected(timestamp, source="mic")
                        return

                    if is_hallucination_only(text):
                        self._on_silence_detected(timestamp, source="mic")
                        return

                    # Valid speech from mic
                    self._on_mic_transcription(text, timestamp)

                    # Feed trigger pipeline (same as system audio)
                    if not is_noise(text):
                        triggers = self._orchestrator.buffer.add_chunk(text, timestamp)
                        self._handle_mic_triggers(triggers)
                except Exception as e:
                    logger.error("Mic chunk error: %s", e)

            try:
                self._mic_capture.start_stream(process_mic_chunk)
            finally:
                # Ensure mic capture is cleaned up even if start_stream raises
                if self._mic_capture:
                    try:
                        self._mic_capture.stop()
                    except Exception:
                        pass
        except Exception as e:
            logger.error("Mic pipeline error: %s", e, exc_info=True)

    def _handle_mic_triggers(self, triggers: list) -> None:
        """Handle triggers from mic audio.

        Thread-safe: generation goes through RAGAnswerGenerator.generate_text()
        which holds an internal lock.
        """
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
                logger.warning(
                    "Queue full, dropping %s event",
                    item.get("type", "unknown"),
                )
