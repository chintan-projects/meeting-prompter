"""Session manager — wraps MeetingOrchestrator for the API layer.

Bridges the audio pipeline to WebSocket consumers by collecting
transcript turns and trigger results into queues that the
WebSocket routes drain.

Key change from v1: raw ASR chunks are accumulated into speech turns
via TranscriptBuffer before being stored and streamed. This means the
UI receives coherent speech blocks instead of fragmented 4s chunks.
"""
import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from lib.config import AppConfig, load_config
from lib.conversation.meeting_context import MeetingContext, load_meeting_context
from lib.filters import is_noise
from lib.triggers.types import TriggerType

from .transcript_buffer import TranscriptBuffer, Turn
from .transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


class Session:
    """Managed meeting session that bridges audio pipeline to async API.

    The audio pipeline runs in a background thread. Transcript turns
    and trigger results are pushed into asyncio queues that WebSocket
    handlers consume. We use loop.call_soon_threadsafe to bridge the
    thread boundary safely.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.transcript = TranscriptStore()
        self.meeting_context: Optional[MeetingContext] = None

        # Turn-based buffer: accumulates raw chunks into speech turns
        self._transcript_buffer = TranscriptBuffer(
            turn_pause=2.0,
            max_turn_duration=30.0,
            min_turn_words=2,
            on_update=self._on_turn_update,
            on_final=self._on_turn_final,
        )

        # Async queues for WebSocket consumers
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._prompt_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._orchestrator: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
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
        """Start the audio pipeline in a background thread.

        Model loading happens in the background thread so we don't
        block the async event loop.
        """
        if self._running or self._loading:
            logger.warning("Session already running or loading")
            return

        # Capture the event loop so background thread can safely enqueue
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
        """Stop the audio pipeline and finalize any active turn."""
        self._running = False
        if self._orchestrator:
            self._orchestrator._running = False

        # Flush any in-progress turn so it's captured
        self._transcript_buffer.flush()

        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Session stopped after %.0fs", self.elapsed_seconds)

    def get_status(self) -> dict:
        """Get current session status."""
        audio_health: dict = {}
        if self._orchestrator and hasattr(self._orchestrator, 'audio'):
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

    # --- Turn callbacks (called from pipeline background thread) ---

    def _on_turn_update(self, turn: Turn) -> None:
        """Called when a turn is updated (new chunk added)."""
        # Upsert into store so it's always persisted
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=False,
        )
        # Push partial update to WebSocket
        self._thread_safe_put(self._transcript_queue, {
            "type": "transcript_update",
            **turn.to_dict(),
        })

    def _on_turn_final(self, turn: Turn) -> None:
        """Called when a turn is finalized (pause detected)."""
        # Mark as final in store
        self.transcript.upsert(
            seg_id=turn.id,
            text=turn.text,
            timestamp=turn.start_timestamp,
            end_timestamp=turn.end_timestamp,
            is_final=True,
        )
        # Push finalization to WebSocket
        self._thread_safe_put(self._transcript_queue, {
            "type": "transcript_final",
            **turn.to_dict(),
        })

    # --- Pipeline thread ---

    def _run_pipeline(self) -> None:
        """Background thread: load models then run the audio capture loop."""
        try:
            # Heavy model loading happens here, not in the event loop
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

            # Override process_chunk to feed our queues
            original_process = self._orchestrator.process_chunk
            self._orchestrator.process_chunk = self._wrap_process_chunk(original_process)

            # Wire audio-level silence to the transcript buffer.
            # The audio capture detects silence BEFORE the callback runs,
            # so silent chunks never reach _wrap_process_chunk. We must
            # intercept the audio capture's on_silence to notify the
            # transcript buffer about actual pauses in speech.
            original_audio_silence = self._orchestrator.audio.on_silence

            def _on_audio_silence(timestamp: float) -> None:
                self._transcript_buffer.on_silence(timestamp)
                if original_audio_silence:
                    original_audio_silence(timestamp)

            self._orchestrator.audio.on_silence = _on_audio_silence

            self._loading = False
            logger.info("Models loaded, starting audio capture...")

            self._orchestrator.run()
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            self._loading = False
            self._running = False

    def _wrap_process_chunk(self, original_fn):  # noqa: ANN001
        """Wrap orchestrator.process_chunk to intercept results for queues."""
        orch = self._orchestrator

        def wrapped(audio_path: Path, timestamp: Optional[float] = None) -> None:
            timestamp = timestamp or time.time()

            try:
                logger.info("[pipeline] Transcribing chunk: %s", audio_path)
                text = orch.lfm2.transcribe(audio_path)
                logger.info("[pipeline] Transcription result: %r", text[:100] if text else "")

                if not text or text.startswith("["):
                    logger.debug("[pipeline] Empty/error transcription, flushing buffer")
                    # Silence — notify transcript buffer for turn finalization
                    self._transcript_buffer.on_silence(timestamp)
                    triggers = orch.buffer.force_flush()
                    self._push_triggers(triggers)
                    return

                if is_noise(text):
                    logger.debug("[pipeline] Noise filtered: %r", text[:60])
                    self._transcript_buffer.on_silence(timestamp)
                    triggers = orch.buffer.force_flush()
                    self._push_triggers(triggers)
                    return

                orch.chunk_count += 1
                logger.info("[pipeline] Valid speech chunk #%d: %r", orch.chunk_count, text[:80])

                # Feed into turn-based buffer (callbacks handle store + WebSocket)
                self._transcript_buffer.add_chunk(text, timestamp)

                # Trigger detection (ConversationBuffer handles its own buffering)
                triggers = orch.buffer.add_chunk(text, timestamp)
                self._push_triggers(triggers)

            except Exception as e:
                logger.error("Chunk processing error: %s", e, exc_info=True)

        return wrapped

    def _thread_safe_put(self, queue: asyncio.Queue, item: dict) -> None:
        """Put an item on an asyncio.Queue from a background thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
        else:
            # Fallback: direct put (may not wake waiters across threads)
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    def _push_triggers(self, triggers: list) -> None:
        """Process triggers and push results to prompt queue."""
        if not self._orchestrator:
            return

        for trigger in triggers:
            logger.info(
                "[trigger] %s fired: %r (conf=%.2f)",
                trigger.type.value, trigger.text[:60], trigger.confidence,
            )
            result = self._orchestrator._process_trigger(trigger)
            if not result or not result.answer:
                logger.debug("[trigger] %s -> no result", trigger.type.value)
                continue
            if result.answer.startswith("["):
                logger.debug(
                    "[trigger] %s -> suppressed: %s",
                    trigger.type.value, result.answer[:60],
                )
                continue

            logger.info(
                "[trigger] %s -> %s (conf=%.2f, %dms)",
                trigger.type.value, result.method, result.confidence, result.latency_ms,
            )
            self._orchestrator.dashboard.add_result(result)

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
