"""Rolling conversation buffer with trigger integration.

Replaces question_buffer.py with a richer model that:
- Maintains a rolling transcript window (default 90s)
- Routes text through the trigger engine
- Tracks speech segments with timestamps
- Provides conversation context for generation prompts
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lib.config import BufferConfig
from lib.triggers.engine import TriggerEngine
from lib.triggers.question_trigger import score_question
from lib.triggers.types import Trigger

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    text: str
    timestamp: float
    is_question: bool = False


class ConversationBuffer:
    """Rolling conversation buffer with trigger-based event detection.

    Thread-safe. Audio capture runs in a separate thread, so all public
    methods are guarded by a lock.

    Args:
        config: Buffer timing/threshold configuration.
        trigger_engine: Multi-mode trigger engine for event detection.
        window_seconds: How far back the rolling window extends (default 90s).
    """

    def __init__(
        self,
        config: BufferConfig,
        trigger_engine: TriggerEngine,
        window_seconds: float = 90.0,
    ) -> None:
        self.config = config
        self.trigger_engine = trigger_engine
        self.window_seconds = window_seconds

        self._lock = threading.Lock()
        self._segments: List[TranscriptSegment] = []
        self._current_chunks: List[Tuple[str, float]] = []
        self._segment_start: float = 0.0
        self._last_speech: float = 0.0
        self._is_buffering: bool = False
        self._qa_history: List[Tuple[str, str]] = []  # (question, answer) pairs

    def add_chunk(self, text: str, timestamp: float) -> List[Trigger]:
        """Add transcribed text chunk. Returns any triggers that fired.

        This is the main entry point, called for each transcribed audio chunk.
        """
        with self._lock:
            self._last_speech = timestamp

            # Check if we should flush the current segment first
            triggers: List[Trigger] = []
            if self._is_buffering and self._should_flush(timestamp):
                segment_triggers = self._flush_segment()
                triggers.extend(segment_triggers)

            # Start buffering if this looks like question start or is substantive
            if not self._is_buffering:
                score = score_question(text)
                if score >= self.config.min_start_score or len(text.split()) >= 5:
                    self._is_buffering = True
                    self._segment_start = timestamp
                    self._current_chunks = []

            if self._is_buffering:
                self._current_chunks.append((text, timestamp))

                # Check if the accumulated text forms a complete question
                segment_text = self._get_segment_text()
                if self._is_complete(segment_text):
                    segment_triggers = self._flush_segment()
                    triggers.extend(segment_triggers)
            else:
                # Not buffering — still add to transcript and check triggers
                self._segments.append(TranscriptSegment(text=text, timestamp=timestamp))
                self._prune_window()

                # Evaluate triggers on individual chunks too
                context = self.get_recent_context_unlocked()
                chunk_triggers = self.trigger_engine.evaluate(text, context)
                if not chunk_triggers:
                    # No trigger fired — record as statement for follow-up
                    self.trigger_engine.on_statement(text, timestamp)
                triggers.extend(chunk_triggers)

            return triggers

    def on_silence(self, timestamp: float) -> List[Trigger]:
        """Called when silence is detected in the audio stream."""
        triggers: List[Trigger] = []
        with self._lock:
            if self._is_buffering and self._should_flush(timestamp):
                triggers.extend(self._flush_segment())

            # Check follow-up trigger on pause
            followup = self.trigger_engine.on_pause(timestamp)
            if followup:
                triggers.append(followup)

        return triggers

    def force_flush(self) -> List[Trigger]:
        """Force-flush the current segment buffer."""
        with self._lock:
            if self._is_buffering:
                return self._flush_segment()
        return []

    def get_recent_context(self, seconds: Optional[float] = None) -> str:
        """Get transcript text from the rolling window."""
        with self._lock:
            return self.get_recent_context_unlocked(seconds)

    def get_recent_context_unlocked(self, seconds: Optional[float] = None) -> str:
        """Get transcript (must hold lock)."""
        window = seconds or self.window_seconds
        cutoff = time.time() - window
        texts = [s.text for s in self._segments if s.timestamp >= cutoff]
        return " ".join(texts)

    def add_qa_pair(self, question: str, answer: str) -> None:
        """Record a Q&A pair for context in future prompts."""
        with self._lock:
            self._qa_history.append((question, answer))
            # Keep last 5 Q&A pairs
            if len(self._qa_history) > 5:
                self._qa_history = self._qa_history[-5:]

    def get_qa_history(self) -> List[Tuple[str, str]]:
        """Get recent Q&A pairs for context."""
        with self._lock:
            return list(self._qa_history)

    @property
    def is_buffering(self) -> bool:
        with self._lock:
            return self._is_buffering

    def get_status(self) -> dict:
        """Get current buffer status for display."""
        with self._lock:
            return {
                "is_buffering": self._is_buffering,
                "text_preview": self._get_segment_text() if self._is_buffering else "",
                "segment_count": len(self._segments),
                "window_text_length": len(self.get_recent_context_unlocked()),
            }

    # --- Private methods (caller must hold lock) ---

    def _should_flush(self, current_time: float) -> bool:
        """Check if current segment should be flushed."""
        if not self._current_chunks:
            return False

        pause = current_time - self._last_speech
        if pause >= self.config.pause_threshold:
            return True

        duration = current_time - self._segment_start
        if duration >= self.config.max_buffer_time:
            return True

        return False

    def _flush_segment(self) -> List[Trigger]:
        """Flush current segment, evaluate triggers, add to transcript."""
        text = self._get_segment_text()
        timestamp = self._segment_start
        self._is_buffering = False
        self._current_chunks = []

        if not text or len(text.split()) < self.config.min_words:
            return []

        # Add to transcript history
        is_q = score_question(text) >= self.config.confidence_threshold
        self._segments.append(TranscriptSegment(
            text=text, timestamp=timestamp, is_question=is_q,
        ))
        self._prune_window()

        # Evaluate all triggers
        context = self.get_recent_context_unlocked()
        triggers = self.trigger_engine.evaluate(text, context)

        # If no triggers fired, record as statement for follow-up
        if not triggers:
            self.trigger_engine.on_statement(text, timestamp)

        return triggers

    def _get_segment_text(self) -> str:
        """Join current segment chunks into a single string."""
        return " ".join(chunk for chunk, _ in self._current_chunks).strip()

    def _is_complete(self, text: str) -> bool:
        """Check if buffered text forms a complete utterance."""
        words = text.split()
        if len(words) < self.config.min_words:
            return False

        score = score_question(text)

        if score >= 0.6:
            return True
        if score >= self.config.confidence_threshold and len(words) >= 8:
            return True
        return False

    def _prune_window(self) -> None:
        """Remove segments older than the rolling window."""
        cutoff = time.time() - self.window_seconds
        self._segments = [s for s in self._segments if s.timestamp >= cutoff]
