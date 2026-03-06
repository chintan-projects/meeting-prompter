"""Turn-based transcript buffer.

Accumulates raw ASR chunks into coherent speech turns by detecting
turn boundaries via silence events from the audio capture layer.

Key design insight: The ASR pipeline processes audio in ~4-second chunks,
so consecutive speech chunks arrive ~3.5 seconds apart even during continuous
speech. Time gaps between add_chunk() calls do NOT indicate pauses.
Instead, turn boundaries are detected by actual silence in the audio stream,
reported via on_silence() from the audio capture layer.

Flow:
    Audio Capture → speech chunk → add_chunk() → extends active turn
    Audio Capture → silence      → on_silence() → finalizes active turn
    Next speech chunk             → add_chunk() → new turn starts

Callbacks fire on every update (partial) and on finalize (complete).
"""
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TURN_PAUSE = 2.0
DEFAULT_MAX_TURN_DURATION = 30.0
DEFAULT_MIN_TURN_WORDS = 2


@dataclass
class Turn:
    """A speech turn — one continuous block of speech."""

    id: str
    text: str
    start_timestamp: float
    end_timestamp: float
    is_final: bool = False
    speaker: str = ""
    chunk_count: int = 0

    def to_dict(self) -> dict:
        """Serialize for WebSocket transmission."""
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "is_final": self.is_final,
            "speaker": self.speaker,
        }


TurnCallback = Callable[[Turn], None]


class TranscriptBuffer:
    """Accumulates ASR chunks into speech turns using silence detection.

    Each raw chunk (typically 4s of audio) is added via add_chunk().
    The buffer merges ALL consecutive speech chunks into a single turn.
    Turn boundaries are detected ONLY by:

    1. Silence events (on_silence) — the audio capture layer reports that
       audio dropped below the speech threshold.
    2. Max duration — safety limit to prevent unbounded turns.

    Time gaps between add_chunk() calls are NOT used for boundary detection
    because the ~3.5s gap between chunks is a pipeline artifact, not a
    speech characteristic.

    Emits two event types via callbacks:
    - on_update: called each time a chunk extends the active turn (partial)
    - on_final: called when a turn is finalized (silence or max duration)

    Thread safety: This class is NOT thread-safe. The caller (Session)
    must ensure add_chunk/on_silence/flush are called from the same thread
    (the pipeline background thread).
    """

    def __init__(
        self,
        turn_pause: float = DEFAULT_TURN_PAUSE,
        max_turn_duration: float = DEFAULT_MAX_TURN_DURATION,
        min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
        on_update: Optional[TurnCallback] = None,
        on_final: Optional[TurnCallback] = None,
    ) -> None:
        self._turn_pause = turn_pause
        self._max_turn_duration = max_turn_duration
        self._min_turn_words = min_turn_words
        self._on_update = on_update
        self._on_final = on_final

        self._turn_counter: int = 0
        self._active_turn: Optional[Turn] = None
        self._finalized_turns: List[Turn] = []

        # Set by on_silence() to signal a pause was detected.
        # Checked by add_chunk() to finalize before starting/extending.
        self._silence_seen: bool = False

    @property
    def active_turn(self) -> Optional[Turn]:
        """The currently accumulating turn, or None."""
        return self._active_turn

    @property
    def all_turns(self) -> List[Turn]:
        """All turns: finalized + active (if any)."""
        result = list(self._finalized_turns)
        if self._active_turn:
            result.append(self._active_turn)
        return result

    @property
    def turn_count(self) -> int:
        """Total number of turns (finalized + active)."""
        return len(self._finalized_turns) + (1 if self._active_turn else 0)

    def add_chunk(self, text: str, timestamp: float) -> Optional[Turn]:
        """Add a transcribed text chunk. Returns the active turn if updated.

        If silence was detected since the last speech chunk (via on_silence),
        the active turn is finalized first, then a new turn begins.
        Max duration also triggers finalization.

        Time gaps between add_chunk calls are NOT used for turn detection
        because the ASR pipeline cadence (~3.5s) is longer than any
        reasonable pause threshold.
        """
        text = text.strip()
        if not text:
            return None

        # Check if we need to finalize the active turn first
        if self._active_turn is not None:
            duration = timestamp - self._active_turn.start_timestamp

            # Finalize if silence was detected or max duration exceeded
            if self._silence_seen or duration >= self._max_turn_duration:
                self._finalize_active()

        # Reset silence flag — we have speech now
        self._silence_seen = False

        # Start a new turn or extend the active one
        if self._active_turn is None:
            self._turn_counter += 1
            self._active_turn = Turn(
                id=f"turn-{self._turn_counter}",
                text=text,
                start_timestamp=timestamp,
                end_timestamp=timestamp,
                chunk_count=1,
            )
        else:
            self._active_turn.text = f"{self._active_turn.text} {text}"
            self._active_turn.end_timestamp = timestamp
            self._active_turn.chunk_count += 1

        if self._on_update:
            self._on_update(self._active_turn)

        return self._active_turn

    def flush(self) -> Optional[Turn]:
        """Force-finalize the active turn (e.g., on session stop)."""
        if self._active_turn is not None:
            return self._finalize_active()
        return None

    def on_silence(self, timestamp: float) -> Optional[Turn]:
        """Called when silence is detected in the audio stream.

        Sets a flag so the next add_chunk() knows a pause occurred.
        Also immediately finalizes the active turn if the gap since
        the last speech exceeds turn_pause — this ensures the UI
        updates promptly rather than waiting for the next speech chunk.
        """
        self._silence_seen = True

        if self._active_turn is None:
            return None

        gap = timestamp - self._active_turn.end_timestamp
        if gap >= self._turn_pause:
            return self._finalize_active()
        return None

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._active_turn = None
        self._finalized_turns = []
        self._turn_counter = 0
        self._silence_seen = False

    def _finalize_active(self) -> Optional[Turn]:
        """Finalize the active turn and move it to the finalized list."""
        turn = self._active_turn
        if turn is None:
            return None

        word_count = len(turn.text.split())
        if word_count < self._min_turn_words:
            logger.debug(
                "Discarding short turn %s: %r (%d words)",
                turn.id, turn.text, word_count,
            )
            self._active_turn = None
            return None

        turn.is_final = True
        self._finalized_turns.append(turn)
        self._active_turn = None

        logger.debug(
            "Finalized %s: %r (%d chunks, %d words)",
            turn.id, turn.text[:80], turn.chunk_count, word_count,
        )

        if self._on_final:
            self._on_final(turn)

        return turn
