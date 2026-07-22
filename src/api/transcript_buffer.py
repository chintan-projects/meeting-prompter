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

Thread safety: All public methods are guarded by a lock so that mic and
system audio pipeline threads can call add_chunk/on_silence concurrently.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TURN_PAUSE = 2.0
DEFAULT_MAX_TURN_DURATION = 30.0
DEFAULT_MIN_TURN_WORDS = 0


@dataclass
class Turn:
    """A speech turn — one continuous block of speech."""

    id: str
    text: str
    start_timestamp: float
    end_timestamp: float
    is_final: bool = False
    speaker: str = ""
    source: str = ""  # "mic" or "system" — identifies audio stream origin
    chunk_count: int = 0
    low_confidence: bool = False  # speaker label is a flagged best-effort guess (F-606)

    def to_dict(self) -> dict:
        """Serialize for WebSocket transmission."""
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "is_final": self.is_final,
            "speaker": self.speaker,
            "source": self.source,
            "low_confidence": self.low_confidence,
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

    Thread safety: All public methods are guarded by a threading.Lock so
    that mic and system audio threads can safely call add_chunk/on_silence
    concurrently. Callbacks fire inside the lock to maintain ordering.
    """

    def __init__(
        self,
        turn_pause: float = DEFAULT_TURN_PAUSE,
        max_turn_duration: float = DEFAULT_MAX_TURN_DURATION,
        min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
        on_update: Optional[TurnCallback] = None,
        on_final: Optional[TurnCallback] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._turn_pause = turn_pause
        self._max_turn_duration = max_turn_duration
        self._min_turn_words = min_turn_words
        self._on_update = on_update
        self._on_final = on_final

        self._turn_counter: int = 0
        self._active_turn: Optional[Turn] = None
        self._finalized_turns: List[Turn] = []

        # Per-source silence flags: set by on_silence(source=...) so that
        # system audio silence doesn't prematurely finalize mic turns and
        # vice versa. Key "" is the global fallback for sourceless callers.
        self._silence_seen: dict[str, bool] = {}

    @property
    def active_turn(self) -> Optional[Turn]:
        """The currently accumulating turn, or None."""
        return self._active_turn

    @property
    def all_turns(self) -> List[Turn]:
        """All turns: finalized + active (if any)."""
        with self._lock:
            result = list(self._finalized_turns)
            if self._active_turn:
                result.append(self._active_turn)
            return result

    @property
    def turn_count(self) -> int:
        """Total number of turns (finalized + active)."""
        with self._lock:
            return len(self._finalized_turns) + (1 if self._active_turn else 0)

    def add_chunk(
        self,
        text: str,
        timestamp: float,
        source: str = "",
    ) -> Optional[Turn]:
        """Add a transcribed text chunk. Returns the active turn if updated.

        If silence was detected since the last speech chunk (via on_silence),
        the active turn is finalized first, then a new turn begins.
        Max duration also triggers finalization. Source change (mic vs system)
        also triggers a new turn so speakers don't merge.

        Time gaps between add_chunk calls are NOT used for turn detection
        because the ASR pipeline cadence (~3.5s) is longer than any
        reasonable pause threshold.
        """
        text = text.strip()
        if not text:
            return None

        with self._lock:
            # Check if we need to finalize the active turn first
            if self._active_turn is not None:
                duration = timestamp - self._active_turn.start_timestamp
                source_changed = (
                    source and self._active_turn.source and (source != self._active_turn.source)
                )

                # Check silence for the active turn's source specifically,
                # falling back to global ("") for sourceless callers.
                active_src = self._active_turn.source
                silence_for_source = self._silence_seen.get(
                    active_src, self._silence_seen.get("", False)
                )

                # Finalize if silence, max duration, or source change
                if silence_for_source or duration >= self._max_turn_duration or source_changed:
                    self._finalize_active()

            # Reset silence flags for this source — we have speech now
            if source:
                self._silence_seen[source] = False
            else:
                self._silence_seen[""] = False

            # Start a new turn or extend the active one
            if self._active_turn is None:
                self._turn_counter += 1
                self._active_turn = Turn(
                    id=f"turn-{self._turn_counter}",
                    text=text,
                    start_timestamp=timestamp,
                    end_timestamp=timestamp,
                    source=source,
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
        with self._lock:
            if self._active_turn is not None:
                return self._finalize_active()
            return None

    def on_silence(self, timestamp: float, source: str = "") -> Optional[Turn]:
        """Called when silence is detected in the audio stream.

        Sets a per-source flag so the next add_chunk() from the SAME
        source knows a pause occurred. This prevents system audio silence
        from prematurely finalizing mic turns and vice versa.

        Also immediately finalizes the active turn if the gap exceeds
        turn_pause AND the silence source matches the active turn.
        """
        with self._lock:
            key = source or ""
            self._silence_seen[key] = True

            if self._active_turn is None:
                return None

            # Only finalize if silence matches the active turn's source
            # (or if either is sourceless for backward compatibility)
            active_src = self._active_turn.source
            source_matches = not source or not active_src or source == active_src

            if source_matches:
                gap = timestamp - self._active_turn.end_timestamp
                if gap >= self._turn_pause:
                    return self._finalize_active()
            return None

    def reset(self) -> None:
        """Clear all state for a new session."""
        with self._lock:
            self._active_turn = None
            self._finalized_turns = []
            self._turn_counter = 0
            self._silence_seen = {}

    def _finalize_active(self) -> Optional[Turn]:
        """Finalize the active turn and move it to the finalized list.

        All turns are emitted regardless of length. Short utterances
        like "Yeah." or "Okay." are valid speech and should reach the UI.

        Note: Caller must hold self._lock.
        """
        turn = self._active_turn
        if turn is None:
            return None

        turn.is_final = True
        self._finalized_turns.append(turn)
        self._active_turn = None

        logger.debug(
            "Finalized %s: %r (%d chunks, %d words)",
            turn.id,
            turn.text[:80],
            turn.chunk_count,
            len(turn.text.split()),
        )

        if self._on_final:
            self._on_final(turn)

        return turn
