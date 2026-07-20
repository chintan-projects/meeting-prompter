"""Cross-stream deduplication for dual audio pipelines.

When mic and system audio both capture the same speech (acoustic coupling),
two ASR passes produce near-duplicate transcriptions tagged with different
sources. This module detects those duplicates and suppresses the echo.

Algorithm:
    1. Normalize text (lowercase, strip punctuation, collapse whitespace).
    2. Compare against the OTHER source's recent chunks using
       difflib.SequenceMatcher — handles ASR variation (insertions,
       deletions, reorderings) better than raw word-set overlap.
    3. Apply temporal window: only compare chunks within window_seconds.
    4. Short-text guard: require higher threshold for < N words.

Thread safety: check() is called from both mic and system audio threads.
All state is guarded by a threading.Lock.
"""

import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Deque, Dict, Literal, Tuple

from lib.config import DualStreamConfig

logger = logging.getLogger(__name__)

# Pre-compiled pattern for punctuation stripping
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class DedupResult:
    """Result of a deduplication check."""

    action: Literal["keep", "suppress"]
    similarity: float  # 0.0–1.0
    matched_text: str  # other-stream text that triggered suppression (empty if keep)
    matched_source: str  # other stream's source tag (empty if keep)


def _normalize(text: str) -> str:
    """Normalize text for similarity comparison.

    Lowercase, strip punctuation, collapse whitespace.
    """
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


class StreamDeduplicator:
    """Detects and suppresses cross-stream echo in dual audio pipelines.

    Maintains a rolling window of recent transcriptions per source.
    When a new chunk arrives, compares it against the OTHER source's
    recent chunks using sequence similarity. If the similarity exceeds
    the configured threshold, the chunk is flagged for suppression.

    Args:
        config: Dual-stream configuration (thresholds, window size).
    """

    def __init__(self, config: DualStreamConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        # source → deque of (normalized_text, raw_text, timestamp)
        self._recent: Dict[str, Deque[Tuple[str, str, float]]] = {}

    def check(self, text: str, source: str, timestamp: float) -> DedupResult:
        """Check if a transcription is a cross-stream echo.

        Always records the chunk for future comparisons, then checks
        against the other source's recent window.

        Args:
            text: Raw transcription text.
            source: Audio source tag ("mic" or "system").
            timestamp: Chunk timestamp.

        Returns:
            DedupResult indicating whether to keep or suppress.
        """
        if not self._config.enabled:
            return DedupResult(action="keep", similarity=0.0, matched_text="", matched_source="")

        normalized = _normalize(text)
        if not normalized:
            return DedupResult(action="keep", similarity=0.0, matched_text="", matched_source="")

        with self._lock:
            # Record this chunk
            self._record(normalized, text, source, timestamp)

            # Find best match from other sources
            best_similarity = 0.0
            best_text = ""
            best_source = ""

            for other_source, chunks in self._recent.items():
                if other_source == source:
                    continue
                for other_norm, other_raw, other_ts in chunks:
                    if abs(timestamp - other_ts) > self._config.window_seconds:
                        continue
                    sim = SequenceMatcher(None, normalized, other_norm).ratio()
                    if sim > best_similarity:
                        best_similarity = sim
                        best_text = other_raw
                        best_source = other_source

        # Determine threshold: stricter for short text
        word_count = len(normalized.split())
        threshold = (
            self._config.short_text_threshold
            if word_count < self._config.short_text_min_words
            else self._config.similarity_threshold
        )

        if best_similarity >= threshold:
            logger.debug(
                "[dedup] %s echo detected: %.0f%% similar to %s chunk",
                source,
                best_similarity * 100,
                best_source,
            )
            return DedupResult(
                action="suppress",
                similarity=best_similarity,
                matched_text=best_text,
                matched_source=best_source,
            )

        return DedupResult(
            action="keep",
            similarity=best_similarity,
            matched_text="",
            matched_source="",
        )

    def reset(self) -> None:
        """Clear all state (e.g., on session restart)."""
        with self._lock:
            self._recent.clear()

    def _record(self, normalized: str, raw: str, source: str, timestamp: float) -> None:
        """Record a chunk and prune expired entries. Caller must hold lock."""
        if source not in self._recent:
            self._recent[source] = deque(maxlen=50)

        self._recent[source].append((normalized, raw, timestamp))

        # Prune entries older than 2x window (keep some buffer for edge cases)
        cutoff = timestamp - (self._config.window_seconds * 2)
        for src_deque in self._recent.values():
            while src_deque and src_deque[0][2] < cutoff:
                src_deque.popleft()
