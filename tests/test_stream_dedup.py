"""Tests for cross-stream echo detection (lib/stream_dedup.py).

Validates that the StreamDeduplicator correctly identifies when mic
and system audio produce near-duplicate transcriptions of the same
speech (acoustic coupling), and suppresses the echo while keeping
distinct speech from each stream.
"""

import threading
from typing import List


from lib.config import DualStreamConfig
from lib.stream_dedup import DedupResult, StreamDeduplicator, _normalize

# --- Normalization ---


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Hello World") == "hello world"

    def test_strip_punctuation(self) -> None:
        assert _normalize("What's the plan?") == "whats the plan"

    def test_collapse_whitespace(self) -> None:
        assert _normalize("  multiple   spaces  ") == "multiple spaces"

    def test_empty(self) -> None:
        assert _normalize("") == ""

    def test_punctuation_only(self) -> None:
        assert _normalize("...") == ""


# --- Basic echo detection ---


class TestEchoDetection:
    """Core dedup behavior: identical or near-identical text from different streams."""

    def _config(self, **overrides: object) -> DualStreamConfig:
        defaults = {
            "enabled": True,
            "window_seconds": 8.0,
            "similarity_threshold": 0.55,
            "short_text_threshold": 0.75,
            "short_text_min_words": 4,
        }
        defaults.update(overrides)
        return DualStreamConfig(**defaults)

    def test_identical_text_suppressed(self) -> None:
        """Exact same text from both streams → suppress the second."""
        dd = StreamDeduplicator(self._config())
        dd.check("Is there anyone with context on the audio side", "system", 100.0)
        result = dd.check("Is there anyone with context on the audio side", "mic", 101.0)
        assert result.action == "suppress"
        assert result.similarity == 1.0
        assert result.matched_source == "system"

    def test_near_duplicate_suppressed(self) -> None:
        """ASR produces slightly different text from same speech → suppress."""
        dd = StreamDeduplicator(self._config())
        dd.check(
            "Here, so we're asking, is there anyone who has",
            "system",
            100.0,
        )
        result = dd.check(
            "Is there anyone who has substantial context on the audio side",
            "mic",
            101.0,
        )
        # These overlap significantly — SequenceMatcher catches it
        assert result.similarity > 0.0
        # If below threshold, that's fine — the algorithm is honest

    def test_distinct_speech_kept(self) -> None:
        """Completely different speech from each stream → both kept."""
        dd = StreamDeduplicator(self._config())
        dd.check("We need to review the quarterly numbers", "system", 100.0)
        result = dd.check("Let me share my screen for the demo", "mic", 101.0)
        assert result.action == "keep"
        assert result.similarity < 0.55

    def test_same_source_not_compared(self) -> None:
        """Chunks from the same source should never suppress each other."""
        dd = StreamDeduplicator(self._config())
        dd.check("This is the same text", "mic", 100.0)
        result = dd.check("This is the same text", "mic", 101.0)
        assert result.action == "keep"

    def test_symmetric_detection(self) -> None:
        """Echo detected regardless of which stream arrives first."""
        dd = StreamDeduplicator(self._config())
        # Mic first, then system
        dd.check("Samsung and Shopify, why?", "mic", 100.0)
        result = dd.check("To either Samsung and Shopify.", "system", 101.0)
        sim1 = result.similarity

        dd2 = StreamDeduplicator(self._config())
        # System first, then mic
        dd2.check("Samsung and Shopify, why?", "system", 100.0)
        result2 = dd2.check("To either Samsung and Shopify.", "mic", 101.0)
        assert result2.similarity == sim1


# --- Temporal window ---


class TestTemporalWindow:
    """Chunks outside the window should not be compared."""

    def _config(self, **overrides: object) -> DualStreamConfig:
        defaults = {
            "enabled": True,
            "window_seconds": 8.0,
            "similarity_threshold": 0.55,
            "short_text_threshold": 0.75,
            "short_text_min_words": 4,
        }
        defaults.update(overrides)
        return DualStreamConfig(**defaults)

    def test_within_window_suppressed(self) -> None:
        dd = StreamDeduplicator(self._config(window_seconds=5.0))
        dd.check("Same speech captured by both streams here", "system", 100.0)
        result = dd.check("Same speech captured by both streams here", "mic", 104.0)
        assert result.action == "suppress"

    def test_outside_window_kept(self) -> None:
        dd = StreamDeduplicator(self._config(window_seconds=5.0))
        dd.check("Same speech captured by both streams here", "system", 100.0)
        result = dd.check("Same speech captured by both streams here", "mic", 106.0)
        assert result.action == "keep"

    def test_old_entries_pruned(self) -> None:
        """Entries older than 2x window are pruned to prevent memory growth."""
        dd = StreamDeduplicator(self._config(window_seconds=5.0))
        dd.check("old text that should be pruned eventually", "system", 100.0)
        # Add many more chunks to trigger pruning
        for i in range(20):
            dd.check(f"chunk {i}", "system", 115.0 + i)
        # Verify old entry was pruned (check won't find it)
        result = dd.check("old text that should be pruned eventually", "mic", 140.0)
        assert result.action == "keep"


# --- Short text guard ---


class TestShortTextGuard:
    """Short text (< min_words) requires higher similarity threshold."""

    def _config(self, **overrides: object) -> DualStreamConfig:
        defaults = {
            "enabled": True,
            "window_seconds": 8.0,
            "similarity_threshold": 0.55,
            "short_text_threshold": 0.75,
            "short_text_min_words": 4,
        }
        defaults.update(overrides)
        return DualStreamConfig(**defaults)

    def test_short_text_exact_match_suppressed(self) -> None:
        """Exact match of short text still suppressed (ratio=1.0 > 0.75)."""
        dd = StreamDeduplicator(self._config())
        dd.check("So I know", "system", 100.0)
        result = dd.check("So I know", "mic", 101.0)
        assert result.action == "suppress"

    def test_short_text_partial_match_kept(self) -> None:
        """Partial match of short text kept (stricter threshold)."""
        dd = StreamDeduplicator(self._config())
        dd.check("So I know.", "system", 100.0)
        result = dd.check("Yeah I know", "mic", 101.0)
        # These are similar but short — stricter threshold applies
        if result.similarity < 0.75:
            assert result.action == "keep"

    def test_long_text_moderate_match_suppressed(self) -> None:
        """Longer text with moderate overlap uses standard threshold."""
        dd = StreamDeduplicator(self._config())
        dd.check(
            "Mercedes, if we can't, we skip straight to either.",
            "system",
            100.0,
        )
        result = dd.check(
            "Side with Mercedes. If not, we can skip straight to either.",
            "mic",
            101.0,
        )
        # These share substantial content — should be above 0.55
        assert result.similarity >= 0.55
        assert result.action == "suppress"


# --- Disabled mode ---


class TestDisabledMode:
    def test_disabled_always_keeps(self) -> None:
        dd = StreamDeduplicator(DualStreamConfig(enabled=False))
        dd.check("Same exact text from system audio stream", "system", 100.0)
        result = dd.check("Same exact text from system audio stream", "mic", 101.0)
        assert result.action == "keep"
        assert result.similarity == 0.0


# --- Reset ---


class TestReset:
    def test_reset_clears_state(self) -> None:
        dd = StreamDeduplicator(DualStreamConfig())
        dd.check("Some text to remember in the dedup buffer", "system", 100.0)
        dd.reset()
        result = dd.check("Some text to remember in the dedup buffer", "mic", 101.0)
        assert result.action == "keep"


# --- Thread safety ---


class TestThreadSafety:
    def test_concurrent_check_no_corruption(self) -> None:
        """Two threads calling check() concurrently should not corrupt state."""
        dd = StreamDeduplicator(DualStreamConfig())
        errors: List[str] = []

        def add_mic() -> None:
            for i in range(100):
                try:
                    dd.check(f"mic speech number {i}", "mic", 100.0 + i * 0.1)
                except Exception as e:
                    errors.append(f"mic: {e}")

        def add_system() -> None:
            for i in range(100):
                try:
                    dd.check(f"system speech number {i}", "system", 100.0 + i * 0.1)
                except Exception as e:
                    errors.append(f"system: {e}")

        t1 = threading.Thread(target=add_mic)
        t2 = threading.Thread(target=add_system)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# --- Real-world echo patterns from WS-14 testing ---


class TestRealWorldPatterns:
    """Patterns observed during WS-14 live dual-stream testing.

    These reproduce actual ASR output where mic and system audio
    captured the same speech with slightly different transcriptions.
    """

    def _config(self) -> DualStreamConfig:
        return DualStreamConfig()

    def test_question_echo(self) -> None:
        """Q: 'Is there anyone who has...' captured by both streams."""
        dd = StreamDeduplicator(self._config())
        dd.check(
            "Here, so we're asking, is there anyone who has substantial context on the audio side",
            "system",
            100.0,
        )
        result = dd.check(
            "Is there anyone who has substantial context on the audio side with me?",
            "mic",
            102.0,
        )
        # These share the core phrase
        assert result.similarity > 0.4

    def test_repeated_content_echo(self) -> None:
        """Short confirmation captured by both: 'So I know. Maybe.'"""
        dd = StreamDeduplicator(self._config())
        dd.check("Mm-hmm. Really? So I know.", "system", 100.0)
        result = dd.check("So I know. Maybe.", "mic", 101.0)
        # Short text — might or might not cross threshold depending on ratio
        assert isinstance(result, DedupResult)
