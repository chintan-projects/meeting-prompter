"""Tests for lib.diarization — speaker embedding clustering without real models."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.config import DiarizationConfig
from lib.diarization import SpeakerDiarizer, _SPEAKER_LABELS


@pytest.fixture
def config() -> DiarizationConfig:
    """Default diarization config."""
    return DiarizationConfig(enabled=True, max_speakers=4, similarity_threshold=0.65)


@pytest.fixture
def diarizer(config: DiarizationConfig) -> SpeakerDiarizer:
    """SpeakerDiarizer with mocked model (no speechbrain dependency)."""
    with patch.object(SpeakerDiarizer, "_load_model"):
        d = SpeakerDiarizer(config)
        d._classifier = MagicMock()  # Fake classifier so .available is True
    return d


def _make_embedding(seed: int, dim: int = 192) -> np.ndarray:
    """Create a deterministic unit-norm embedding from a seed."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestCosineSimiliarity:
    """Tests for the static cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        """Identical vectors → similarity 1.0."""
        v = _make_embedding(42)
        assert SpeakerDiarizer._cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors → similarity 0.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert SpeakerDiarizer._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors → similarity -1.0."""
        v = _make_embedding(7)
        assert SpeakerDiarizer._cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self) -> None:
        """Zero vector should return 0.0 (no division by zero)."""
        z = np.zeros(192, dtype=np.float32)
        v = _make_embedding(1)
        assert SpeakerDiarizer._cosine_similarity(z, v) == 0.0
        assert SpeakerDiarizer._cosine_similarity(v, z) == 0.0


class TestAssignSpeaker:
    """Tests for online centroid clustering via _assign_speaker."""

    def test_first_embedding_creates_speaker_a(self, diarizer: SpeakerDiarizer) -> None:
        """First embedding should always create Speaker A."""
        emb = _make_embedding(1)
        label = diarizer._assign_speaker(emb)
        assert label == "Speaker A"
        assert diarizer.speaker_count == 1

    def test_same_embedding_returns_same_speaker(self, diarizer: SpeakerDiarizer) -> None:
        """Identical embedding should match the same speaker."""
        emb = _make_embedding(1)
        label1 = diarizer._assign_speaker(emb)
        label2 = diarizer._assign_speaker(emb)
        assert label1 == label2 == "Speaker A"
        assert diarizer.speaker_count == 1

    def test_different_embedding_creates_new_speaker(self, diarizer: SpeakerDiarizer) -> None:
        """Sufficiently different embedding should create a new speaker."""
        emb_a = _make_embedding(1)
        emb_b = _make_embedding(99)  # Very different seed → different direction in 192-dim

        label_a = diarizer._assign_speaker(emb_a)
        label_b = diarizer._assign_speaker(emb_b)

        assert label_a == "Speaker A"
        assert label_b == "Speaker B"
        assert diarizer.speaker_count == 2

    def test_max_speakers_cap(self, diarizer: SpeakerDiarizer) -> None:
        """After hitting max_speakers, new embeddings assign to closest existing."""
        # Create max_speakers (4) distinct speakers
        for seed in [1, 99, 200, 300]:
            diarizer._assign_speaker(_make_embedding(seed))

        assert diarizer.speaker_count == 4

        # Next embedding should NOT create a 5th speaker
        label = diarizer._assign_speaker(_make_embedding(400))
        assert diarizer.speaker_count == 4
        assert label in _SPEAKER_LABELS[:4]

    def test_centroid_updates_on_match(self, diarizer: SpeakerDiarizer) -> None:
        """Matching speaker should update its centroid (running average)."""
        emb = _make_embedding(1)
        diarizer._assign_speaker(emb)

        _, count_before = diarizer._centroids[0]
        assert count_before == 1

        # Same embedding → match → count increases
        diarizer._assign_speaker(emb)
        _, count_after = diarizer._centroids[0]
        assert count_after == 2

    def test_speaker_labels_sequential(self, diarizer: SpeakerDiarizer) -> None:
        """Speakers should be labeled A, B, C, D sequentially."""
        labels = []
        for seed in [10, 100, 200, 300]:
            labels.append(diarizer._assign_speaker(_make_embedding(seed)))
        assert labels == ["Speaker A", "Speaker B", "Speaker C", "Speaker D"]


class TestProcessTurn:
    """Tests for the full process_turn pipeline."""

    def test_unavailable_returns_none(self, config: DiarizationConfig) -> None:
        """When model is not loaded, process_turn returns None."""
        with patch.object(SpeakerDiarizer, "_load_model"):
            d = SpeakerDiarizer(config)
            d._classifier = None  # Model not available
        assert d.available is False
        result = d.process_turn(np.zeros(32000, dtype=np.float32))
        assert result is None

    def test_short_audio_returns_none(self, diarizer: SpeakerDiarizer) -> None:
        """Audio shorter than min_audio_duration should return None."""
        # 0.5s at 16kHz = 8000 samples, min_audio_duration = 1.0s
        short = np.zeros(8000, dtype=np.float32)
        result = diarizer.process_turn(short, sample_rate=16000)
        assert result is None

    def test_process_turn_calls_extract_and_assign(
        self,
        diarizer: SpeakerDiarizer,
    ) -> None:
        """process_turn should call _extract_embedding then _assign_speaker."""
        fake_emb = _make_embedding(42)
        with patch.object(diarizer, "_extract_embedding", return_value=fake_emb):
            label = diarizer.process_turn(np.zeros(32000, dtype=np.float32))
        assert label == "Speaker A"

    def test_process_turn_none_when_extraction_fails(
        self,
        diarizer: SpeakerDiarizer,
    ) -> None:
        """If _extract_embedding returns None, process_turn returns None."""
        with patch.object(diarizer, "_extract_embedding", return_value=None):
            result = diarizer.process_turn(np.zeros(32000, dtype=np.float32))
        assert result is None


class TestReset:
    """Tests for speaker state reset."""

    def test_reset_clears_centroids(self, diarizer: SpeakerDiarizer) -> None:
        """reset() should clear all speaker clusters."""
        diarizer._assign_speaker(_make_embedding(1))
        diarizer._assign_speaker(_make_embedding(99))
        assert diarizer.speaker_count == 2

        diarizer.reset()
        assert diarizer.speaker_count == 0

    def test_reset_allows_fresh_speakers(self, diarizer: SpeakerDiarizer) -> None:
        """After reset, new embeddings should create Speaker A again."""
        diarizer._assign_speaker(_make_embedding(1))
        diarizer.reset()

        label = diarizer._assign_speaker(_make_embedding(99))
        assert label == "Speaker A"


class TestSpeakerSummary:
    """Tests for get_speaker_summary."""

    def test_empty_summary(self, diarizer: SpeakerDiarizer) -> None:
        """Empty diarizer should return empty summary."""
        assert diarizer.get_speaker_summary() == {}

    def test_summary_tracks_counts(self, diarizer: SpeakerDiarizer) -> None:
        """Summary should reflect turn counts per speaker."""
        emb_a = _make_embedding(1)
        emb_b = _make_embedding(99)

        diarizer._assign_speaker(emb_a)
        diarizer._assign_speaker(emb_a)
        diarizer._assign_speaker(emb_b)

        summary = diarizer.get_speaker_summary()
        assert summary["Speaker A"] == 2
        assert summary["Speaker B"] == 1


class TestThreadSafety:
    """Tests for thread safety of process_turn."""

    def test_concurrent_process_turn(self, diarizer: SpeakerDiarizer) -> None:
        """Concurrent process_turn calls should not corrupt state."""
        results: list = []
        errors: list = []

        fake_emb = _make_embedding(42)

        def worker() -> None:
            try:
                label = diarizer.process_turn(np.zeros(32000, dtype=np.float32))
                results.append(label)
            except Exception as e:
                errors.append(e)

        # Patch once before spawning threads (patch.object is not thread-safe)
        with patch.object(diarizer, "_extract_embedding", return_value=fake_emb):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(results) == 10
        # All should be Speaker A (same embedding)
        assert all(r == "Speaker A" for r in results)


class TestGracefulFallback:
    """Tests for graceful behavior when speechbrain is not installed."""

    def test_import_error_sets_unavailable(self) -> None:
        """ImportError during model load should set available=False."""
        config = DiarizationConfig(enabled=True)
        with patch(
            "lib.diarization.SpeakerDiarizer._load_model",
            side_effect=ImportError("no speechbrain"),
        ):
            # _load_model is called in __init__ but we're patching it to raise
            # Need to bypass __init__'s call by using object.__new__
            d = object.__new__(SpeakerDiarizer)
            d._config = config
            d._lock = threading.Lock()
            d._centroids = []
            d._model = None
            d._classifier = None

        assert d.available is False
        assert d.process_turn(np.zeros(32000, dtype=np.float32)) is None


# ─── F-604: roster-bounded clustering ────────────────────────────────────


class TestRosterBoundedClustering:
    """set_roster_size caps clusters below/independent of max_speakers."""

    def test_roster_size_caps_clusters(self, diarizer: SpeakerDiarizer) -> None:
        # max_speakers is 4, but a roster of 2 caps clusters at 2.
        diarizer.set_roster_size(2)
        labels = {diarizer._assign_speaker(_make_embedding(s)) for s in range(10)}
        assert len(labels) <= 2
        assert diarizer.speaker_count == 2

    def test_roster_reassigns_to_nearest(self, diarizer: SpeakerDiarizer) -> None:
        diarizer.set_roster_size(2)
        diarizer._assign_speaker(_make_embedding(1))
        diarizer._assign_speaker(_make_embedding(500))
        # A third distinct speaker must re-assign to one of the two clusters.
        third = diarizer._assign_speaker(_make_embedding(999))
        assert third in {_SPEAKER_LABELS[0], _SPEAKER_LABELS[1]}
        assert diarizer.speaker_count == 2

    def test_none_roster_falls_back_to_max_speakers(self, diarizer: SpeakerDiarizer) -> None:
        diarizer.set_roster_size(None)
        assert diarizer._max_clusters() == 4  # max_speakers from config

    def test_zero_roster_ignored(self, diarizer: SpeakerDiarizer) -> None:
        diarizer.set_roster_size(0)
        assert diarizer._max_clusters() == 4


# ─── F-604: speaker-change segmentation ──────────────────────────────────


class TestChangePointDetection:
    def test_no_change_when_similar(self) -> None:
        base = _make_embedding(7)
        embs = [base, base.copy(), base.copy()]
        assert SpeakerDiarizer.detect_change_points(embs, threshold=0.55) == []

    def test_change_detected_between_speakers(self) -> None:
        a = _make_embedding(1)
        b = _make_embedding(500)
        # a, a, b, b → one boundary at index 2
        embs = [a, a.copy(), b, b.copy()]
        assert SpeakerDiarizer.detect_change_points(embs, threshold=0.55) == [2]

    def test_empty_and_single(self) -> None:
        assert SpeakerDiarizer.detect_change_points([], 0.55) == []
        assert SpeakerDiarizer.detect_change_points([_make_embedding(1)], 0.55) == []


class TestTurnSegmentation:
    def test_single_speaker_turn_one_segment(self, diarizer: SpeakerDiarizer) -> None:
        emb = _make_embedding(3)
        with patch.object(diarizer, "_extract_embedding", return_value=emb):
            segs = diarizer.process_turn_segments(np.zeros(48000, dtype=np.float32))
        labels = {s[0] for s in segs}
        assert labels == {"Speaker A"}

    def test_two_speaker_turn_segments_and_dominant(self, diarizer: SpeakerDiarizer) -> None:
        a = _make_embedding(1)
        b = _make_embedding(500)
        # 5 windows: A A A B B → dominant is A, two segments.
        five_windows = [np.zeros(24000, dtype=np.float32)] * 5
        seq = [a, a, a, b, b]
        with patch.object(diarizer, "_window_audio", return_value=five_windows), patch.object(
            diarizer, "_extract_embedding", side_effect=seq
        ):
            segs = diarizer.process_turn_segments(np.zeros(160000, dtype=np.float32))
        seg_labels = [s[0] for s in segs]
        assert len(segs) == 2
        assert seg_labels[0] == "Speaker A"
        assert seg_labels[1] == "Speaker B"

    def test_process_turn_returns_dominant_speaker(self, diarizer: SpeakerDiarizer) -> None:
        a = _make_embedding(1)
        b = _make_embedding(500)
        five_windows = [np.zeros(24000, dtype=np.float32)] * 5
        seq = [a, a, a, b, b]  # A dominates by window span
        with patch.object(diarizer, "_window_audio", return_value=five_windows), patch.object(
            diarizer, "_extract_embedding", side_effect=seq
        ):
            label = diarizer.process_turn(np.zeros(160000, dtype=np.float32))
        assert label == "Speaker A"

    def test_window_audio_short_turn_single_window(self, diarizer: SpeakerDiarizer) -> None:
        windows = diarizer._window_audio(np.zeros(16000, dtype=np.float32), 16000)
        assert len(windows) == 1
