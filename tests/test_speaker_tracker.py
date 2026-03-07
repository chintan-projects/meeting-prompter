"""Tests for spectral speaker attribution with cosine similarity."""
import numpy as np
import pytest

from lib.speaker_tracker import FEATURE_NAMES, SpeakerProfile, SpeakerTracker, _features_to_vector


# --- Helpers: realistic feature dicts for testing ---

def _low_pitch_speaker(**overrides: float) -> dict:
    """Features resembling a low-pitched male voice."""
    base = {
        "rms": 0.04, "zcr": 0.08,
        "spectral_centroid": 250.0, "spectral_bandwidth": 120.0,
        "spectral_rolloff": 500.0,
        "mfcc_0": -5.0, "mfcc_1": 1.2, "mfcc_2": 0.4,
        "mfcc_3": -0.3, "mfcc_4": 0.2, "mfcc_5": 0.1,
    }
    base.update(overrides)
    return base


def _high_pitch_speaker(**overrides: float) -> dict:
    """Features resembling a higher-pitched voice."""
    base = {
        "rms": 0.06, "zcr": 0.22,
        "spectral_centroid": 1800.0, "spectral_bandwidth": 700.0,
        "spectral_rolloff": 3500.0,
        "mfcc_0": -2.0, "mfcc_1": -1.5, "mfcc_2": -0.8,
        "mfcc_3": 0.5, "mfcc_4": -0.3, "mfcc_5": -0.4,
    }
    base.update(overrides)
    return base


class TestFeaturesToVector:
    """Tests for dict → numpy vector conversion."""

    def test_full_features_to_vector(self) -> None:
        features = _low_pitch_speaker()
        vec = _features_to_vector(features)
        assert vec.shape == (11,)
        assert vec[0] == pytest.approx(0.04)  # rms
        assert vec[2] == pytest.approx(250.0)  # spectral_centroid

    def test_partial_features_fill_zeros(self) -> None:
        """Legacy dicts with only rms/zcr should fill zeros for missing keys."""
        vec = _features_to_vector({"rms": 0.05, "zcr": 0.15})
        assert vec.shape == (11,)
        assert vec[0] == pytest.approx(0.05)
        assert vec[1] == pytest.approx(0.15)
        assert vec[2] == 0.0  # spectral_centroid
        assert vec[5] == 0.0  # mfcc_0

    def test_feature_names_order(self) -> None:
        assert len(FEATURE_NAMES) == 11
        assert FEATURE_NAMES[0] == "rms"
        assert FEATURE_NAMES[-1] == "mfcc_5"


class TestSpeakerProfile:
    """Tests for SpeakerProfile EMA updates."""

    def test_first_update_sets_values(self) -> None:
        profile = SpeakerProfile(label="Speaker 1", feature_vector=np.zeros(11))
        vec = np.array([0.05, 0.15, 500.0, 200.0, 1000.0, -4.0, 1.0, 0.5, -0.3, 0.2, 0.1])
        profile.update(vec)
        np.testing.assert_array_almost_equal(profile.feature_vector, vec)
        assert profile.turn_count == 1

    def test_ema_smoothing(self) -> None:
        profile = SpeakerProfile(label="Speaker 1", feature_vector=np.zeros(11), _alpha=0.3)
        first = np.array([0.10, 0.20] + [0.0] * 9)
        second = np.array([0.04, 0.10] + [0.0] * 9)
        profile.update(first)
        profile.update(second)
        # EMA rms: 0.3 * 0.04 + 0.7 * 0.10 = 0.082
        assert profile.feature_vector[0] == pytest.approx(0.082, abs=0.001)
        # EMA zcr: 0.3 * 0.10 + 0.7 * 0.20 = 0.17
        assert profile.feature_vector[1] == pytest.approx(0.17, abs=0.001)
        assert profile.turn_count == 2


class TestSpeakerTracker:
    """Tests for speaker tracking with spectral features."""

    def test_first_turn_creates_speaker_1(self) -> None:
        tracker = SpeakerTracker()
        label = tracker.on_turn_features([_low_pitch_speaker()])
        assert label == "Speaker 1"
        assert tracker.speaker_count == 1

    def test_similar_features_same_speaker(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        label1 = tracker.on_turn_features([_low_pitch_speaker()])
        # Slightly different — same speaker
        label2 = tracker.on_turn_features([_low_pitch_speaker(rms=0.042, zcr=0.082)])
        assert label1 == label2
        assert tracker.speaker_count == 1

    def test_different_features_new_speaker(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        label1 = tracker.on_turn_features([_low_pitch_speaker()])
        label2 = tracker.on_turn_features([_high_pitch_speaker()])
        assert label1 != label2
        assert tracker.speaker_count == 2
        assert label1 == "Speaker 1"
        assert label2 == "Speaker 2"

    def test_speaker_returns_after_different(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        label_a = tracker.on_turn_features([_low_pitch_speaker()])
        label_b = tracker.on_turn_features([_high_pitch_speaker()])
        # Speaker A returns with slightly different features
        label_a2 = tracker.on_turn_features([_low_pitch_speaker(rms=0.041, zcr=0.079)])
        assert label_a == label_a2
        assert label_a != label_b
        assert tracker.speaker_count == 2

    def test_three_distinct_speakers(self) -> None:
        """Three speakers with distinct spectral profiles get unique labels."""
        tracker = SpeakerTracker(similarity_threshold=0.6)
        low = tracker.on_turn_features([_low_pitch_speaker()])
        high = tracker.on_turn_features([_high_pitch_speaker()])
        # Third speaker: mid-range with very different MFCC pattern
        mid = tracker.on_turn_features([{
            "rms": 0.05, "zcr": 0.14,
            "spectral_centroid": 600.0, "spectral_bandwidth": 300.0,
            "spectral_rolloff": 1200.0,
            "mfcc_0": 2.0, "mfcc_1": 3.0, "mfcc_2": -2.5,
            "mfcc_3": -1.5, "mfcc_4": 1.5, "mfcc_5": 1.0,
        }])
        assert len({low, high, mid}) == 3
        assert tracker.speaker_count == 3

    def test_empty_features_returns_last_speaker(self) -> None:
        tracker = SpeakerTracker()
        label1 = tracker.on_turn_features([_low_pitch_speaker()])
        label2 = tracker.on_turn_features([])
        assert label2 == label1

    def test_silent_turn_returns_last_speaker(self) -> None:
        tracker = SpeakerTracker()
        label1 = tracker.on_turn_features([_low_pitch_speaker()])
        label2 = tracker.on_turn_features([_low_pitch_speaker(rms=0.001)])
        assert label2 == label1

    def test_multiple_chunks_averaged(self) -> None:
        tracker = SpeakerTracker()
        features = [
            _low_pitch_speaker(rms=0.03),
            _low_pitch_speaker(rms=0.05),
        ]
        label = tracker.on_turn_features(features)
        assert label == "Speaker 1"
        # Profile vector should reflect averaged rms
        assert tracker._profiles[0].feature_vector[0] == pytest.approx(0.04, abs=0.001)

    def test_reset_clears_profiles(self) -> None:
        tracker = SpeakerTracker()
        tracker.on_turn_features([_low_pitch_speaker()])
        assert tracker.speaker_count == 1
        tracker.reset()
        assert tracker.speaker_count == 0
        label = tracker.on_turn_features([_low_pitch_speaker()])
        assert label == "Speaker 1"

    def test_cosine_similarity_scale_invariance(self) -> None:
        """Vectors pointing same direction but different magnitude should match."""
        tracker = SpeakerTracker(similarity_threshold=0.6)
        base = _low_pitch_speaker()
        scaled = {k: v * 2 for k, v in base.items()}
        label1 = tracker.on_turn_features([base])
        label2 = tracker.on_turn_features([scaled])
        assert label1 == label2
        assert tracker.speaker_count == 1

    def test_custom_silence_rms(self) -> None:
        """Custom silence_rms threshold controls what counts as silence."""
        tracker = SpeakerTracker(silence_rms=0.01)
        label = tracker.on_turn_features([_low_pitch_speaker(rms=0.005)])
        assert label == "Speaker 1"  # Fallback creates first speaker

    def test_similarity_threshold_controls_sensitivity(self) -> None:
        low_tracker = SpeakerTracker(similarity_threshold=0.3)
        low_tracker.on_turn_features([_low_pitch_speaker()])
        label = low_tracker.on_turn_features([_low_pitch_speaker(rms=0.06, zcr=0.12)])
        assert label == "Speaker 1"

        high_tracker = SpeakerTracker(similarity_threshold=0.999)
        high_tracker.on_turn_features([_low_pitch_speaker()])
        high_tracker.on_turn_features([_low_pitch_speaker(rms=0.06, zcr=0.12)])
        assert high_tracker.speaker_count >= 1


class TestSpeakerConfig:
    """Tests for SpeakerConfig loading."""

    def test_default_speaker_config(self) -> None:
        from lib.config import SpeakerConfig

        config = SpeakerConfig()
        assert config.enabled is True
        assert config.similarity_threshold == 0.6
        assert config.ema_alpha == 0.3

    def test_speaker_config_custom_values(self) -> None:
        from lib.config import SpeakerConfig

        config = SpeakerConfig(enabled=False, similarity_threshold=0.8, ema_alpha=0.5)
        assert config.enabled is False
        assert config.similarity_threshold == 0.8
        assert config.ema_alpha == 0.5

    def test_app_config_includes_speaker(self) -> None:
        from lib.config import AppConfig

        config = AppConfig()
        assert hasattr(config, "speaker")
        assert config.speaker.enabled is True

    def test_load_config_with_speaker_section(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Config loader should parse speaker section from YAML."""
        from lib.config import load_config

        yaml_content = """
speaker:
  enabled: false
  similarity_threshold: 0.75
  ema_alpha: 0.4
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)
        config = load_config(config_file)
        assert config.speaker.enabled is False
        assert config.speaker.similarity_threshold == 0.75
        assert config.speaker.ema_alpha == 0.4
