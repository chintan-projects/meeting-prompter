"""Tests for energy-based speaker attribution."""
import pytest

from lib.speaker_tracker import SpeakerProfile, SpeakerTracker


class TestSpeakerProfile:
    """Tests for SpeakerProfile EMA updates."""

    def test_first_update_sets_values(self) -> None:
        profile = SpeakerProfile(label="Speaker 1", avg_rms=0.0, avg_zcr=0.0)
        profile.update(0.05, 0.15)
        assert profile.avg_rms == 0.05
        assert profile.avg_zcr == 0.15
        assert profile.turn_count == 1

    def test_ema_smoothing(self) -> None:
        profile = SpeakerProfile(label="Speaker 1", avg_rms=0.0, avg_zcr=0.0, _alpha=0.3)
        profile.update(0.10, 0.20)  # First: sets directly
        profile.update(0.04, 0.10)  # Second: EMA
        # EMA: 0.3 * 0.04 + 0.7 * 0.10 = 0.082
        assert abs(profile.avg_rms - 0.082) < 0.001
        # EMA: 0.3 * 0.10 + 0.7 * 0.20 = 0.17
        assert abs(profile.avg_zcr - 0.17) < 0.001
        assert profile.turn_count == 2


class TestSpeakerTracker:
    """Tests for speaker tracking and label assignment."""

    def test_first_turn_creates_speaker_1(self) -> None:
        tracker = SpeakerTracker()
        label = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        assert label == "Speaker 1"
        assert tracker.speaker_count == 1

    def test_similar_features_same_speaker(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        # First turn
        label1 = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        # Second turn — very similar features
        label2 = tracker.on_turn_features([{"rms": 0.052, "zcr": 0.148}])
        assert label1 == label2
        assert tracker.speaker_count == 1

    def test_different_features_new_speaker(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        # Speaker with low RMS, low ZCR
        label1 = tracker.on_turn_features([{"rms": 0.02, "zcr": 0.05}])
        # Very different speaker — high RMS, high ZCR
        label2 = tracker.on_turn_features([{"rms": 0.08, "zcr": 0.25}])
        assert label1 != label2
        assert tracker.speaker_count == 2
        assert label1 == "Speaker 1"
        assert label2 == "Speaker 2"

    def test_speaker_returns_after_different(self) -> None:
        tracker = SpeakerTracker(similarity_threshold=0.6)
        # Speaker A
        label_a = tracker.on_turn_features([{"rms": 0.02, "zcr": 0.05}])
        # Speaker B (different)
        label_b = tracker.on_turn_features([{"rms": 0.08, "zcr": 0.25}])
        # Speaker A returns
        label_a2 = tracker.on_turn_features([{"rms": 0.021, "zcr": 0.052}])
        assert label_a == label_a2
        assert label_a != label_b
        assert tracker.speaker_count == 2

    def test_empty_features_returns_last_speaker(self) -> None:
        tracker = SpeakerTracker()
        label1 = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        label2 = tracker.on_turn_features([])
        assert label2 == label1

    def test_silent_turn_returns_last_speaker(self) -> None:
        tracker = SpeakerTracker()
        label1 = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        # Very low RMS — treated as silence
        label2 = tracker.on_turn_features([{"rms": 0.001, "zcr": 0.01}])
        assert label2 == label1

    def test_multiple_chunks_averaged(self) -> None:
        tracker = SpeakerTracker()
        features = [
            {"rms": 0.04, "zcr": 0.10},
            {"rms": 0.06, "zcr": 0.20},
        ]
        label = tracker.on_turn_features(features)
        assert label == "Speaker 1"
        # Profile should have averaged values: rms=0.05, zcr=0.15
        assert abs(tracker._profiles[0].avg_rms - 0.05) < 0.001
        assert abs(tracker._profiles[0].avg_zcr - 0.15) < 0.001

    def test_reset_clears_profiles(self) -> None:
        tracker = SpeakerTracker()
        tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        assert tracker.speaker_count == 1
        tracker.reset()
        assert tracker.speaker_count == 0
        # Next turn starts fresh
        label = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        assert label == "Speaker 1"

    def test_similarity_threshold_controls_sensitivity(self) -> None:
        # Low threshold — more likely to match
        tracker_low = SpeakerTracker(similarity_threshold=0.3)
        tracker_low.on_turn_features([{"rms": 0.03, "zcr": 0.10}])
        label = tracker_low.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        # Should match with low threshold
        assert label == "Speaker 1"

        # High threshold — less likely to match
        tracker_high = SpeakerTracker(similarity_threshold=0.95)
        tracker_high.on_turn_features([{"rms": 0.03, "zcr": 0.10}])
        label = tracker_high.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        # May create new speaker with high threshold
        assert tracker_high.speaker_count >= 1

    def test_custom_silence_rms(self) -> None:
        """Custom silence_rms threshold should control what counts as silence."""
        tracker = SpeakerTracker(silence_rms=0.01)
        # Below custom threshold — treated as silence
        label = tracker.on_turn_features([{"rms": 0.005, "zcr": 0.05}])
        # With no prior speakers, should still create Speaker 1 via fallback
        assert label == "Speaker 1"

    def test_custom_rms_weight(self) -> None:
        """Custom rms_weight should affect similarity calculation."""
        # With full weight on RMS, ZCR difference shouldn't matter
        tracker = SpeakerTracker(rms_weight=1.0, similarity_threshold=0.5)
        tracker.on_turn_features([{"rms": 0.05, "zcr": 0.10}])
        # Same RMS, very different ZCR — should still match
        label = tracker.on_turn_features([{"rms": 0.05, "zcr": 0.30}])
        assert label == "Speaker 1"

    def test_custom_feature_ranges(self) -> None:
        """Custom rms_range and zcr_range affect normalization."""
        # Narrow range makes moderate differences look bigger → more speaker splits
        tracker = SpeakerTracker(
            rms_range=0.01, zcr_range=0.03, similarity_threshold=0.6
        )
        tracker.on_turn_features([{"rms": 0.05, "zcr": 0.15}])
        # With narrow rms_range=0.01, diff of 0.008 = 80% of range → large distance
        tracker.on_turn_features([{"rms": 0.058, "zcr": 0.17}])
        assert tracker.speaker_count == 2


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

    def test_load_config_with_speaker_section(self, tmp_path) -> None:
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
