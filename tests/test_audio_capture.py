"""Tests for lib.audio_capture — audio level detection and health tracking."""
import numpy as np
import pytest

from lib.audio_capture import AudioCapture


@pytest.fixture
def capture() -> AudioCapture:
    """AudioCapture instance with default settings (no real device needed)."""
    return AudioCapture(device="test-device", sample_rate=16000, chunk_duration=4.0)


class TestCheckAudioLevel:
    """Tests for _check_audio_level threshold logic."""

    def test_silence_returns_false(self, capture: AudioCapture) -> None:
        """All-zero audio should be below threshold."""
        silence = np.zeros(64000, dtype=np.float32)
        assert capture._check_audio_level(silence) is False

    def test_speech_returns_true(self, capture: AudioCapture) -> None:
        """Loud audio should be above threshold."""
        # Sine wave at 0.1 amplitude — well above RMS=0.002 and peak=0.01
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        speech = 0.1 * np.sin(2 * np.pi * 440 * t)
        assert capture._check_audio_level(speech) is True

    def test_low_noise_below_threshold(self, capture: AudioCapture) -> None:
        """Very low amplitude noise should be filtered."""
        noise = np.random.randn(16000).astype(np.float32) * 0.0005
        assert capture._check_audio_level(noise) is False

    def test_rms_threshold_boundary(self, capture: AudioCapture) -> None:
        """Audio below RMS threshold should be filtered."""
        # RMS=0.001 is below the 0.002 threshold
        below = np.full(16000, 0.001, dtype=np.float32)
        assert capture._check_audio_level(below) is False

        # RMS=0.02 with peak=0.02 — above both thresholds
        above = np.full(16000, 0.02, dtype=np.float32)
        assert capture._check_audio_level(above) is True


class TestAudioHealth:
    """Tests for get_audio_health diagnostics."""

    def test_initial_health(self, capture: AudioCapture) -> None:
        """Fresh capture has zero state."""
        health = capture.get_audio_health()
        assert health["total_chunks"] == 0
        assert health["speech_chunks"] == 0
        assert health["all_silent"] is False  # Not enough chunks yet

    def test_all_silent_after_threshold(self, capture: AudioCapture) -> None:
        """all_silent should be True after >3 silent chunks."""
        silence = np.zeros(16000, dtype=np.float32)
        for _ in range(5):
            capture._check_audio_level(silence)

        health = capture.get_audio_health()
        assert health["total_chunks"] == 5
        assert health["speech_chunks"] == 0
        assert health["all_silent"] is True

    def test_not_silent_with_speech(self, capture: AudioCapture) -> None:
        """all_silent should be False if any speech detected."""
        silence = np.zeros(16000, dtype=np.float32)
        speech = np.full(16000, 0.1, dtype=np.float32)

        for _ in range(4):
            capture._check_audio_level(silence)
        capture._check_audio_level(speech)

        health = capture.get_audio_health()
        assert health["total_chunks"] == 5
        assert health["speech_chunks"] == 1
        assert health["all_silent"] is False

    def test_last_rms_peak_updated(self, capture: AudioCapture) -> None:
        """last_rms and last_peak should track most recent chunk."""
        loud = np.full(16000, 0.5, dtype=np.float32)
        capture._check_audio_level(loud)

        health = capture.get_audio_health()
        assert health["last_rms"] == pytest.approx(0.5, abs=0.01)
        assert health["last_peak"] == pytest.approx(0.5, abs=0.01)
