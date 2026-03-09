"""Tests for lib.audio_protocol — verify both capture classes satisfy the protocol."""

from lib.audio_capture import AudioCapture
from lib.audio_protocol import AudioCaptureProtocol
from lib.system_audio_capture import SystemAudioCapture


class TestProtocolConformance:
    """Both AudioCapture and SystemAudioCapture must satisfy AudioCaptureProtocol."""

    def test_audio_capture_satisfies_protocol(self) -> None:
        assert isinstance(AudioCapture(device="test"), AudioCaptureProtocol)

    def test_system_audio_capture_satisfies_protocol(self) -> None:
        assert isinstance(SystemAudioCapture(pid=1), AudioCaptureProtocol)

    def test_protocol_has_required_methods(self) -> None:
        """Protocol defines the expected interface."""
        required = [
            "start_stream",
            "stop",
            "pause",
            "resume",
            "get_audio_health",
            "get_audio_segment",
            "save_recording",
            "get_recent_features",
        ]
        for method in required:
            assert hasattr(AudioCaptureProtocol, method), f"Missing: {method}"

    def test_protocol_has_required_attributes(self) -> None:
        """Protocol defines expected attributes."""
        ac = AudioCapture(device="test")
        sc = SystemAudioCapture(pid=1)
        for attr in ["device", "sample_rate", "running"]:
            assert hasattr(ac, attr), f"AudioCapture missing: {attr}"
            assert hasattr(sc, attr), f"SystemAudioCapture missing: {attr}"
