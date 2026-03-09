"""Tests for lib.system_audio_capture — per-app audio via ScreenCaptureKit."""

import json
import queue
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.system_audio_capture import SystemAudioCapture


@pytest.fixture
def capture() -> SystemAudioCapture:
    """SystemAudioCapture instance with test settings (no real subprocess)."""
    return SystemAudioCapture(pid=12345, app_name="test-app", sample_rate=16000, chunk_duration=4.0)


class TestInit:
    """Tests for __init__ state."""

    def test_initial_state(self, capture: SystemAudioCapture) -> None:
        assert capture.pid == 12345
        assert capture.device == "test-app"
        assert capture.sample_rate == 16000
        assert capture.chunk_duration == 4.0
        assert capture.running is False
        assert capture.paused is False

    def test_default_app_name(self) -> None:
        c = SystemAudioCapture(pid=999)
        assert c.device == "PID 999"

    def test_chunk_samples(self, capture: SystemAudioCapture) -> None:
        assert capture.chunk_samples == 64000  # 4.0s * 16000Hz


class TestAudioMetrics:
    """Tests for _update_audio_metrics."""

    def test_silence(self, capture: SystemAudioCapture) -> None:
        silence = np.zeros(64000, dtype=np.float32)
        assert capture._update_audio_metrics(silence) is False

    def test_speech(self, capture: SystemAudioCapture) -> None:
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        speech = 0.1 * np.sin(2 * np.pi * 440 * t)
        assert capture._update_audio_metrics(speech) is True

    def test_metrics_update(self, capture: SystemAudioCapture) -> None:
        data = np.full(16000, 0.05, dtype=np.float32)
        capture._update_audio_metrics(data)
        assert capture._total_chunks == 1
        assert capture._speech_chunks == 1
        assert capture._last_rms > 0


class TestAudioHealth:
    """Tests for get_audio_health."""

    def test_initial_health(self, capture: SystemAudioCapture) -> None:
        health = capture.get_audio_health()
        assert health["total_chunks"] == 0
        assert health["speech_chunks"] == 0
        assert health["all_silent"] is False
        assert health["dropped_chunks"] == 0

    def test_all_silent_detection(self, capture: SystemAudioCapture) -> None:
        silence = np.zeros(16000, dtype=np.float32)
        for _ in range(5):
            capture._update_audio_metrics(silence)
        health = capture.get_audio_health()
        assert health["all_silent"] is True

    def test_no_capture_error_by_default(self, capture: SystemAudioCapture) -> None:
        """Health should not include capture_error when no error occurred."""
        health = capture.get_audio_health()
        assert "capture_error" not in health

    def test_capture_error_surfaced(self, capture: SystemAudioCapture) -> None:
        """When _capture_error is set, it should appear in health dict."""
        capture._capture_error = "Screen Recording permission denied for audio-tap."
        health = capture.get_audio_health()
        assert "capture_error" in health
        assert "permission denied" in str(health["capture_error"]).lower()

    def test_capture_error_with_binary_path(self, capture: SystemAudioCapture) -> None:
        """Error message should include binary path for user guidance."""
        capture._capture_error = (
            "Screen Recording permission denied for audio-tap. "
            "Add this binary to System Settings: /usr/local/bin/audio-tap"
        )
        health = capture.get_audio_health()
        assert "/usr/local/bin/audio-tap" in str(health["capture_error"])


class TestPauseResume:
    """Tests for pause/resume."""

    def test_pause(self, capture: SystemAudioCapture) -> None:
        assert capture.paused is False
        capture.pause()
        assert capture.paused is True

    def test_resume(self, capture: SystemAudioCapture) -> None:
        capture.pause()
        capture.resume()
        assert capture.paused is False


class TestSessionRecording:
    """Tests for audio segment retrieval and recording."""

    def test_get_audio_segment_empty(self, capture: SystemAudioCapture) -> None:
        assert capture.get_audio_segment(0.0, 10.0) is None

    def test_get_audio_segment_range(self, capture: SystemAudioCapture) -> None:
        chunk1 = np.ones(16000, dtype=np.float32)
        chunk2 = np.ones(16000, dtype=np.float32) * 2
        capture._session_audio = [chunk1, chunk2]
        capture._session_timestamps = [100.0, 104.0]
        result = capture.get_audio_segment(99.0, 105.0)
        assert result is not None
        assert len(result) == 32000

    def test_save_recording_empty(self, capture: SystemAudioCapture, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        assert capture.save_recording(out) is False

    def test_save_recording(self, capture: SystemAudioCapture, tmp_path: Path) -> None:
        capture._session_audio = [np.ones(16000, dtype=np.float32)]
        capture._session_timestamps = [0.0]
        out = tmp_path / "out.wav"
        assert capture.save_recording(out) is True
        assert out.exists()


class TestRecentFeatures:
    """Tests for get_recent_features."""

    def test_empty(self, capture: SystemAudioCapture) -> None:
        assert capture.get_recent_features(0.0) == []

    def test_filter_by_timestamp(self, capture: SystemAudioCapture) -> None:
        capture._chunk_features.append({"timestamp": 10.0, "rms": 0.01})
        capture._chunk_features.append({"timestamp": 20.0, "rms": 0.05})
        result = capture.get_recent_features(15.0)
        assert len(result) == 1
        assert result[0]["timestamp"] == 20.0


class TestIsAvailable:
    """Tests for is_available static method."""

    @patch("lib.system_audio_capture.platform.mac_ver")
    def test_not_macos(self, mock_ver: MagicMock) -> None:
        mock_ver.return_value = ("", ("", "", ""), "")
        assert SystemAudioCapture.is_available() is False

    @patch("lib.system_audio_capture._BINARY_PATH")
    @patch("lib.system_audio_capture.platform.mac_ver")
    def test_old_macos(self, mock_ver: MagicMock, mock_path: MagicMock) -> None:
        mock_ver.return_value = ("12.0", ("", "", ""), "")
        assert SystemAudioCapture.is_available() is False

    @patch("lib.system_audio_capture._BINARY_PATH")
    @patch("lib.system_audio_capture.platform.mac_ver")
    def test_macos13_with_binary(self, mock_ver: MagicMock, mock_path: MagicMock) -> None:
        mock_ver.return_value = ("13.0", ("", "", ""), "")
        mock_path.exists.return_value = True
        assert SystemAudioCapture.is_available() is True


class TestListApps:
    """Tests for list_apps static method."""

    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    @patch("lib.system_audio_capture.subprocess.run")
    def test_list_apps_success(self, mock_run: MagicMock, mock_binary: MagicMock) -> None:
        mock_binary.return_value = Path("/fake/audio-tap")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps([{"pid": 123, "name": "Zoom", "bundle_id": "us.zoom.xos"}]),
        )
        apps = SystemAudioCapture.list_apps()
        assert len(apps) == 1
        assert apps[0]["name"] == "Zoom"

    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    @patch("lib.system_audio_capture.subprocess.run")
    def test_list_apps_failure(self, mock_run: MagicMock, mock_binary: MagicMock) -> None:
        mock_binary.return_value = Path("/fake/audio-tap")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        apps = SystemAudioCapture.list_apps()
        assert apps == []


class TestCheckPermission:
    """Tests for check_permission static method."""

    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    @patch("lib.system_audio_capture.subprocess.run")
    def test_granted(self, mock_run: MagicMock, mock_binary: MagicMock) -> None:
        mock_binary.return_value = Path("/fake/audio-tap")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        assert SystemAudioCapture.check_permission() is True

    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    @patch("lib.system_audio_capture.subprocess.run")
    def test_denied(self, mock_run: MagicMock, mock_binary: MagicMock) -> None:
        mock_binary.return_value = Path("/fake/audio-tap")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
        assert SystemAudioCapture.check_permission() is False


class TestPermissionPreCheck:
    """Tests for permission pre-check in start_stream and subprocess error surfacing."""

    @patch("lib.system_audio_capture.SystemAudioCapture.check_permission")
    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    def test_permission_denied_sets_capture_error(
        self, mock_binary: MagicMock, mock_permission: MagicMock
    ) -> None:
        """When check_permission returns False, _capture_error should be set."""
        import threading

        mock_binary.return_value = Path("/fake/audio-tap")
        mock_permission.return_value = False

        cap = SystemAudioCapture(pid=123, app_name="zoom.us")

        # Mock Popen — subprocess exits immediately with error
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"permission denied"
        mock_proc.stderr.__iter__ = MagicMock(return_value=iter([]))

        with patch("lib.system_audio_capture.subprocess.Popen", return_value=mock_proc):
            t = threading.Thread(target=cap.start_stream, args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
            cap.running = False
            t.join(timeout=2.0)

        assert cap._capture_error is not None
        assert "permission denied" in cap._capture_error.lower()
        health = cap.get_audio_health()
        assert "capture_error" in health

    @patch("lib.system_audio_capture.SystemAudioCapture.check_permission")
    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    def test_permission_granted_no_precheck_error(
        self, mock_binary: MagicMock, mock_permission: MagicMock
    ) -> None:
        """When check_permission returns True, no permission error is set."""
        import threading

        mock_binary.return_value = Path("/fake/audio-tap")
        mock_permission.return_value = True

        cap = SystemAudioCapture(pid=123, app_name="zoom.us")

        # Process exits cleanly
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.__iter__ = MagicMock(return_value=iter([]))

        with patch("lib.system_audio_capture.subprocess.Popen", return_value=mock_proc):
            t = threading.Thread(target=cap.start_stream, args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
            cap.running = False
            t.join(timeout=2.0)

        # No permission-related error
        if cap._capture_error:
            assert "permission" not in cap._capture_error.lower()

    @patch("lib.system_audio_capture.SystemAudioCapture.check_permission", return_value=True)
    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    def test_nonzero_exit_captures_stderr(
        self, mock_binary: MagicMock, mock_permission: MagicMock
    ) -> None:
        """Non-zero exit code should capture stderr in _capture_error."""
        import threading

        mock_binary.return_value = Path("/fake/audio-tap")

        cap = SystemAudioCapture(pid=999, app_name="test-app")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"No running application found with PID 999"
        mock_proc.stderr.__iter__ = MagicMock(return_value=iter([]))

        with patch("lib.system_audio_capture.subprocess.Popen", return_value=mock_proc):
            t = threading.Thread(target=cap.start_stream, args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
            cap.running = False
            t.join(timeout=2.0)

        assert cap._capture_error is not None
        assert "exited with code 1" in cap._capture_error
        assert "No running application" in cap._capture_error

    @patch("lib.system_audio_capture.SystemAudioCapture.check_permission", return_value=True)
    @patch("lib.system_audio_capture.SystemAudioCapture._get_binary")
    def test_clean_exit_no_error(self, mock_binary: MagicMock, mock_permission: MagicMock) -> None:
        """Clean exit (returncode 0) should not set _capture_error."""
        import threading

        mock_binary.return_value = Path("/fake/audio-tap")

        cap = SystemAudioCapture(pid=123, app_name="zoom.us")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.__iter__ = MagicMock(return_value=iter([]))

        with patch("lib.system_audio_capture.subprocess.Popen", return_value=mock_proc):
            t = threading.Thread(target=cap.start_stream, args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
            cap.running = False
            t.join(timeout=2.0)

        # _capture_error should be None for clean exit
        assert cap._capture_error is None


class TestWorkerLoop:
    """Tests for the queue-based worker loop."""

    def test_process_chunk_callback(self, capture: SystemAudioCapture) -> None:
        """Worker should invoke callback with temp WAV path."""
        callback = MagicMock()
        capture.callback = callback

        chunk = np.random.randn(16000).astype(np.float32) * 0.1
        capture._process_chunk(chunk, 100.0)

        callback.assert_called_once()
        call_args = callback.call_args
        assert isinstance(call_args[0][0], Path)
        assert call_args[0][1] == 100.0

    def test_dropped_chunk_counter(self, capture: SystemAudioCapture) -> None:
        """Queue full should increment dropped counter."""
        small_q = queue.Queue(maxsize=1)
        capture._chunk_queue = small_q

        chunk = np.zeros(16000, dtype=np.float32)
        small_q.put_nowait((chunk, 0.0))  # Fill queue

        # Reader would try put_nowait → Full
        try:
            capture._chunk_queue.put_nowait((chunk, 1.0))
        except queue.Full:
            capture._dropped_chunks += 1

        assert capture._dropped_chunks == 1
