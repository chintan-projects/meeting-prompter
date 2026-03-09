"""Tests for lib.audio_capture — queue-based processing and health tracking."""
import queue
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from lib.audio_capture import AudioCapture, _STOP_SENTINEL


@pytest.fixture
def capture() -> AudioCapture:
    """AudioCapture instance with default settings (no real device needed)."""
    return AudioCapture(device="test-device", sample_rate=16000, chunk_duration=4.0)


class TestUpdateAudioMetrics:
    """Tests for _update_audio_metrics threshold logic."""

    def test_silence_returns_false(self, capture: AudioCapture) -> None:
        """All-zero audio should be below threshold."""
        silence = np.zeros(64000, dtype=np.float32)
        assert capture._update_audio_metrics(silence) is False

    def test_speech_returns_true(self, capture: AudioCapture) -> None:
        """Loud audio should be above threshold."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        speech = 0.1 * np.sin(2 * np.pi * 440 * t)
        assert capture._update_audio_metrics(speech) is True

    def test_low_noise_below_threshold(self, capture: AudioCapture) -> None:
        """Very low amplitude noise should be filtered."""
        noise = np.random.randn(16000).astype(np.float32) * 0.0005
        assert capture._update_audio_metrics(noise) is False

    def test_rms_threshold_boundary(self, capture: AudioCapture) -> None:
        """Audio below RMS threshold should be classified as silence."""
        below = np.full(16000, 0.001, dtype=np.float32)
        assert capture._update_audio_metrics(below) is False

        above = np.full(16000, 0.02, dtype=np.float32)
        assert capture._update_audio_metrics(above) is True


class TestAudioHealth:
    """Tests for get_audio_health diagnostics."""

    def test_initial_health(self, capture: AudioCapture) -> None:
        """Fresh capture has zero state."""
        health = capture.get_audio_health()
        assert health["total_chunks"] == 0
        assert health["speech_chunks"] == 0
        assert health["all_silent"] is False
        assert health["dropped_chunks"] == 0
        assert health["queue_size"] == 0

    def test_all_silent_after_threshold(self, capture: AudioCapture) -> None:
        """all_silent should be True after >3 silent chunks."""
        silence = np.zeros(16000, dtype=np.float32)
        for _ in range(5):
            capture._update_audio_metrics(silence)

        health = capture.get_audio_health()
        assert health["total_chunks"] == 5
        assert health["speech_chunks"] == 0
        assert health["all_silent"] is True

    def test_not_silent_with_speech(self, capture: AudioCapture) -> None:
        """all_silent should be False if any speech detected."""
        silence = np.zeros(16000, dtype=np.float32)
        speech = np.full(16000, 0.1, dtype=np.float32)

        for _ in range(4):
            capture._update_audio_metrics(silence)
        capture._update_audio_metrics(speech)

        health = capture.get_audio_health()
        assert health["total_chunks"] == 5
        assert health["speech_chunks"] == 1
        assert health["all_silent"] is False

    def test_last_rms_peak_updated(self, capture: AudioCapture) -> None:
        """last_rms and last_peak should track most recent chunk."""
        loud = np.full(16000, 0.5, dtype=np.float32)
        capture._update_audio_metrics(loud)

        health = capture.get_audio_health()
        assert health["last_rms"] == pytest.approx(0.5, abs=0.01)
        assert health["last_peak"] == pytest.approx(0.5, abs=0.01)


class TestChunkQueue:
    """Tests for queue-based chunk processing."""

    def test_queue_created_with_correct_size(self) -> None:
        """Queue should be bounded to the specified size."""
        cap = AudioCapture(device="test", queue_size=10)
        assert cap._chunk_queue.maxsize == 10

    def test_worker_processes_chunks_sequentially(self) -> None:
        """Worker thread processes chunks in FIFO order."""
        cap = AudioCapture(device="test", sample_rate=16000)
        processed: list = []

        def fake_callback(path, ts):  # type: ignore[no-untyped-def]
            processed.append(ts)

        cap.callback = fake_callback
        cap.running = True

        # Start worker
        cap._worker_thread = threading.Thread(target=cap._worker_loop, daemon=True)
        cap._worker_thread.start()

        # Enqueue 5 chunks
        for i in range(5):
            chunk = np.zeros(64000, dtype=np.float32)
            cap._chunk_queue.put((chunk, float(i)))

        # Signal stop and wait
        cap._chunk_queue.put(_STOP_SENTINEL)
        cap._worker_thread.join(timeout=5.0)

        assert processed == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_worker_drains_on_stop(self) -> None:
        """Worker drains remaining chunks after receiving stop sentinel."""
        cap = AudioCapture(device="test", sample_rate=16000)
        processed: list = []

        def fake_callback(path, ts):  # type: ignore[no-untyped-def]
            processed.append(ts)

        cap.callback = fake_callback
        cap.running = False  # Already stopped

        # Pre-load queue with chunks then sentinel
        chunk = np.zeros(64000, dtype=np.float32)
        cap._chunk_queue.put((chunk, 10.0))
        cap._chunk_queue.put((chunk, 20.0))
        cap._chunk_queue.put(_STOP_SENTINEL)
        cap._chunk_queue.put((chunk, 30.0))  # After sentinel

        # Run worker (it will stop at sentinel, then drain remaining)
        cap._worker_loop()

        # Should process chunks before sentinel + drain after
        assert 10.0 in processed
        assert 20.0 in processed
        assert 30.0 in processed

    def test_queue_full_increments_dropped(self, capture: AudioCapture) -> None:
        """When queue is full, dropped_chunks counter should increment."""
        small_cap = AudioCapture(device="test", queue_size=2)

        chunk = np.zeros(64000, dtype=np.float32)
        # Fill the queue
        small_cap._chunk_queue.put((chunk, 1.0))
        small_cap._chunk_queue.put((chunk, 2.0))

        # This should fail and increment dropped
        try:
            small_cap._chunk_queue.put_nowait((chunk, 3.0))
        except queue.Full:
            small_cap._dropped_chunks += 1

        assert small_cap._dropped_chunks == 1


class TestSessionRecording:
    """Tests for session WAV recording."""

    def test_session_audio_accumulates(self, capture: AudioCapture) -> None:
        """Session audio list should grow with each processed chunk."""
        assert len(capture._session_audio) == 0

        chunk = np.ones(16000, dtype=np.float32) * 0.1
        with capture._recording_lock:
            capture._session_audio.append(chunk)

        assert len(capture._session_audio) == 1

    def test_save_recording_empty(self, capture: AudioCapture, tmp_path) -> None:
        """Saving with no audio should return False."""
        result = capture.save_recording(tmp_path / "test.wav")
        assert result is False

    def test_save_recording_with_audio(self, capture: AudioCapture, tmp_path) -> None:
        """Saving with audio should write WAV and return True."""
        chunk = np.ones(16000, dtype=np.float32) * 0.1
        capture._session_audio.append(chunk)

        out = tmp_path / "session.wav"
        result = capture.save_recording(out)
        assert result is True
        assert out.exists()
        assert out.stat().st_size > 0


class TestChunkFeatures:
    """Tests for per-chunk audio feature extraction (RMS + ZCR diagnostics)."""

    def test_compute_chunk_features_silence(self) -> None:
        """Silent audio should have near-zero features."""
        silence = np.zeros(16000, dtype=np.float32)
        features = AudioCapture.compute_chunk_features(silence)
        assert features["rms"] == pytest.approx(0.0, abs=1e-8)
        assert features["zcr"] == pytest.approx(0.0, abs=1e-8)

    def test_compute_chunk_features_sine(self) -> None:
        """Sine wave should have non-zero RMS and ZCR."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        sine = 0.1 * np.sin(2 * np.pi * 440 * t)
        features = AudioCapture.compute_chunk_features(sine)
        assert features["rms"] > 0.05
        assert features["zcr"] > 0.01

    def test_feature_vector_has_2_keys(self) -> None:
        """Feature dict should contain rms and zcr (spectral features removed)."""
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)
        features = AudioCapture.compute_chunk_features(audio)
        assert set(features.keys()) == {"rms", "zcr"}

    def test_feature_deque_stores_with_timestamp(self, capture: AudioCapture) -> None:
        """Features should be stored with timestamps in the deque."""
        chunk = np.full(64000, 0.05, dtype=np.float32)
        capture._update_audio_metrics(chunk)
        features = AudioCapture.compute_chunk_features(chunk)
        features["timestamp"] = 100.0
        with capture._features_lock:
            capture._chunk_features.append(features)

        result = capture.get_recent_features(99.0)
        assert len(result) == 1
        assert result[0]["timestamp"] == 100.0
        assert result[0]["rms"] > 0

    def test_get_recent_features_filters_by_timestamp(self, capture: AudioCapture) -> None:
        """get_recent_features should only return features after given timestamp."""
        for ts in [10.0, 20.0, 30.0, 40.0]:
            with capture._features_lock:
                capture._chunk_features.append({"rms": 0.05, "zcr": 0.1, "timestamp": ts})

        recent = capture.get_recent_features(25.0)
        assert len(recent) == 2
        assert recent[0]["timestamp"] == 30.0
        assert recent[1]["timestamp"] == 40.0

    def test_feature_deque_bounded(self) -> None:
        """Feature deque should not exceed maxlen."""
        cap = AudioCapture(device="test")
        for i in range(60):
            with cap._features_lock:
                cap._chunk_features.append({"rms": 0.01, "zcr": 0.01, "timestamp": float(i)})
        assert len(cap._chunk_features) == 50  # maxlen=50

    def test_compute_chunk_features_empty_array(self) -> None:
        """Empty array should return zeros."""
        empty = np.array([], dtype=np.float32)
        features = AudioCapture.compute_chunk_features(empty)
        assert features["rms"] == 0.0
        assert features["zcr"] == 0.0
        assert len(features) == 2

    def test_compute_chunk_features_single_sample(self) -> None:
        """Single sample array should return RMS but zero ZCR."""
        single = np.array([0.5], dtype=np.float32)
        features = AudioCapture.compute_chunk_features(single)
        assert features["rms"] == pytest.approx(0.5, abs=0.01)
        assert features["zcr"] == 0.0

    def test_compute_chunk_features_2d_array(self) -> None:
        """2D array should be flattened and handled correctly."""
        stereo = np.full((16000, 2), 0.05, dtype=np.float32)
        features = AudioCapture.compute_chunk_features(stereo)
        assert features["rms"] == pytest.approx(0.05, abs=0.001)

    def test_get_recent_features_empty_deque(self, capture: AudioCapture) -> None:
        """get_recent_features on empty deque should return empty list."""
        result = capture.get_recent_features(0.0)
        assert result == []


class TestGetAudioSegment:
    """Tests for get_audio_segment — time-range audio retrieval for diarization."""

    def test_empty_session_returns_none(self, capture: AudioCapture) -> None:
        """No recorded audio should return None."""
        result = capture.get_audio_segment(0.0, 10.0)
        assert result is None

    def test_retrieves_matching_chunks(self, capture: AudioCapture) -> None:
        """Should return audio chunks that overlap with the time range."""
        chunk = np.ones(64000, dtype=np.float32) * 0.1
        with capture._recording_lock:
            capture._session_audio.append(chunk)
            capture._session_timestamps.append(100.0)
            capture._session_audio.append(chunk * 2)
            capture._session_timestamps.append(104.0)

        result = capture.get_audio_segment(99.0, 105.0)
        assert result is not None
        assert len(result) == 128000  # Two 64000-sample chunks

    def test_partial_overlap(self, capture: AudioCapture) -> None:
        """Chunks partially overlapping the range should be included."""
        chunk = np.ones(64000, dtype=np.float32) * 0.1
        with capture._recording_lock:
            capture._session_audio.append(chunk)
            capture._session_timestamps.append(100.0)
            capture._session_audio.append(chunk * 2)
            capture._session_timestamps.append(104.0)
            capture._session_audio.append(chunk * 3)
            capture._session_timestamps.append(108.0)

        # Only first two chunks overlap with [101, 106]
        result = capture.get_audio_segment(101.0, 106.0)
        assert result is not None
        assert len(result) == 128000

    def test_no_overlap_returns_none(self, capture: AudioCapture) -> None:
        """Time range with no matching chunks should return None."""
        chunk = np.ones(64000, dtype=np.float32)
        with capture._recording_lock:
            capture._session_audio.append(chunk)
            capture._session_timestamps.append(100.0)

        result = capture.get_audio_segment(200.0, 210.0)
        assert result is None

    def test_single_chunk_exact_match(self, capture: AudioCapture) -> None:
        """Exact time range matching one chunk should return it."""
        chunk = np.full(64000, 0.5, dtype=np.float32)
        with capture._recording_lock:
            capture._session_audio.append(chunk)
            capture._session_timestamps.append(100.0)

        result = capture.get_audio_segment(100.0, 104.0)
        assert result is not None
        assert len(result) == 64000
        assert result[0] == pytest.approx(0.5)

    def test_timestamps_tracked_in_process_chunk(self) -> None:
        """_process_chunk should store timestamps alongside audio."""
        cap = AudioCapture(device="test", sample_rate=16000)
        cap.callback = MagicMock()  # Prevent callback issues

        chunk = np.zeros(64000, dtype=np.float32)
        cap._process_chunk(chunk, 123.456)

        assert len(cap._session_timestamps) == 1
        assert cap._session_timestamps[0] == 123.456
        assert len(cap._session_audio) == 1


class TestPauseResume:
    """Tests for pause/resume behavior."""

    def test_initial_not_paused(self, capture: AudioCapture) -> None:
        """Fresh capture should not be paused."""
        assert capture.paused is False

    def test_pause_sets_flag(self, capture: AudioCapture) -> None:
        """pause() should set the paused property to True."""
        capture.pause()
        assert capture.paused is True

    def test_resume_clears_flag(self, capture: AudioCapture) -> None:
        """resume() should clear the paused flag."""
        capture.pause()
        assert capture.paused is True
        capture.resume()
        assert capture.paused is False

    def test_resume_clears_buffer(self, capture: AudioCapture) -> None:
        """resume() should clear the audio buffer to avoid stale overlap."""
        with capture.buffer_lock:
            capture.buffer = np.ones(1000, dtype=np.float32)
        capture.pause()
        capture.resume()
        assert len(capture.buffer) == 0

    def test_audio_callback_discards_when_paused(self, capture: AudioCapture) -> None:
        """Audio callback should not enqueue chunks when paused."""
        capture.pause()
        audio = np.ones((1600, 1), dtype=np.float32) * 0.1  # 100ms block
        capture._audio_callback(audio, 1600, None, MagicMock(spec=False))
        # Buffer should stay empty since paused early-returns
        assert len(capture.buffer) == 0
        assert capture._chunk_queue.empty()

    def test_audio_callback_works_after_resume(self, capture: AudioCapture) -> None:
        """After resume, audio callback should enqueue chunks normally."""
        capture.pause()
        capture.resume()

        # Feed enough audio to create a full chunk (4s * 16000 = 64000 samples)
        # Feed in 100ms blocks
        for _ in range(41):  # 41 * 1600 = 65600 > 64000
            audio = np.ones((1600, 1), dtype=np.float32) * 0.05
            capture._audio_callback(audio, 1600, None, MagicMock(spec=False))

        assert not capture._chunk_queue.empty()


class TestStop:
    """Tests for stop() behavior."""

    def test_stop_sets_running_false(self, capture: AudioCapture) -> None:
        """stop() should set running to False."""
        capture.running = True
        capture.stop()
        assert capture.running is False

    def test_stop_sends_sentinel(self, capture: AudioCapture) -> None:
        """stop() should put sentinel on the queue."""
        capture.running = True
        capture.stop()
        # Check sentinel was enqueued
        item = capture._chunk_queue.get_nowait()
        assert item is _STOP_SENTINEL
