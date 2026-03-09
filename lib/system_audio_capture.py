"""Per-app audio capture via ScreenCaptureKit (macOS 13+).

Wraps the `audio-tap` Swift CLI tool which captures audio from a specific
application using ScreenCaptureKit. Implements the same interface as
AudioCapture so Session and Orchestrator can use either interchangeably.

The Swift tool outputs raw float32 PCM to stdout. This class reads the
pipe, chunks it, and feeds it through the same queue-based processing
pipeline as AudioCapture (temp WAV → callback → ASR).
"""

import json
import logging
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import soundfile as sf

from lib.paths import get_project_root, get_runner_dir

logger = logging.getLogger(__name__)

# Sentinel value to signal worker thread to stop
_STOP_SENTINEL = None

# Default queue size: 20 chunks × 4s = 80s buffer
_DEFAULT_QUEUE_SIZE = 20

# Paths (resolved via lib.paths for dev + packaged mode)
_BINARY_PATH = get_runner_dir() / "audio-tap"
_SOURCE_DIR = get_project_root() / "tools" / "audio-tap"


class SystemAudioCapture:
    """Per-app audio capture via ScreenCaptureKit.

    Launches the `audio-tap` Swift CLI as a subprocess, reads raw float32
    PCM from its stdout, and processes chunks through the same pipeline
    as AudioCapture (queue → worker → temp WAV → callback).

    Usage:
        capture = SystemAudioCapture(pid=12345, app_name="zoom.us")
        capture.start_stream(my_callback)  # blocks until stop()
    """

    def __init__(
        self,
        pid: int,
        app_name: str = "",
        sample_rate: int = 16000,
        chunk_duration: float = 4.0,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.pid = pid
        self.device = app_name or f"PID {pid}"  # display name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration

        self.chunk_samples = int(chunk_duration * sample_rate)
        self.running = False
        self._paused = False
        self.callback: Optional[Callable[[Path, float], None]] = None

        # Chunk queue — same pattern as AudioCapture
        self._chunk_queue: queue.Queue[Optional[tuple]] = queue.Queue(
            maxsize=queue_size,
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[bytes]] = None

        # Audio health metrics
        self._total_chunks: int = 0
        self._speech_chunks: int = 0
        self._last_rms: float = 0.0
        self._last_peak: float = 0.0
        self._dropped_chunks: int = 0

        # Session recording
        self._session_audio: List[np.ndarray] = []
        self._session_timestamps: List[float] = []
        self._recording_lock = threading.Lock()

        # Per-chunk features
        self._chunk_features: Deque[Dict[str, float]] = deque(maxlen=50)
        self._features_lock = threading.Lock()

    @property
    def paused(self) -> bool:
        """Whether audio capture is paused."""
        return self._paused

    # --- Static utilities ---

    @staticmethod
    def is_available() -> bool:
        """Check if per-app audio capture is available on this system.

        Requires macOS 13.0+ and either a pre-built binary or swiftc.
        """
        # Check macOS version
        ver = platform.mac_ver()[0]
        if not ver:
            return False
        parts = ver.split(".")
        major = int(parts[0]) if parts else 0
        if major < 13:
            return False

        # Check binary or build tools
        if _BINARY_PATH.exists():
            return True
        return shutil.which("swiftc") is not None and _SOURCE_DIR.exists()

    @staticmethod
    def list_apps() -> List[Dict[str, object]]:
        """List running apps available for audio capture.

        Returns list of dicts with pid, name, bundle_id keys.
        """
        binary = SystemAudioCapture._get_binary()
        result = subprocess.run(
            [str(binary), "--list-apps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("list-apps failed: %s", result.stderr.strip())
            return []
        apps: List[Dict[str, object]] = json.loads(result.stdout)
        return apps

    @staticmethod
    def check_permission() -> bool:
        """Check if Screen Recording permission is granted."""
        binary = SystemAudioCapture._get_binary()
        result = subprocess.run(
            [str(binary), "--check-permission"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0

    @staticmethod
    def _get_binary() -> Path:
        """Get path to audio-tap binary, building from source if needed."""
        if _BINARY_PATH.exists():
            return _BINARY_PATH

        build_script = _SOURCE_DIR / "build.sh"
        if not build_script.exists():
            raise RuntimeError(
                f"audio-tap binary not found at {_BINARY_PATH} "
                f"and source not available at {_SOURCE_DIR}"
            )

        logger.info("Building audio-tap from source...")
        result = subprocess.run(
            ["bash", str(build_script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"audio-tap build failed: {result.stderr}")

        if not _BINARY_PATH.exists():
            raise RuntimeError("audio-tap build succeeded but binary not found")

        logger.info("audio-tap built successfully")
        return _BINARY_PATH

    # --- Stream lifecycle ---

    def start_stream(self, callback: Callable[[Path, float], None]) -> None:
        """Start per-app audio capture with queue-based processing.

        Launches the Swift subprocess, starts reader + worker threads,
        and blocks until stop() is called.
        """
        self.callback = callback
        self.running = True
        self._session_audio = []
        self._session_timestamps = []
        self._dropped_chunks = 0
        self._capture_error: Optional[str] = None

        binary = self._get_binary()

        # Pre-flight: check Screen Recording permission
        if not self.check_permission():
            self._capture_error = (
                "Screen Recording permission denied for audio-tap. "
                f"Add this binary to System Settings → Privacy & Security → "
                f"Screen & System Audio Recording: {binary}"
            )
            logger.error(self._capture_error)

        # Launch Swift subprocess
        cmd = [
            str(binary),
            "--pid",
            str(self.pid),
            "--sample-rate",
            str(self.sample_rate),
            "--chunk-duration",
            str(self.chunk_duration),
        ]
        logger.info("Starting audio-tap: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Start worker thread (processes chunks → temp WAV → callback)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="system-audio-worker",
        )
        self._worker_thread.start()

        # Start reader thread (reads PCM from subprocess stdout → queue)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="system-audio-reader",
        )
        self._reader_thread.start()

        # Start stderr logger thread
        stderr_thread = threading.Thread(
            target=self._stderr_loop,
            daemon=True,
            name="system-audio-stderr",
        )
        stderr_thread.start()

        logger.info("SystemAudioCapture started: %s (PID %d)", self.device, self.pid)

        # Block until stopped (same pattern as AudioCapture.start_stream)
        try:
            while self.running:
                # Check if subprocess died unexpectedly
                if self._process.poll() is not None:
                    rc = self._process.returncode
                    if rc != 0 and self.running:
                        # Collect stderr for error context
                        stderr_tail = ""
                        if self._process.stderr:
                            try:
                                stderr_tail = self._process.stderr.read().decode(
                                    "utf-8", errors="replace"
                                )
                            except Exception:
                                pass
                        self._capture_error = (
                            f"audio-tap exited with code {rc}. " f"{stderr_tail.strip()}"
                        )
                        logger.error(
                            "audio-tap exited with code %d: %s",
                            rc,
                            stderr_tail.strip() or "(no stderr)",
                        )
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False

    def stop(self) -> None:
        """Stop capture and clean up."""
        self.running = False

        # Signal worker to stop
        try:
            self._chunk_queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            pass

        # Terminate subprocess
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=5.0)

    def pause(self) -> None:
        """Pause capture (subprocess keeps running, data discarded)."""
        self._paused = True
        logger.info("SystemAudioCapture paused: %s", self.device)

    def resume(self) -> None:
        """Resume capture after pause."""
        self._paused = False
        logger.info("SystemAudioCapture resumed: %s", self.device)

    # --- Reader thread ---

    def _reader_loop(self) -> None:
        """Read raw float32 PCM from subprocess stdout and enqueue chunks."""
        if not self._process or not self._process.stdout:
            return

        # Each chunk is chunk_samples * 4 bytes (float32)
        chunk_bytes = self.chunk_samples * 4
        stdout = self._process.stdout

        logger.info("Reader thread started (chunk size: %d bytes)", chunk_bytes)

        while self.running:
            try:
                data = stdout.read(chunk_bytes)
                if not data:
                    break  # EOF — subprocess exited

                if self._paused:
                    continue  # Discard data while paused

                # Convert raw bytes to float32 numpy array
                n_samples = len(data) // 4
                if n_samples == 0:
                    continue

                chunk = np.frombuffer(data[: n_samples * 4], dtype=np.float32).copy()
                timestamp = time.time()

                try:
                    self._chunk_queue.put_nowait((chunk, timestamp))
                except queue.Full:
                    self._dropped_chunks += 1
                    logger.error(
                        "Chunk queue full — dropped chunk %d",
                        self._dropped_chunks,
                    )
            except Exception as e:
                if self.running:
                    logger.error("Reader error: %s", e)
                break

        logger.info("Reader thread stopped")

    def _stderr_loop(self) -> None:
        """Log subprocess stderr output at WARNING level so errors are visible."""
        if not self._process or not self._process.stderr:
            return
        for line in self._process.stderr:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                # Surface audio-tap messages at WARNING so they're visible in logs
                if "error" in text.lower() or "fatal" in text.lower() or "denied" in text.lower():
                    logger.error("[audio-tap] %s", text)
                else:
                    logger.warning("[audio-tap] %s", text)

    # --- Worker thread (identical pattern to AudioCapture) ---

    def _worker_loop(self) -> None:
        """Process chunks sequentially from the queue."""
        logger.info("Worker thread started")
        while True:
            try:
                item = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running:
                    break
                continue

            if item is _STOP_SENTINEL:
                self._chunk_queue.task_done()
                break

            chunk, timestamp = item
            self._process_chunk(chunk, timestamp)
            self._chunk_queue.task_done()

        # Drain remaining chunks
        while not self._chunk_queue.empty():
            try:
                item = self._chunk_queue.get_nowait()
                if item is not _STOP_SENTINEL and item is not None:
                    chunk, timestamp = item
                    self._process_chunk(chunk, timestamp)
                self._chunk_queue.task_done()
            except queue.Empty:
                break

        logger.info(
            "Worker thread stopped (processed %d chunks, %d dropped)",
            self._total_chunks,
            self._dropped_chunks,
        )

    def _process_chunk(self, chunk: np.ndarray, timestamp: float) -> None:
        """Process a chunk: update metrics, record, write WAV, call callback."""
        try:
            self._update_audio_metrics(chunk)

            # Compute features
            from lib.audio_capture import AudioCapture

            features = AudioCapture.compute_chunk_features(chunk, self.sample_rate)
            features["timestamp"] = timestamp
            with self._features_lock:
                self._chunk_features.append(features)

            # Accumulate for recording
            with self._recording_lock:
                self._session_audio.append(chunk)
                self._session_timestamps.append(timestamp)

            # Write temp WAV for ASR
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)

            sf.write(temp_path, chunk, self.sample_rate)

            if self.callback:
                self.callback(temp_path, timestamp)

            temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error("Chunk processing error: %s", e)

    def _update_audio_metrics(self, audio_data: np.ndarray) -> bool:
        """Update audio level metrics. Returns True if speech detected."""
        rms = float(np.sqrt(np.mean(audio_data**2)))
        peak = float(np.max(np.abs(audio_data)))
        has_audio = rms > 0.002 and peak > 0.01

        self._last_rms = rms
        self._last_peak = peak
        self._total_chunks += 1
        if has_audio:
            self._speech_chunks += 1

        if self._total_chunks <= 3 or self._total_chunks % 10 == 0:
            logger.debug(
                "Audio level: rms=%.6f peak=%.4f %s (chunk %d)",
                rms,
                peak,
                "SPEECH" if has_audio else "silence",
                self._total_chunks,
            )
        return has_audio

    # --- Health and recording ---

    def get_audio_health(self) -> Dict[str, object]:
        """Get audio level health info for diagnostics."""
        health: Dict[str, object] = {
            "total_chunks": self._total_chunks,
            "speech_chunks": self._speech_chunks,
            "last_rms": self._last_rms,
            "last_peak": self._last_peak,
            "all_silent": self._total_chunks > 3 and self._speech_chunks == 0,
            "dropped_chunks": self._dropped_chunks,
            "queue_size": self._chunk_queue.qsize(),
        }
        if hasattr(self, "_capture_error") and self._capture_error:
            health["capture_error"] = self._capture_error
        return health

    def get_audio_segment(
        self,
        start_time: float,
        end_time: float,
    ) -> Optional[np.ndarray]:
        """Retrieve raw audio for a time range (for diarization)."""
        with self._recording_lock:
            if not self._session_audio:
                return None

            chunks: List[np.ndarray] = []
            chunk_duration = self.chunk_duration

            for i, ts in enumerate(self._session_timestamps):
                chunk_end = ts + chunk_duration
                if chunk_end >= start_time and ts <= end_time:
                    chunks.append(self._session_audio[i])

            if not chunks:
                return None

            return np.concatenate(chunks)

    def save_recording(self, output_path: Path) -> bool:
        """Save full session audio to WAV."""
        with self._recording_lock:
            if not self._session_audio:
                logger.info("No audio captured — nothing to save")
                return False
            full_audio = np.concatenate(self._session_audio)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), full_audio, self.sample_rate)
        duration_s = len(full_audio) / self.sample_rate
        logger.info(
            "Session recording saved: %s (%.1fs, %.1f MB)",
            output_path,
            duration_s,
            output_path.stat().st_size / (1024 * 1024),
        )
        return True

    def get_recent_features(
        self,
        since_timestamp: float,
    ) -> List[Dict[str, float]]:
        """Get chunk features recorded since a given timestamp."""
        with self._features_lock:
            return [f for f in self._chunk_features if f.get("timestamp", 0) >= since_timestamp]
