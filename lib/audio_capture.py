"""Audio Capture — queue-based streaming from mic/BlackHole with chunking.

Redesigned for zero data loss:
- Bounded chunk queue replaces fire-and-forget thread spawning
- Single worker thread processes chunks sequentially and in order
- No audio level gating — ASR model decides what is speech
- Optional session WAV recording for full meeting persistence
"""
import logging
import queue
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

# Sentinel value to signal worker thread to stop
_STOP_SENTINEL = None

# Queue size: 20 chunks × 4s = 80s buffer before backpressure
_DEFAULT_QUEUE_SIZE = 20


class AudioCapture:
    """Continuous audio capture with ordered, queue-based chunk processing.

    Audio flows through a bounded queue to a single worker thread,
    guaranteeing sequential processing and preventing data loss from
    thread contention. All audio reaches the callback — no amplitude
    gating. Audio levels are tracked for diagnostics only.
    """

    def __init__(
        self,
        device: str = "BlackHole 2ch",
        sample_rate: int = 16000,
        chunk_duration: float = 4.0,
        overlap: float = 0.5,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples

        self.buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.callback: Optional[Callable] = None

        # Chunk processing queue — replaces thread-per-chunk
        self._chunk_queue: queue.Queue[Optional[tuple]] = queue.Queue(
            maxsize=queue_size,
        )
        self._worker_thread: Optional[threading.Thread] = None

        # Audio health metrics (diagnostics only, no gating)
        self._total_chunks: int = 0
        self._speech_chunks: int = 0
        self._last_rms: float = 0.0
        self._last_peak: float = 0.0
        self._dropped_chunks: int = 0

        # Session recording — accumulates all raw audio
        self._session_audio: List[np.ndarray] = []
        self._recording_lock = threading.Lock()

        # Per-chunk audio features for speaker attribution
        self._chunk_features: Deque[Dict[str, float]] = deque(maxlen=50)
        self._features_lock = threading.Lock()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block (~100ms).

        Must not block. Extracts complete chunks from the rolling buffer
        and enqueues them for the worker thread.
        """
        if status:
            logger.warning("Audio status: %s", status)

        audio_data = indata.flatten().astype(np.float32)

        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, audio_data])

            while len(self.buffer) >= self.chunk_samples:
                chunk = self.buffer[: self.chunk_samples].copy()
                self.buffer = self.buffer[self.step_samples :]  # Keep overlap
                chunk_timestamp = time.time()

                # Enqueue — never block the audio thread
                try:
                    self._chunk_queue.put_nowait((chunk, chunk_timestamp))
                except queue.Full:
                    self._dropped_chunks += 1
                    logger.error(
                        "Chunk queue full — dropped chunk %d (queue=%d). "
                        "ASR may be falling behind.",
                        self._dropped_chunks,
                        self._chunk_queue.qsize(),
                    )

    def _worker_loop(self) -> None:
        """Single worker thread: drain queue, process chunks sequentially.

        Runs until it receives the stop sentinel or self.running is False.
        Drains any remaining chunks after stop to avoid data loss.
        """
        logger.info("Audio worker thread started")
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

        # Drain remaining chunks (don't lose data on shutdown)
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
            "Audio worker thread stopped (processed %d chunks, %d dropped)",
            self._total_chunks,
            self._dropped_chunks,
        )

    def _update_audio_metrics(self, audio_data: np.ndarray) -> bool:
        """Update audio level metrics for diagnostics. Returns True if speech detected.

        This does NOT gate processing — all audio reaches ASR regardless.
        """
        rms = float(np.sqrt(np.mean(audio_data**2)))
        peak = float(np.max(np.abs(audio_data)))
        has_audio = rms > 0.002 and peak > 0.01

        self._last_rms = rms
        self._last_peak = peak
        self._total_chunks += 1
        if has_audio:
            self._speech_chunks += 1

        # Log first 3 chunks + every 10th to monitor levels
        if self._total_chunks <= 3 or self._total_chunks % 10 == 0:
            logger.debug(
                "Audio level: rms=%.6f peak=%.4f %s (chunk %d)",
                rms,
                peak,
                "SPEECH" if has_audio else "silence",
                self._total_chunks,
            )
        return has_audio

    def get_audio_health(self) -> dict:
        """Get audio level health info for diagnostics."""
        return {
            "total_chunks": self._total_chunks,
            "speech_chunks": self._speech_chunks,
            "last_rms": self._last_rms,
            "last_peak": self._last_peak,
            "all_silent": self._total_chunks > 3 and self._speech_chunks == 0,
            "dropped_chunks": self._dropped_chunks,
            "queue_size": self._chunk_queue.qsize(),
        }

    @staticmethod
    def compute_chunk_features(audio_data: np.ndarray) -> Dict[str, float]:
        """Compute audio features for speaker attribution.

        Args:
            audio_data: 1D float32 audio array.

        Returns:
            Dict with 'rms' (root mean square energy) and 'zcr' (zero-crossing rate).
            Returns zeros for empty or invalid input.
        """
        if audio_data.size == 0:
            return {"rms": 0.0, "zcr": 0.0}

        flat = audio_data.flatten().astype(np.float32)
        rms = float(np.sqrt(np.mean(flat**2)))
        # Zero-crossing rate: fraction of adjacent samples with sign change
        if flat.size > 1:
            signs = np.sign(flat)
            sign_changes = np.abs(np.diff(signs))
            zcr = float(np.mean(sign_changes > 0))
        else:
            zcr = 0.0
        return {"rms": rms, "zcr": zcr}

    def get_recent_features(self, since_timestamp: float) -> List[Dict[str, float]]:
        """Get chunk features recorded since a given timestamp.

        Args:
            since_timestamp: Unix timestamp. Returns features after this time.

        Returns:
            List of feature dicts with 'rms', 'zcr', and 'timestamp'.
        """
        with self._features_lock:
            return [f for f in self._chunk_features if f.get("timestamp", 0) >= since_timestamp]

    def _process_chunk(self, chunk: np.ndarray, timestamp: float) -> None:
        """Process a single audio chunk: record, write temp WAV, call callback.

        No audio level gating — every chunk reaches the ASR model.
        The ASR output determines speech vs silence downstream.
        """
        try:
            # Track audio metrics (diagnostics only)
            self._update_audio_metrics(chunk)

            # Compute and store features for speaker attribution
            features = self.compute_chunk_features(chunk)
            features["timestamp"] = timestamp
            with self._features_lock:
                self._chunk_features.append(features)

            # Accumulate for session recording
            with self._recording_lock:
                self._session_audio.append(chunk)

            # Save to temporary WAV file for ASR
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)

            sf.write(temp_path, chunk, self.sample_rate)

            # Call the processing callback (ASR + pipeline)
            if self.callback:
                self.callback(temp_path, timestamp)

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error("Chunk processing error: %s", e)

    def start_stream(self, callback: Callable[[Path, float], None]) -> None:
        """Start continuous audio capture with queue-based processing."""
        self.callback = callback
        self.running = True
        self._session_audio = []
        self._dropped_chunks = 0

        # Start the single worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="audio-worker",
        )
        self._worker_thread.start()

        # Find the device index
        devices = sd.query_devices()
        device_idx = None
        for i, dev in enumerate(devices):
            if self.device.lower() in dev["name"].lower():
                device_idx = i
                break

        if device_idx is None:
            raise RuntimeError(
                f"Audio device '{self.device}' not found. "
                f"Available: {[d['name'] for d in devices]}"
            )

        logger.info("Starting audio capture from: %s", devices[device_idx]["name"])

        with sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        ):
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.running = False

    def stop(self) -> None:
        """Stop audio capture and drain remaining chunks."""
        self.running = False

        # Signal worker to stop and wait for it to drain
        try:
            self._chunk_queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            pass  # Worker will exit via running=False check

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)

    def save_recording(self, output_path: Path) -> bool:
        """Save the full session audio to a WAV file.

        Returns True if recording was saved, False if no audio was captured.
        """
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


def list_audio_devices() -> None:
    """List available audio devices."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        logger.info(
            "  [%d] %s (in: %d, out: %d)",
            i,
            dev["name"],
            dev["max_input_channels"],
            dev["max_output_channels"],
        )
