"""Audio Capture - Real-time streaming from BlackHole with chunking."""
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioCapture:
    """Continuous audio capture with chunking for real-time processing."""

    def __init__(
        self,
        device: str = "BlackHole 2ch",
        sample_rate: int = 16000,
        chunk_duration: float = 4.0,
        overlap: float = 0.5,
    ):
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
        self.on_silence: Optional[Callable] = None  # Called when silence detected
        self._silence_count = 0  # Track consecutive silence chunks
        self._total_chunks = 0  # Total chunks processed
        self._speech_chunks = 0  # Chunks with audio above threshold
        self._last_rms: float = 0.0
        self._last_peak: float = 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            logger.warning("Audio status: %s", status)

        audio_data = indata.flatten().astype(np.float32)

        with self.buffer_lock:
            self.buffer = np.concatenate([self.buffer, audio_data])

            # Check if we have enough for a chunk
            while len(self.buffer) >= self.chunk_samples:
                chunk = self.buffer[:self.chunk_samples].copy()
                self.buffer = self.buffer[self.step_samples:]  # Keep overlap
                chunk_timestamp = time.time()

                # Process chunk in separate thread to avoid blocking audio
                if self.callback:
                    threading.Thread(
                        target=self._process_chunk,
                        args=(chunk, chunk_timestamp),
                        daemon=True
                    ).start()

    def _check_audio_level(self, audio_data: np.ndarray) -> bool:
        """Check if audio level is sufficient for transcription."""
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        peak = float(np.max(np.abs(audio_data)))
        has_audio = rms > 0.002 and peak > 0.01

        self._last_rms = rms
        self._last_peak = peak
        self._total_chunks += 1
        if has_audio:
            self._speech_chunks += 1

        # Log first 3 chunks + every 10th to monitor levels
        if self._total_chunks <= 3 or self._total_chunks % 10 == 0:
            logger.info(
                "Audio level: rms=%.6f peak=%.4f %s (chunk %d)",
                rms, peak, "SPEECH" if has_audio else "silence", self._total_chunks,
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
        }

    def _process_chunk(self, chunk: np.ndarray, timestamp: float):
        """Save chunk to temp file and call callback with timestamp."""
        try:
            # Check audio level - notify on silence instead of silent discard
            if not self._check_audio_level(chunk):
                self._silence_count += 1
                # Notify silence callback so buffer can handle pause detection
                if self.on_silence:
                    self.on_silence(timestamp)
                return

            # Reset silence counter when we get speech
            self._silence_count = 0

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = Path(f.name)

            sf.write(temp_path, chunk, self.sample_rate)

            # Call the processing callback with timestamp
            if self.callback:
                self.callback(temp_path, timestamp)

            # Clean up temp file
            temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error("Chunk processing error: %s", e)

    def start_stream(self, callback: Callable[[Path], None]):
        """Start continuous audio capture."""
        self.callback = callback
        self.running = True

        # Find the device index
        devices = sd.query_devices()
        device_idx = None
        for i, dev in enumerate(devices):
            if self.device.lower() in dev['name'].lower():
                device_idx = i
                break

        if device_idx is None:
            raise RuntimeError(
                f"Audio device '{self.device}' not found. "
                f"Available: {[d['name'] for d in devices]}"
            )

        logger.info("Starting audio capture from: %s", devices[device_idx]['name'])

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

    def stop(self):
        """Stop audio capture."""
        self.running = False


def list_audio_devices():
    """List available audio devices."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        logger.info(
            "  [%d] %s (in: %d, out: %d)",
            i, dev['name'], dev['max_input_channels'], dev['max_output_channels'],
        )
