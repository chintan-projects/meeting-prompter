"""Audio capture protocol — interface shared by AudioCapture and SystemAudioCapture.

Both sounddevice-based capture (AudioCapture) and ScreenCaptureKit-based
per-app capture (SystemAudioCapture) implement this protocol, allowing
Session and MeetingOrchestrator to use either interchangeably.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioCaptureProtocol(Protocol):
    """Protocol for audio capture implementations."""

    device: str
    sample_rate: int
    running: bool

    @property
    def paused(self) -> bool: ...

    def start_stream(self, callback: Callable[[Path, float], None]) -> None:
        """Start audio capture. Calls callback(wav_path, timestamp) for each chunk."""
        ...

    def stop(self) -> None:
        """Stop capture and clean up resources."""
        ...

    def pause(self) -> None:
        """Pause capture (stream stays alive, data discarded)."""
        ...

    def resume(self) -> None:
        """Resume capture after pause."""
        ...

    def get_audio_health(self) -> Dict[str, object]:
        """Return audio health diagnostics dict."""
        ...

    def get_audio_segment(
        self,
        start_time: float,
        end_time: float,
    ) -> Optional[np.ndarray]:
        """Retrieve raw audio for a time range (for diarization)."""
        ...

    def save_recording(self, output_path: Path) -> bool:
        """Save full session audio to WAV. Returns True if saved."""
        ...

    def get_recent_features(
        self,
        since_timestamp: float,
    ) -> List[Dict[str, float]]:
        """Get chunk features recorded since a given timestamp."""
        ...
