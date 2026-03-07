"""Audio feature extraction for speaker attribution.

Computes spectral features (centroid, bandwidth, rolloff) and
MFCCs from raw audio using numpy/scipy only — no librosa dependency.
These features capture vocal tract characteristics that persist even
in mono mixed audio, enabling better speaker discrimination than
energy-only features (RMS/ZCR).
"""
from typing import Dict, List

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_spectral_features(
    audio: np.ndarray, sample_rate: int = 16000,
) -> Dict[str, float]:
    """Compute spectral features from audio via FFT.

    Args:
        audio: 1D float32 audio array.
        sample_rate: Sample rate in Hz.

    Returns:
        Dict with spectral_centroid, spectral_bandwidth, spectral_rolloff.
        Returns zeros for empty or silent input.
    """
    zeros = {"spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "spectral_rolloff": 0.0}
    if audio.size == 0:
        return zeros

    flat = audio.flatten().astype(np.float32)
    spectrum = np.abs(rfft(flat))
    freqs = rfftfreq(len(flat), d=1.0 / sample_rate)

    spectrum_sum = float(np.sum(spectrum))
    if spectrum_sum < 1e-10:
        return zeros

    # Spectral centroid: weighted mean of frequencies
    centroid = float(np.sum(freqs * spectrum) / spectrum_sum)

    # Spectral bandwidth: weighted std of frequencies around centroid
    bandwidth = float(np.sqrt(np.sum(spectrum * (freqs - centroid) ** 2) / spectrum_sum))

    # Spectral rolloff: frequency below which 85% of energy lies
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * spectrum_sum))
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff = float(freqs[rolloff_idx])

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }


def compute_mfccs(
    audio: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 6, n_mels: int = 26,
) -> List[float]:
    """Compute MFCCs from audio using numpy/scipy only.

    Pipeline: power spectrum → mel filterbank → log → DCT.

    Args:
        audio: 1D float32 audio array.
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients to return.
        n_mels: Number of mel filter bands.

    Returns:
        List of n_mfcc float coefficients. Returns zeros for empty input.
    """
    if audio.size == 0:
        return [0.0] * n_mfcc

    flat = audio.flatten().astype(np.float32)

    # Power spectrum
    spectrum = np.abs(rfft(flat)) ** 2
    fft_freqs = rfftfreq(len(flat), d=1.0 / sample_rate)

    # Mel filterbank
    mel_filters = _mel_filterbank(n_mels, fft_freqs, sample_rate)

    # Apply filterbank → log compress
    mel_energies = mel_filters @ spectrum
    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel = np.log(mel_energies)

    # DCT Type II — first n_mfcc coefficients
    n = len(log_mel)
    indices = np.arange(n)
    dct_matrix = np.zeros((n_mfcc, n))
    for k in range(n_mfcc):
        dct_matrix[k] = np.cos(np.pi * k * (2 * indices + 1) / (2 * n))

    mfccs = dct_matrix @ log_mel
    return [float(c) for c in mfccs]


def _mel_filterbank(
    n_mels: int, fft_freqs: np.ndarray, sample_rate: int,
) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix.

    Args:
        n_mels: Number of mel filter bands.
        fft_freqs: Array of FFT bin center frequencies.
        sample_rate: Audio sample rate in Hz.

    Returns:
        (n_mels, n_fft_bins) filterbank matrix.
    """
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(float(m)) for m in mel_points])

    n_bins = len(fft_freqs)
    filters = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # Rising slope
        if center > low:
            mask_rise = (fft_freqs >= low) & (fft_freqs <= center)
            filters[i, mask_rise] = (fft_freqs[mask_rise] - low) / (center - low)
        # Falling slope
        if high > center:
            mask_fall = (fft_freqs > center) & (fft_freqs <= high)
            filters[i, mask_fall] = (high - fft_freqs[mask_fall]) / (high - center)

    return filters


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
