"""Spectral speaker attribution using cosine similarity on audio feature vectors.

Tracks speaker profiles via exponential moving average (EMA) of 11-dimensional
feature vectors (RMS, ZCR, spectral centroid/bandwidth/rolloff, 6 MFCCs).
When a new turn's features diverge from all known profiles, a new speaker
label is assigned.

This is a heuristic — not a neural diarization model. Works best when
speakers take turns (not overlapping). Spectral features capture vocal tract
shape, which is more speaker-specific than energy-only features.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature order — must match AudioCapture.compute_chunk_features() keys.
FEATURE_NAMES: List[str] = [
    "rms", "zcr",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
]

# Reference scales for normalizing features before similarity computation.
# Without normalization, spectral features (~1000s) completely dominate
# cosine similarity, making all speakers look identical. Each feature
# is divided by its scale so all dimensions contribute roughly equally.
_FEATURE_SCALES: np.ndarray = np.array([
    0.2,      # rms: typical speech 0.02–0.2
    0.3,      # zcr: typical 0.02–0.3
    2000.0,   # spectral_centroid: 100–4000 Hz
    1000.0,   # spectral_bandwidth: 50–2000 Hz
    4000.0,   # spectral_rolloff: 200–8000 Hz
    10.0,     # mfcc_0: range ~-20 to 0
    5.0,      # mfcc_1: range ~-5 to 5
    5.0,      # mfcc_2
    5.0,      # mfcc_3
    5.0,      # mfcc_4
    5.0,      # mfcc_5
], dtype=np.float64)


def _features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a numpy vector in canonical order.

    Missing features are filled with 0.0, so legacy dicts with only
    'rms' and 'zcr' still work (with reduced discrimination).
    """
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


@dataclass
class SpeakerProfile:
    """Running EMA of feature vectors for a speaker."""

    label: str
    feature_vector: np.ndarray
    turn_count: int = 0
    _alpha: float = field(default=0.3, repr=False)

    def update(self, features: np.ndarray) -> None:
        """Update profile with EMA of new feature vector."""
        self.turn_count += 1
        if self.turn_count == 1:
            self.feature_vector = features.copy()
        else:
            self.feature_vector = (
                self._alpha * features + (1 - self._alpha) * self.feature_vector
            )


class SpeakerTracker:
    """Assigns speaker labels based on spectral audio features.

    Uses cosine similarity on scale-normalized 11-dimensional feature
    vectors to compare turns against known speaker profiles. Features
    are normalized by reference scales before comparison so that each
    dimension (RMS ~0.05, centroid ~500 Hz, MFCC ~-4) contributes
    roughly equally to the similarity score.
    """

    DEFAULT_SILENCE_RMS = 0.002

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        ema_alpha: float = 0.3,
        silence_rms: float = DEFAULT_SILENCE_RMS,
    ) -> None:
        self._threshold = similarity_threshold
        self._alpha = ema_alpha
        self._silence_rms = silence_rms
        self._profiles: List[SpeakerProfile] = []
        self._next_speaker_num: int = 1

    @property
    def speaker_count(self) -> int:
        """Number of distinct speakers tracked."""
        return len(self._profiles)

    def on_turn_features(self, features: List[Dict[str, float]]) -> str:
        """Determine speaker label from per-chunk audio features.

        Args:
            features: List of feature dicts (one per chunk in the turn).

        Returns:
            Speaker label string, e.g. "Speaker 1".
        """
        if not features:
            return self._last_or_new_speaker()

        # Convert to vectors and average across chunks in this turn
        vectors = [_features_to_vector(f) for f in features]
        avg_vector = np.mean(vectors, axis=0)

        # Skip near-silent turns
        avg_rms = float(avg_vector[0])  # rms is index 0 in FEATURE_NAMES
        if avg_rms < self._silence_rms:
            return self._last_or_new_speaker()

        # Find best matching profile
        best_profile: Optional[SpeakerProfile] = None
        best_similarity = 0.0

        for profile in self._profiles:
            sim = self._compute_similarity(avg_vector, profile)
            if sim > best_similarity:
                best_similarity = sim
                best_profile = profile

        if best_profile and best_similarity >= self._threshold:
            best_profile.update(avg_vector)
            logger.debug(
                "Speaker match: %s (sim=%.3f, rms=%.4f)",
                best_profile.label,
                best_similarity,
                avg_rms,
            )
            return best_profile.label

        # New speaker
        label = f"Speaker {self._next_speaker_num}"
        self._next_speaker_num += 1
        profile = SpeakerProfile(
            label=label,
            feature_vector=avg_vector,
            _alpha=self._alpha,
        )
        profile.turn_count = 1
        self._profiles.append(profile)
        logger.info(
            "New speaker: %s (rms=%.4f, total=%d)",
            label,
            avg_rms,
            len(self._profiles),
        )
        return label

    def _compute_similarity(
        self, features: np.ndarray, profile: SpeakerProfile,
    ) -> float:
        """Compute cosine similarity on scale-normalized feature vectors.

        Features are divided by reference scales (_FEATURE_SCALES) before
        computing cosine similarity. This ensures each dimension contributes
        roughly equally — without normalization, spectral features (~1000s)
        dominate and all speakers appear identical.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        a = features / _FEATURE_SCALES
        b = profile.feature_vector / _FEATURE_SCALES

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        cosine_sim = float(np.dot(a, b)) / (norm_a * norm_b)
        return max(0.0, min(1.0, cosine_sim))

    def _last_or_new_speaker(self) -> str:
        """Return the most recent speaker label or create the first one."""
        if self._profiles:
            return self._profiles[-1].label
        return self.on_turn_features([{"rms": 0.01, "zcr": 0.05}])

    def reset(self) -> None:
        """Clear all speaker profiles."""
        self._profiles.clear()
        self._next_speaker_num = 1
