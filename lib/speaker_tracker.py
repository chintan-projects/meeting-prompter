"""Energy-based speaker attribution using RMS energy and zero-crossing rate.

Tracks speaker profiles via exponential moving average (EMA) of audio
features. When a new turn's features diverge from all known profiles,
a new speaker label is assigned.

This is a heuristic — not a neural diarization model. It works well for
2-3 speakers with distinct vocal characteristics and degrades gracefully
with more speakers (labels may merge or split occasionally).
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Running average of audio features for a speaker."""

    label: str
    avg_rms: float
    avg_zcr: float
    turn_count: int = 0
    _alpha: float = field(default=0.3, repr=False)

    def update(self, rms: float, zcr: float) -> None:
        """Update profile with EMA of new features."""
        self.turn_count += 1
        if self.turn_count == 1:
            self.avg_rms = rms
            self.avg_zcr = zcr
        else:
            self.avg_rms = self._alpha * rms + (1 - self._alpha) * self.avg_rms
            self.avg_zcr = self._alpha * zcr + (1 - self._alpha) * self.avg_zcr


class SpeakerTracker:
    """Assigns speaker labels based on audio energy features.

    Uses RMS energy and zero-crossing rate (ZCR) to build per-speaker
    profiles. New turns are compared against profiles — if similar enough,
    the same label is reused; otherwise a new speaker is created.
    """

    # Feature normalization ranges (typical speech values)
    DEFAULT_RMS_RANGE = 0.1  # RMS spans ~0 to 0.1
    DEFAULT_ZCR_RANGE = 0.3  # ZCR spans ~0 to 0.3
    DEFAULT_RMS_WEIGHT = 0.6  # RMS slightly more discriminative than ZCR
    DEFAULT_SILENCE_RMS = 0.002  # Below this, treat as silence

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        ema_alpha: float = 0.3,
        silence_rms: float = DEFAULT_SILENCE_RMS,
        rms_range: float = DEFAULT_RMS_RANGE,
        zcr_range: float = DEFAULT_ZCR_RANGE,
        rms_weight: float = DEFAULT_RMS_WEIGHT,
    ) -> None:
        self._threshold = similarity_threshold
        self._alpha = ema_alpha
        self._silence_rms = silence_rms
        self._rms_range = rms_range
        self._zcr_range = zcr_range
        self._rms_weight = rms_weight
        self._profiles: List[SpeakerProfile] = []
        self._next_speaker_num: int = 1

    @property
    def speaker_count(self) -> int:
        """Number of distinct speakers tracked."""
        return len(self._profiles)

    def on_turn_features(self, features: List[Dict[str, float]]) -> str:
        """Determine speaker label from a list of per-chunk audio features.

        Args:
            features: List of dicts with 'rms' and 'zcr' keys, one per chunk
                      in the turn.

        Returns:
            Speaker label string, e.g. "Speaker 1".
        """
        if not features:
            return self._last_or_new_speaker()

        # Average features across chunks in this turn
        avg_rms = sum(f.get("rms", 0.0) for f in features) / len(features)
        avg_zcr = sum(f.get("zcr", 0.0) for f in features) / len(features)

        # Skip near-silent turns — don't create new speakers for silence
        if avg_rms < self._silence_rms:
            return self._last_or_new_speaker()

        # Find best matching profile
        best_profile: Optional[SpeakerProfile] = None
        best_similarity = 0.0

        for profile in self._profiles:
            sim = self._compute_similarity(avg_rms, avg_zcr, profile)
            if sim > best_similarity:
                best_similarity = sim
                best_profile = profile

        if best_profile and best_similarity >= self._threshold:
            best_profile.update(avg_rms, avg_zcr)
            logger.debug(
                "Speaker match: %s (sim=%.3f, rms=%.4f, zcr=%.4f)",
                best_profile.label,
                best_similarity,
                avg_rms,
                avg_zcr,
            )
            return best_profile.label

        # New speaker
        label = f"Speaker {self._next_speaker_num}"
        self._next_speaker_num += 1
        profile = SpeakerProfile(
            label=label,
            avg_rms=avg_rms,
            avg_zcr=avg_zcr,
            _alpha=self._alpha,
        )
        profile.turn_count = 1
        self._profiles.append(profile)
        logger.info(
            "New speaker: %s (rms=%.4f, zcr=%.4f, total=%d)",
            label,
            avg_rms,
            avg_zcr,
            len(self._profiles),
        )
        return label

    def _compute_similarity(
        self, rms: float, zcr: float, profile: SpeakerProfile
    ) -> float:
        """Compute similarity between features and a speaker profile.

        Uses normalized distance in RMS/ZCR space, converted to a 0-1
        similarity score. Features are normalized by reference ranges
        to give equal weight.
        """
        rms_diff = abs(rms - profile.avg_rms) / self._rms_range
        zcr_diff = abs(zcr - profile.avg_zcr) / self._zcr_range

        # Weighted distance → similarity
        zcr_weight = 1.0 - self._rms_weight
        distance = self._rms_weight * rms_diff + zcr_weight * zcr_diff
        similarity = max(0.0, 1.0 - distance)
        return similarity

    def _last_or_new_speaker(self) -> str:
        """Return the most recent speaker label or create the first one."""
        if self._profiles:
            return self._profiles[-1].label
        return self.on_turn_features([{"rms": 0.01, "zcr": 0.05}])

    def reset(self) -> None:
        """Clear all speaker profiles."""
        self._profiles.clear()
        self._next_speaker_num = 1
