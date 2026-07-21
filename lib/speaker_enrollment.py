"""Voice enrollment — name acoustic clusters from pre-registered profiles (F-605).

Highest-leverage add for the conference-room / recurring-colleague regime: a
frequent participant records a short sample once, we store their speaker
embedding under a real name, and later acoustic clusters that match are labeled
with that name (L4 in the attribution hierarchy) instead of "Speaker A".

This module is the pure store + matching + persistence. It operates on
embeddings (192-dim ECAPA vectors), so it is unit-testable without audio or a
model. The diarizer consumes it via ``SpeakerDiarizer.set_enrollment``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MATCH_THRESHOLD = 0.70


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return the L2-normalized vector (zero vector unchanged)."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VoiceEnrollment:
    """A store of named voice profiles with nearest-profile identification."""

    def __init__(self, match_threshold: float = _DEFAULT_MATCH_THRESHOLD) -> None:
        self._profiles: Dict[str, np.ndarray] = {}
        self._match_threshold = match_threshold

    # ─── Enrollment ──────────────────────────────────────────────────────

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Register (or refine) a named voice profile.

        Re-enrolling an existing name averages the new sample into the profile
        (a running mean of unit vectors), so multiple samples tighten it.
        """
        name = name.strip()
        if not name:
            raise ValueError("enrollment name must be non-empty")
        new = _unit(embedding)
        if name in self._profiles:
            merged = _unit(self._profiles[name] + new)
            self._profiles[name] = merged
        else:
            self._profiles[name] = new

    def remove(self, name: str) -> bool:
        """Remove a profile. Returns True if it existed."""
        return self._profiles.pop(name, None) is not None

    # ─── Identification ──────────────────────────────────────────────────

    def identify(self, embedding: np.ndarray) -> Optional[str]:
        """Return the enrolled name whose profile best matches, or None.

        A match requires cosine similarity ≥ the match threshold; otherwise the
        voice is unknown (the diarizer falls back to an anonymous cluster).
        """
        if not self._profiles:
            return None
        emb = _unit(embedding)
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, profile in self._profiles.items():
            sim = _cosine(emb, profile)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is not None and best_sim >= self._match_threshold:
            return best_name
        return None

    # ─── Introspection ───────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        return list(self._profiles.keys())

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, name: object) -> bool:
        return name in self._profiles

    # ─── Persistence (local JSON — no egress) ────────────────────────────

    def save(self, path: Path) -> None:
        """Persist profiles to a local JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: vec.tolist() for name, vec in self._profiles.items()}
        path.write_text(json.dumps(payload))

    @classmethod
    def load(
        cls, path: Path, match_threshold: float = _DEFAULT_MATCH_THRESHOLD
    ) -> "VoiceEnrollment":
        """Load profiles from a local JSON file (missing file → empty store)."""
        enrollment = cls(match_threshold=match_threshold)
        if not path.exists():
            return enrollment
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read enrollment file %s: %s", path, exc)
            return enrollment
        for name, vec in raw.items():
            try:
                enrollment._profiles[name] = _unit(np.array(vec, dtype=np.float32))
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping bad enrollment profile %r: %s", name, exc)
        return enrollment
