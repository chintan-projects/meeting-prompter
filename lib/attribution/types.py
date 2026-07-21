"""Attribution types — the speaker-attribution hierarchy (F-601).

Attribution quality is bounded by microphone topology and platform access, not by
the diarizer. The resolver composes layers by confidence and degrades honestly.
See docs/architecture/liquid-rearchitecture.md ("The attribution hierarchy").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttributionLayer(str, Enum):
    """Which signal produced a speaker label (strongest first)."""

    L1_CHANNEL = "L1_channel"  # mic vs system — ground truth me/others
    L2_PLATFORM = "L2_platform"  # Zoom SDK active speaker (future, F-608)
    L3_ACOUSTIC = "L3_acoustic"  # neural diarization cluster (estimate)
    L4_ROSTER = "L4_roster"  # roster/enrollment names a cluster
    NONE = "none"  # no attribution available


class Regime(str, Enum):
    """Meeting regime — sets honest expectations for attribution fidelity."""

    UNKNOWN = "unknown"
    SOLO_ENDPOINT = "solo_endpoint"  # one person per endpoint → names are trustworthy
    CONFERENCE_ROOM = "conference_room"  # many people, one far-field mic → degrade honestly


# Deterministic channel labels.
SELF_LABEL = "You"
OTHERS_LABEL = "Others"
# Conference-room degradation bucket (F-606).
ROOM_LABEL = "Others (room)"


@dataclass
class AttributionResult:
    """A resolved speaker label with its provenance and confidence.

    Attributes:
        speaker: The label to display (may be "" when unknown → caller keeps prior).
        layer: Which hierarchy layer produced it.
        confidence: 0.0–1.0 confidence in the *specific* attribution.
        low_confidence: True when the label is a best-effort/flagged guess that
            the UI should mark (e.g. conference-room "Others (room)" or Speaker N).
        note: Optional human-readable reason (e.g. "conference-room regime").
    """

    speaker: str
    layer: AttributionLayer = AttributionLayer.NONE
    confidence: float = 0.0
    low_confidence: bool = False
    note: str = ""
