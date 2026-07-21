"""AttributionResolver — composes the speaker-attribution hierarchy (F-601).

Single place that decides a turn's speaker label. It composes:

  * L1 channel source (deterministic): mic → "You", system → "Others".
  * L3 acoustic: a neural-diarization cluster label for a system turn (estimate).
  * L4 roster: map a cluster label to a known name (interactive rename now;
    voice enrollment in F-605).

Regime detection sets honest expectations. In the CONFERENCE_ROOM regime (many
people, one far-field mic) acoustic clusters are unreliable, so the resolver
degrades to a single flagged ``Others (room)`` bucket rather than emitting
confidently-wrong names (F-606).

Default regime is UNKNOWN and the roster is empty, so with no configuration the
resolver reproduces the prior Tier-1/Tier-2 behavior exactly.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from lib.attribution.types import (
    OTHERS_LABEL,
    ROOM_LABEL,
    SELF_LABEL,
    AttributionLayer,
    AttributionResult,
    Regime,
)

logger = logging.getLogger(__name__)

# Roster size at/above which a single shared mic is treated as a room regime.
_DEFAULT_ROOM_ROSTER_THRESHOLD = 4


class AttributionResolver:
    """Resolves speaker labels by composing the attribution hierarchy."""

    def __init__(
        self,
        roster: Optional[List[str]] = None,
        regime: Regime = Regime.UNKNOWN,
        room_roster_threshold: int = _DEFAULT_ROOM_ROSTER_THRESHOLD,
    ) -> None:
        self._roster: List[str] = list(roster or [])
        self._regime = regime
        self._room_roster_threshold = room_roster_threshold

    # ─── Configuration ───────────────────────────────────────────────────

    @property
    def regime(self) -> Regime:
        return self._regime

    @property
    def roster(self) -> List[str]:
        return list(self._roster)

    def set_roster(self, names: List[str]) -> None:
        self._roster = list(names or [])

    def set_regime(self, regime: Regime) -> None:
        self._regime = regime

    def detect_regime(self, single_shared_mic: bool = False) -> Regime:
        """Infer the regime from roster size + mic topology, and store it.

        A single far-field mic with several expected participants is the
        conference-room regime; otherwise solo-endpoint. Callers that know the
        regime directly should use ``set_regime`` instead.
        """
        if single_shared_mic and len(self._roster) >= self._room_roster_threshold:
            self._regime = Regime.CONFERENCE_ROOM
        else:
            self._regime = Regime.SOLO_ENDPOINT
        logger.info(
            "Attribution regime: %s (roster=%d, shared_mic=%s)",
            self._regime.value,
            len(self._roster),
            single_shared_mic,
        )
        return self._regime

    # ─── L1: channel source (deterministic) ──────────────────────────────

    def resolve_channel(self, source: str) -> AttributionResult:
        """Ground-truth me/others from the audio channel.

        Unknown sources yield an empty label (confidence 0) so the caller keeps
        whatever label the turn already had (legacy compatibility).
        """
        if source == "mic":
            return AttributionResult(SELF_LABEL, AttributionLayer.L1_CHANNEL, 1.0)
        if source == "system":
            # We know it is *not* you; the specific remote speaker is unknown,
            # so "Others" is a high-confidence bucket, not a per-speaker label.
            return AttributionResult(OTHERS_LABEL, AttributionLayer.L1_CHANNEL, 0.9)
        return AttributionResult("", AttributionLayer.NONE, 0.0)

    # ─── L3/L4: acoustic cluster + roster naming ─────────────────────────

    def resolve_acoustic(
        self,
        diar_label: Optional[str],
        names: Optional[Dict[str, str]] = None,
    ) -> AttributionResult:
        """Resolve a system turn's speaker from a diarization cluster label.

        In the conference-room regime, acoustic clusters over one far-field mic
        are not trustworthy → degrade to a single flagged ``Others (room)``
        bucket. Otherwise map the cluster label through the name overrides (L4)
        and return it as an acoustic estimate.
        """
        if self._regime is Regime.CONFERENCE_ROOM:
            return AttributionResult(
                ROOM_LABEL,
                AttributionLayer.L1_CHANNEL,
                confidence=0.4,
                low_confidence=True,
                note="conference-room regime: acoustic clusters unreliable",
            )

        if not diar_label:
            return AttributionResult("", AttributionLayer.NONE, 0.0)

        names = names or {}
        if diar_label in names:
            return AttributionResult(names[diar_label], AttributionLayer.L4_ROSTER, confidence=0.7)
        return AttributionResult(diar_label, AttributionLayer.L3_ACOUSTIC, confidence=0.5)
