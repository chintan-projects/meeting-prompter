"""Tests for the AttributionResolver hierarchy (F-601)."""

from __future__ import annotations

from lib.attribution import (
    AttributionLayer,
    AttributionResolver,
    Regime,
)
from lib.attribution.types import OTHERS_LABEL, ROOM_LABEL, SELF_LABEL


class TestChannelLayer:
    def test_mic_is_you_ground_truth(self) -> None:
        r = AttributionResolver().resolve_channel("mic")
        assert r.speaker == SELF_LABEL
        assert r.layer is AttributionLayer.L1_CHANNEL
        assert r.confidence == 1.0

    def test_system_is_others_bucket(self) -> None:
        r = AttributionResolver().resolve_channel("system")
        assert r.speaker == OTHERS_LABEL
        assert r.layer is AttributionLayer.L1_CHANNEL

    def test_unknown_source_empty(self) -> None:
        r = AttributionResolver().resolve_channel("")
        assert r.speaker == ""
        assert r.layer is AttributionLayer.NONE
        assert r.confidence == 0.0


class TestAcousticLayer:
    def test_diar_label_is_acoustic_estimate(self) -> None:
        r = AttributionResolver().resolve_acoustic("Speaker B")
        assert r.speaker == "Speaker B"
        assert r.layer is AttributionLayer.L3_ACOUSTIC

    def test_name_override_is_roster_layer(self) -> None:
        r = AttributionResolver().resolve_acoustic("Speaker B", names={"Speaker B": "Priya"})
        assert r.speaker == "Priya"
        assert r.layer is AttributionLayer.L4_ROSTER

    def test_empty_label_empty_result(self) -> None:
        r = AttributionResolver().resolve_acoustic(None)
        assert r.speaker == ""
        assert r.layer is AttributionLayer.NONE

    def test_conference_room_degrades_honestly(self) -> None:
        resolver = AttributionResolver(regime=Regime.CONFERENCE_ROOM)
        r = resolver.resolve_acoustic("Speaker B", names={"Speaker B": "Priya"})
        # Names are NOT trusted in a shared-far-field-mic regime.
        assert r.speaker == ROOM_LABEL
        assert r.low_confidence is True
        assert "conference-room" in r.note


class TestRegimeDetection:
    def test_shared_mic_large_roster_is_conference(self) -> None:
        resolver = AttributionResolver(roster=["a", "b", "c", "d", "e"])
        assert resolver.detect_regime(single_shared_mic=True) is Regime.CONFERENCE_ROOM

    def test_small_roster_is_solo(self) -> None:
        resolver = AttributionResolver(roster=["a", "b"])
        assert resolver.detect_regime(single_shared_mic=True) is Regime.SOLO_ENDPOINT

    def test_not_shared_mic_is_solo(self) -> None:
        resolver = AttributionResolver(roster=["a", "b", "c", "d", "e"])
        assert resolver.detect_regime(single_shared_mic=False) is Regime.SOLO_ENDPOINT

    def test_set_roster_and_regime(self) -> None:
        resolver = AttributionResolver()
        resolver.set_roster(["Ana", "Ben"])
        resolver.set_regime(Regime.SOLO_ENDPOINT)
        assert resolver.roster == ["Ana", "Ben"]
        assert resolver.regime is Regime.SOLO_ENDPOINT


class TestSessionIntegration:
    def test_default_resolver_reproduces_tier1(self) -> None:
        """Default resolver (unknown regime, empty roster) = prior You/Others."""
        resolver = AttributionResolver()
        assert resolver.resolve_channel("mic").speaker == "You"
        assert resolver.resolve_channel("system").speaker == "Others"
        # Diarization passthrough unchanged when not in a room.
        assert resolver.resolve_acoustic("Speaker C").speaker == "Speaker C"
