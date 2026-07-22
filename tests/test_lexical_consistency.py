"""Tests for the lexical speaker-consistency pass (F-607)."""

from __future__ import annotations

from dataclasses import dataclass

from lib.attribution import LexicalConsistencyPass, SpeakerCorrection


@dataclass
class _Turn:
    id: str
    text: str
    speaker: str
    source: str


def _turns(*rows: tuple) -> list:
    return [_Turn(id=f"t{i}", text=t, speaker=s, source=src) for i, (t, s, src) in enumerate(rows)]


ROSTER = ["Priya (Eng)", "Raj", "Ana Costa (PM)"]


class TestHandoffCue:
    def test_handoff_names_next_remote_turn(self) -> None:
        turns = _turns(
            ("So, over to you, Priya.", "You", "mic"),
            ("Sure, here's my update.", "Others", "system"),
        )
        out = LexicalConsistencyPass(ROSTER).analyze(turns)
        assert len(out) == 1
        assert out[0].turn_id == "t1"
        assert out[0].suggested_speaker == "Priya"

    def test_question_handoff(self) -> None:
        turns = _turns(
            ("Raj, what do you think?", "You", "mic"),
            ("I think we should ship.", "Speaker A", "system"),
        )
        out = LexicalConsistencyPass(ROSTER).analyze(turns)
        assert out and out[0].suggested_speaker == "Raj"
        assert out[0].turn_id == "t1"

    def test_no_correction_if_next_already_named(self) -> None:
        turns = _turns(
            ("Over to you, Priya.", "You", "mic"),
            ("Sure.", "Priya", "system"),  # already named → not generic
        )
        assert LexicalConsistencyPass(ROSTER).analyze(turns) == []

    def test_no_correction_if_next_is_mic(self) -> None:
        turns = _turns(
            ("Over to you, Priya.", "You", "mic"),
            ("Actually let me add.", "You", "mic"),
        )
        assert LexicalConsistencyPass(ROSTER).analyze(turns) == []


class TestGratitudeCue:
    def test_gratitude_names_previous_remote_turn(self) -> None:
        turns = _turns(
            ("...and that's the plan.", "Others", "system"),
            ("Thanks, Raj.", "You", "mic"),
        )
        out = LexicalConsistencyPass(ROSTER).analyze(turns)
        assert len(out) == 1
        assert out[0].turn_id == "t0"
        assert out[0].suggested_speaker == "Raj"

    def test_multiword_roster_name(self) -> None:
        turns = _turns(
            ("Here is the roadmap.", "Others", "system"),
            ("Good point, Ana.", "You", "mic"),
        )
        out = LexicalConsistencyPass(ROSTER).analyze(turns)
        assert out and out[0].suggested_speaker == "Ana Costa"


class TestConservatism:
    def test_unknown_name_ignored(self) -> None:
        turns = _turns(
            ("Over to you, Dave.", "You", "mic"),  # not on roster
            ("Hello.", "Others", "system"),
        )
        assert LexicalConsistencyPass(ROSTER).analyze(turns) == []

    def test_empty_roster_no_corrections(self) -> None:
        turns = _turns(
            ("Over to you, Priya.", "You", "mic"),
            ("Sure.", "Others", "system"),
        )
        assert LexicalConsistencyPass([]).analyze(turns) == []

    def test_returns_correction_type(self) -> None:
        turns = _turns(
            ("Over to you, Priya.", "You", "mic"),
            ("Sure.", "Others", "system"),
        )
        out = LexicalConsistencyPass(ROSTER).analyze(turns)
        assert isinstance(out[0], SpeakerCorrection)
        assert 0.0 < out[0].confidence <= 1.0
        assert out[0].reason


# ─── correct_segments (notes-time application) ───────────────────────────

from lib.attribution import correct_segments  # noqa: E402


class TestCorrectSegments:
    def _segs(self):
        return [
            {"id": "t0", "text": "Over to you, Priya.", "speaker": "You", "source": "mic"},
            {"id": "t1", "text": "Here's my update.", "speaker": "Others", "source": "system"},
        ]

    def test_applies_correction_non_destructive(self) -> None:
        original = self._segs()
        out = correct_segments(original, ROSTER)
        assert out[1]["speaker"] == "Priya"
        assert out[1]["low_confidence"] is True
        # original untouched
        assert original[1]["speaker"] == "Others"

    def test_no_roster_returns_input(self) -> None:
        segs = self._segs()
        assert correct_segments(segs, []) is segs

    def test_no_cue_returns_input(self) -> None:
        segs = [
            {"id": "t0", "text": "Hello everyone.", "speaker": "You", "source": "mic"},
            {"id": "t1", "text": "Hi.", "speaker": "Others", "source": "system"},
        ]
        assert correct_segments(segs, ROSTER) is segs
