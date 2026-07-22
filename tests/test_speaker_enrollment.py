"""Tests for voice enrollment (F-605) and its diarizer integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.config import DiarizationConfig
from lib.diarization import SpeakerDiarizer
from lib.speaker_enrollment import VoiceEnrollment


def _emb(seed: int, dim: int = 192) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestVoiceEnrollmentStore:
    def test_enroll_and_identify(self) -> None:
        enr = VoiceEnrollment(match_threshold=0.7)
        enr.enroll("Priya", _emb(1))
        enr.enroll("Raj", _emb(500))
        assert enr.identify(_emb(1)) == "Priya"
        assert enr.identify(_emb(500)) == "Raj"

    def test_unknown_voice_returns_none(self) -> None:
        enr = VoiceEnrollment(match_threshold=0.9)
        enr.enroll("Priya", _emb(1))
        assert enr.identify(_emb(999)) is None

    def test_empty_store_identifies_none(self) -> None:
        assert VoiceEnrollment().identify(_emb(1)) is None

    def test_enroll_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            VoiceEnrollment().enroll("  ", _emb(1))

    def test_reenroll_refines_profile(self) -> None:
        enr = VoiceEnrollment()
        enr.enroll("Priya", _emb(1))
        enr.enroll("Priya", _emb(2))  # second sample averaged in
        assert enr.names == ["Priya"]
        assert len(enr) == 1

    def test_remove(self) -> None:
        enr = VoiceEnrollment()
        enr.enroll("Priya", _emb(1))
        assert enr.remove("Priya") is True
        assert enr.remove("Priya") is False
        assert "Priya" not in enr

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        enr = VoiceEnrollment(match_threshold=0.6)
        enr.enroll("Priya", _emb(1))
        enr.enroll("Raj", _emb(500))
        path = tmp_path / "profiles.json"
        enr.save(path)

        loaded = VoiceEnrollment.load(path, match_threshold=0.6)
        assert set(loaded.names) == {"Priya", "Raj"}
        assert loaded.identify(_emb(1)) == "Priya"

    def test_load_missing_file_is_empty(self, tmp_path: Path) -> None:
        loaded = VoiceEnrollment.load(tmp_path / "nope.json")
        assert len(loaded) == 0

    def test_load_corrupt_file_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert len(VoiceEnrollment.load(path)) == 0


class TestDiarizerEnrollmentIntegration:
    @pytest.fixture
    def diarizer(self) -> SpeakerDiarizer:
        cfg = DiarizationConfig(enabled=True, max_speakers=4, similarity_threshold=0.65)
        with patch.object(SpeakerDiarizer, "_load_model"):
            d = SpeakerDiarizer(cfg)
            d._classifier = MagicMock()
        return d

    def test_enrolled_voice_gets_real_name(self, diarizer: SpeakerDiarizer) -> None:
        enr = VoiceEnrollment(match_threshold=0.7)
        enr.enroll("Priya", _emb(1))
        diarizer.set_enrollment(enr)
        # A voice matching Priya's profile is labeled "Priya", not "Speaker A".
        assert diarizer._assign_speaker(_emb(1)) == "Priya"

    def test_unenrolled_voice_stays_anonymous(self, diarizer: SpeakerDiarizer) -> None:
        enr = VoiceEnrollment(match_threshold=0.9)
        enr.enroll("Priya", _emb(1))
        diarizer.set_enrollment(enr)
        assert diarizer._assign_speaker(_emb(999)) == "Speaker A"

    def test_mixed_named_and_anonymous(self, diarizer: SpeakerDiarizer) -> None:
        enr = VoiceEnrollment(match_threshold=0.7)
        enr.enroll("Priya", _emb(1))
        diarizer.set_enrollment(enr)
        first = diarizer._assign_speaker(_emb(1))  # Priya (enrolled)
        second = diarizer._assign_speaker(_emb(500))  # unknown → Speaker B
        assert first == "Priya"
        assert second == "Speaker B"
        assert diarizer.get_speaker_summary().get("Priya") == 1

    def test_reset_clears_names(self, diarizer: SpeakerDiarizer) -> None:
        enr = VoiceEnrollment(match_threshold=0.7)
        enr.enroll("Priya", _emb(1))
        diarizer.set_enrollment(enr)
        diarizer._assign_speaker(_emb(1))
        diarizer.reset()
        assert diarizer.speaker_count == 0
        assert diarizer._names == []

    def test_no_enrollment_all_anonymous(self, diarizer: SpeakerDiarizer) -> None:
        assert diarizer._assign_speaker(_emb(1)) == "Speaker A"
        assert diarizer._assign_speaker(_emb(500)) == "Speaker B"
