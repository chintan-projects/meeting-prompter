"""Tests for lib.filters — two-level filtering and normalization."""
import pytest

from lib.filters import is_hallucination, is_hallucination_only, is_noise, normalize_text


class TestIsHallucination:
    """Test hallucination detection patterns."""

    def test_hallucination_starters(self) -> None:
        assert is_hallucination("I don't know what this is about") is True
        assert is_hallucination("She chose to leave early") is True
        assert is_hallucination("He chose the other option") is True
        assert is_hallucination("I think it's going to work") is True

    def test_third_person_narration(self) -> None:
        assert is_hallucination("She went to the store") is True
        assert is_hallucination("He said something about it") is True
        assert is_hallucination("They chose the first option") is True
        assert is_hallucination("It was a beautiful day") is True

    def test_vague_questions(self) -> None:
        assert is_hallucination("Can you explain to me?") is True
        assert is_hallucination("Can you tell me?") is True
        assert is_hallucination("What do you mean?") is True

    def test_repetitive_phrases(self) -> None:
        assert is_hallucination("the cat sat on the cat sat on") is True
        assert is_hallucination("we need to we need to we need to") is True

    def test_valid_speech_not_hallucination(self) -> None:
        assert is_hallucination("What's the deployment timeline for the SDK?") is False
        assert is_hallucination("We should review the quarterly numbers") is False
        assert is_hallucination("Let me share my screen") is False
        assert is_hallucination("Can we move on to the next topic?") is False


class TestIsHallucinationOnly:
    """Test the light transcript filter — only ASR artifacts."""

    def test_empty_is_hallucination(self) -> None:
        assert is_hallucination_only("") is True
        assert is_hallucination_only("  ") is True

    def test_hallucination_caught(self) -> None:
        assert is_hallucination_only("She chose to leave early") is True
        assert is_hallucination_only("I don't know what happened") is True

    def test_short_valid_speech_passes(self) -> None:
        """Short utterances should NOT be filtered by this function."""
        assert is_hallucination_only("Yeah") is False
        assert is_hallucination_only("Ok") is False
        assert is_hallucination_only("Exactly") is False
        assert is_hallucination_only("I agree") is False
        assert is_hallucination_only("No") is False

    def test_filler_words_pass(self) -> None:
        """Filler words should NOT be filtered (transcript shows everything)."""
        assert is_hallucination_only("um") is False
        assert is_hallucination_only("uh huh") is False
        assert is_hallucination_only("right right") is False

    def test_normal_speech_passes(self) -> None:
        assert is_hallucination_only("What about the Q2 timeline?") is False
        assert is_hallucination_only("Let me check that") is False


class TestIsNoise:
    """Test the strict trigger pipeline filter."""

    def test_empty_is_noise(self) -> None:
        assert is_noise("") is True
        assert is_noise("  ") is True

    def test_hallucinations_caught(self) -> None:
        assert is_noise("She chose the first option") is True
        assert is_noise("I don't know what this is about") is True

    def test_noise_phrases_caught(self) -> None:
        assert is_noise("yeah") is True
        assert is_noise("um") is True
        assert is_noise("uh huh") is True
        assert is_noise("okay") is True
        assert is_noise("oh well") is True

    def test_filler_heavy_caught(self) -> None:
        """Text with only filler words should be noise."""
        assert is_noise("yeah um okay") is True
        assert is_noise("uh so like") is True

    def test_short_filler_only(self) -> None:
        """Single filler word is noise for triggers."""
        assert is_noise("Yeah") is True
        assert is_noise("Ok") is True
        assert is_noise("Right") is True

    def test_substantive_speech_passes(self) -> None:
        """Real questions and statements should pass the strict filter."""
        assert is_noise("What's the deployment timeline?") is False
        assert is_noise("We should review the quarterly numbers") is False
        assert is_noise("The release is scheduled for next week") is False

    def test_mixed_content_with_enough_substance(self) -> None:
        """Filler + meaningful words should pass if enough meaningful content."""
        assert is_noise("yeah so the deployment timeline is next week") is False


class TestFilterLevelDifference:
    """Verify the gap between is_hallucination_only and is_noise."""

    def test_short_speech_passes_hallucination_but_caught_by_noise(self) -> None:
        """Short valid speech: passes transcript filter, caught by trigger filter."""
        short_utterances = ["Yeah", "Ok", "Right", "Sure", "No"]
        for text in short_utterances:
            assert is_hallucination_only(text) is False, f"{text} should pass transcript filter"
            assert is_noise(text) is True, f"{text} should be caught by trigger filter"

    def test_substantive_speech_passes_both(self) -> None:
        """Real speech passes both filters."""
        texts = [
            "What about the Q2 release timeline?",
            "We need to discuss the budget allocation",
            "Let me share the latest metrics from the dashboard",
        ]
        for text in texts:
            assert is_hallucination_only(text) is False, f"{text} should pass transcript"
            assert is_noise(text) is False, f"{text} should pass trigger"

    def test_hallucinations_caught_by_both(self) -> None:
        """ASR artifacts caught by both filters."""
        texts = [
            "She chose to leave early",
            "I don't know what happened here",
        ]
        for text in texts:
            assert is_hallucination_only(text) is True, f"{text} should fail transcript"
            assert is_noise(text) is True, f"{text} should fail trigger"


class TestNormalizeText:
    """Test text normalization."""

    def test_duplicate_word_removal(self) -> None:
        assert normalize_text("the the cat") == "The cat"

    def test_mishearing_correction(self) -> None:
        assert "Liquid" in normalize_text("L Those product")

    def test_whitespace_cleanup(self) -> None:
        assert normalize_text("  hello   world  ") == "Hello world"

    def test_question_detection(self) -> None:
        result = normalize_text("how does this work")
        assert result.endswith("?")

    def test_preserves_existing_punctuation(self) -> None:
        result = normalize_text("What about the timeline?")
        assert result.endswith("?")
        assert result.count("?") == 1

    def test_capitalization(self) -> None:
        assert normalize_text("hello world") == "Hello world"

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""
