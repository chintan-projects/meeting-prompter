"""Tests for lib.text_refiner — LFM-based text cleanup."""
from unittest.mock import MagicMock

import pytest

from lib.text_refiner import TextRefiner, _CLEANUP_PROMPT, _STOP_TOKENS


@pytest.fixture
def mock_generator() -> MagicMock:
    """Mock RAGAnswerGenerator with generate_text()."""
    gen = MagicMock()
    gen.generate_text.return_value = "Polished output text here."
    return gen


@pytest.fixture
def refiner(mock_generator: MagicMock) -> TextRefiner:
    """TextRefiner with a mock generator."""
    return TextRefiner(mock_generator, min_words_to_refine=5, max_tokens_ratio=1.5)


class TestRefineBasic:
    """Basic refinement behavior."""

    def test_refine_returns_polished_text(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Refine should return cleaned text from the model."""
        result = refiner.refine("This is some raw asr text that needs cleanup")
        assert result == "Polished output text here."
        mock_generator.generate_text.assert_called_once()

    def test_refine_calls_generate_text_with_correct_params(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Refine should call generate_text with the cleanup prompt and correct settings."""
        refiner.refine("Some raw asr text that definitely needs cleaning up")

        call_args = mock_generator.generate_text.call_args
        prompt = call_args[0][0]
        assert "Clean up this raw speech transcript" in prompt
        assert "Some raw asr text" in prompt
        assert call_args[1]["stop"] == _STOP_TOKENS

    def test_refine_max_tokens_based_on_input(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Max tokens should be proportional to input word count."""
        # 8 words * 1.5 ratio = 12, but min is 100
        refiner.refine("one two three four five six seven eight")
        call_args = mock_generator.generate_text.call_args
        assert call_args[1]["max_tokens"] == 100  # min(12, 100) -> 100

        # 100 words * 1.5 = 150 > 100
        mock_generator.generate_text.return_value = " ".join(["polished"] * 80)
        long_text = " ".join(["word"] * 100)
        refiner.refine(long_text)
        call_args = mock_generator.generate_text.call_args
        assert call_args[1]["max_tokens"] == 150


class TestRefineSkipShort:
    """Short text should be returned unchanged."""

    def test_empty_string(self, refiner: TextRefiner) -> None:
        assert refiner.refine("") == ""

    def test_whitespace_only(self, refiner: TextRefiner) -> None:
        assert refiner.refine("   ") == ""

    def test_below_min_words(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Text with fewer than min_words_to_refine should pass through."""
        result = refiner.refine("Hello there")
        assert result == "Hello there"
        mock_generator.generate_text.assert_not_called()

    def test_exactly_at_min_words(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Text with exactly min_words should be refined."""
        result = refiner.refine("one two three four five")
        assert result == "Polished output text here."
        mock_generator.generate_text.assert_called_once()


class TestRefineFallback:
    """Refine should return original text on any failure."""

    def test_empty_model_output(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Empty model output returns original."""
        mock_generator.generate_text.return_value = ""
        result = refiner.refine("This is some raw text from the ASR model")
        assert result == "This is some raw text from the ASR model"

    def test_suspicious_length_too_short(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Polished text much shorter than original returns original."""
        mock_generator.generate_text.return_value = "Hi"
        original = "This is a long sentence that should not be shortened to just one word"
        result = refiner.refine(original)
        assert result == original

    def test_suspicious_length_too_long(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Polished text much longer than original returns original."""
        mock_generator.generate_text.return_value = " ".join(["word"] * 100)
        original = "Short sentence to refine here please"
        result = refiner.refine(original)
        assert result == original


class TestRefinePromptFormat:
    """Verify the ChatML prompt structure."""

    def test_prompt_contains_chatml_tags(self) -> None:
        """Cleanup prompt should use ChatML format."""
        assert "<|im_start|>system" in _CLEANUP_PROMPT
        assert "<|im_end|>" in _CLEANUP_PROMPT
        assert "<|im_start|>user" in _CLEANUP_PROMPT
        assert "<|im_start|>assistant" in _CLEANUP_PROMPT

    def test_prompt_has_placeholder(self) -> None:
        """Prompt should contain {raw_text} placeholder."""
        assert "{raw_text}" in _CLEANUP_PROMPT

    def test_stop_tokens_defined(self) -> None:
        """Stop tokens should prevent runaway generation."""
        assert "<|im_end|>" in _STOP_TOKENS
        assert "<|im_start|>" in _STOP_TOKENS


class TestMetaNarrationGuard:
    """The refiner shares the 2.6B reasoning model (D-07), which narrates its
    edits ("Fixed version with corrections:") straight into the transcript —
    observed live 2026-07-22. The transcript is a record the user exports, so
    raw ASR beats corrupted prose.
    """

    def test_detects_fixed_version_preamble(self) -> None:
        from lib.text_refiner import looks_like_meta

        assert looks_like_meta('Fixed version with corrections:\n"The first test was."') is True

    def test_detects_here_is_the_preamble(self) -> None:
        from lib.text_refiner import looks_like_meta

        assert looks_like_meta("Here's the cleaned transcript: we shipped it.") is True

    def test_clean_output_passes(self) -> None:
        from lib.text_refiner import looks_like_meta

        assert looks_like_meta("The first test on what can be on display is this one.") is False

    def test_refine_keeps_raw_text_when_model_narrates(self) -> None:
        from lib.text_refiner import TextRefiner

        raw = "the first test on what can be on display is this one it's sweet"

        class _Narrating:
            def generate_text(self, *a: object, **k: object) -> str:
                return f'Fixed version with corrections:\n"{raw.capitalize()}."'

        refiner = TextRefiner(_Narrating())  # type: ignore[arg-type]
        assert refiner.refine(raw) == raw

    def test_refine_still_accepts_a_clean_edit(self) -> None:
        from lib.text_refiner import TextRefiner

        raw = "the first test on what can be on display is this one it's sweet"
        polished = "The first test on what can be on display is this one. It's sweet."

        class _Clean:
            def generate_text(self, *a: object, **k: object) -> str:
                return polished

        refiner = TextRefiner(_Clean())  # type: ignore[arg-type]
        assert refiner.refine(raw) == polished
