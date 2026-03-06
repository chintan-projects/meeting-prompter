"""Tests for lib.text_refiner — LFM-based text cleanup."""
from unittest.mock import MagicMock, patch

import pytest

from lib.text_refiner import TextRefiner, _CLEANUP_PROMPT, _STOP_TOKENS


@pytest.fixture
def mock_generator() -> MagicMock:
    """Mock RAGAnswerGenerator with a mock Llama instance."""
    gen = MagicMock()
    gen.llm.return_value = {
        "choices": [{"text": "Polished output text here."}],
    }
    return gen


@pytest.fixture
def refiner(mock_generator: MagicMock) -> TextRefiner:
    """TextRefiner with a mock generator."""
    return TextRefiner(mock_generator, min_words_to_refine=5, max_tokens_ratio=1.5)


class TestRefineBasic:
    """Basic refinement behavior."""

    def test_refine_returns_polished_text(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Refine should return cleaned text from the model."""
        result = refiner.refine("This is some raw asr text that needs cleanup")
        assert result == "Polished output text here."
        mock_generator.load.assert_called_once()
        mock_generator._reset_state.assert_called_once()

    def test_refine_calls_llm_with_correct_params(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Refine should call llm with the cleanup prompt and correct settings."""
        refiner.refine("Some raw asr text that definitely needs cleaning up")

        call_args = mock_generator.llm.call_args
        prompt = call_args[0][0]
        assert "Clean up this raw speech transcript" in prompt
        assert "Some raw asr text" in prompt
        assert call_args[1]["stop"] == _STOP_TOKENS
        assert call_args[1]["temperature"] == 0

    def test_refine_max_tokens_based_on_input(
        self, refiner: TextRefiner, mock_generator: MagicMock
    ) -> None:
        """Max tokens should be proportional to input word count."""
        # 8 words * 1.5 ratio = 12, but min is 100
        refiner.refine("one two three four five six seven eight")
        call_args = mock_generator.llm.call_args
        assert call_args[1]["max_tokens"] == 100  # min(12, 100) -> 100

        # 100 words * 1.5 = 150 > 100
        long_text = " ".join(["word"] * 100)
        mock_generator.llm.return_value = {"choices": [{"text": " ".join(["polished"] * 80)}]}
        refiner.refine(long_text)
        call_args = mock_generator.llm.call_args
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
        mock_generator.llm.assert_not_called()

    def test_exactly_at_min_words(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Text with exactly min_words should be refined."""
        result = refiner.refine("one two three four five")
        assert result == "Polished output text here."
        mock_generator.llm.assert_called_once()


class TestRefineFallback:
    """Refine should return original text on any failure."""

    def test_empty_model_output(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Empty model output returns original."""
        mock_generator.llm.return_value = {"choices": [{"text": ""}]}
        result = refiner.refine("This is some raw text from the ASR model")
        assert result == "This is some raw text from the ASR model"

    def test_model_exception(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Exception during refinement returns original."""
        mock_generator.llm.side_effect = RuntimeError("Model crashed")
        result = refiner.refine("This is some raw text from the ASR model")
        assert result == "This is some raw text from the ASR model"

    def test_suspicious_length_too_short(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Polished text much shorter than original returns original."""
        mock_generator.llm.return_value = {"choices": [{"text": "Hi"}]}
        original = "This is a long sentence that should not be shortened to just one word"
        result = refiner.refine(original)
        assert result == original

    def test_suspicious_length_too_long(self, refiner: TextRefiner, mock_generator: MagicMock) -> None:
        """Polished text much longer than original returns original."""
        mock_generator.llm.return_value = {"choices": [{"text": " ".join(["word"] * 100)}]}
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
