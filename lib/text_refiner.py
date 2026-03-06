"""Text Refiner — cleans raw ASR text using LFM2.5-1.2B-Instruct.

Shares the Llama model instance with RAGAnswerGenerator to avoid
loading the model twice. Runs in the pipeline thread after a speech
turn is finalized, polishing grammar, fixing mishearings, and
removing ASR artifacts.

The refiner is conservative: it preserves meaning, speaking style,
and speaker intent. On any failure, it returns the original text.
"""
import logging
import time
from typing import Optional

from lib.rag_generator import RAGAnswerGenerator

logger = logging.getLogger(__name__)

# ChatML prompt for transcript cleanup — conservative by design
_CLEANUP_PROMPT = """<|im_start|>system
You are a transcript cleanup assistant. Your job is to fix a raw speech-to-text transcript.

Rules:
- Fix grammar and punctuation errors
- Remove repeated or stuttered words
- Correct obvious mishearings based on context
- Keep the speaker's original meaning, tone, and vocabulary
- Do NOT summarize, add information, or change what was said
- Do NOT add commentary or explanations
- Output ONLY the cleaned transcript text, nothing else<|im_end|>
<|im_start|>user
Clean up this raw speech transcript:

{raw_text}<|im_end|>
<|im_start|>assistant
"""

_STOP_TOKENS = [
    "<|im_end|>",
    "<|im_start|>",
]


class TextRefiner:
    """Cleans raw ASR text using LFM2.5-1.2B-Instruct.

    Shares the Llama model instance with the RAG generator to avoid
    double-loading. Sequential access in the pipeline thread means
    no locking is needed.

    Args:
        generator: RAGAnswerGenerator instance whose Llama model to share.
        min_words_to_refine: Skip refinement for very short turns.
        max_tokens_ratio: Max output tokens = input words * ratio.
    """

    def __init__(
        self,
        generator: RAGAnswerGenerator,
        min_words_to_refine: int = 5,
        max_tokens_ratio: float = 1.5,
    ) -> None:
        self._generator = generator
        self._min_words = min_words_to_refine
        self._max_tokens_ratio = max_tokens_ratio

    def refine(self, raw_text: str) -> str:
        """Clean up raw ASR text. Returns polished text or original on failure.

        Short turns (below min_words_to_refine) are returned unchanged
        since there isn't enough context for meaningful cleanup.
        """
        raw_text = raw_text.strip()
        if not raw_text:
            return raw_text

        word_count = len(raw_text.split())
        if word_count < self._min_words:
            return raw_text

        start = time.time()

        try:
            self._generator.load()
            self._generator._reset_state()

            prompt = _CLEANUP_PROMPT.format(raw_text=raw_text)
            max_tokens = max(int(word_count * self._max_tokens_ratio), 100)

            response = self._generator.llm(
                prompt,
                max_tokens=max_tokens,
                stop=_STOP_TOKENS,
                temperature=0,
                top_p=1.0,
            )

            polished = response["choices"][0]["text"].strip()
            elapsed_ms = (time.time() - start) * 1000

            if not polished:
                logger.debug("Refiner returned empty — keeping original")
                return raw_text

            # Sanity check: if polished is way shorter or longer, keep original
            polished_words = len(polished.split())
            if polished_words < word_count * 0.3 or polished_words > word_count * 3:
                logger.warning(
                    "Refiner output length suspicious (%d→%d words) — keeping original",
                    word_count,
                    polished_words,
                )
                return raw_text

            logger.info(
                "Refined %d→%d words in %.0fms",
                word_count,
                polished_words,
                elapsed_ms,
            )
            return polished

        except Exception as e:
            logger.error("Text refinement failed: %s — keeping original", e)
            return raw_text
