"""RAG Answer Generator using LFM2.5-1.2B-Instruct (or LFM2-1.2B-RAG fallback).

Uses ChatML format with a system message for grounded answer generation.
LFM2.5-1.2B-Instruct has 32K context and strong instruction following (IFEval 86.23).
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# ChatML prompt with system message — LFM2.5-Instruct handles system messages well
RAG_PROMPT_TEMPLATE = """<|im_start|>system
You are a meeting intelligence assistant. Answer questions using ONLY the provided context. Be concise and direct (2-3 sentences max). If the context does not contain the answer, say "I don't have that information in my documents."<|im_end|>
<|im_start|>user
CONTEXT:
{context}

QUESTION: {question}<|im_end|>
<|im_start|>assistant
"""


class RAGAnswerGenerator:
    """Generates answers using an LFM model with grounded RAG context.

    Args:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size (default 4096 for LFM2.5, was 2048 for LFM2).
        max_context_chars: Max characters of RAG context to include (default 6000).
        max_question_chars: Max characters for the question (default 500).
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        max_context_chars: int = 6000,
        max_question_chars: int = 500,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.max_context_chars = max_context_chars
        self.max_question_chars = max_question_chars
        self.llm: Optional[Llama] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """
        Lazy load the model.

        The model is loaded on first use to save memory during startup.
        Uses Metal GPU acceleration on Mac for faster inference.
        """
        if self.llm is None:
            # Make embedded chat-template compilation non-fatal so newer LFM2.5
            # GGUFs (e.g. 2.6B) whose templates use HF jinja extensions still load
            # via raw completion. See lib/llama_compat.
            from lib.llama_compat import install as _install_llama_compat

            _install_llama_compat()
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=-1,  # Use Metal GPU on Mac
                verbose=False,
            )

    def _reset_state(self) -> None:
        """
        Reset model state before each generation to prevent KV cache issues.

        The llama_decode error can occur when the KV cache gets corrupted.
        Resetting before each call prevents this.
        """
        if self.llm is not None:
            try:
                self.llm.reset()
            except Exception:
                # If reset fails, reload the model
                self.llm = None
                self.load()

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 200,
        stop: Optional[List[str]] = None,
        temperature: float = 0,
        top_p: float = 1.0,
    ) -> str:
        """Thread-safe LLM text generation.

        Acquires the internal lock, then runs the full load → reset → generate
        sequence atomically. All callers (ModeAwareGenerator, TextRefiner,
        notes_generator) should use this instead of accessing .llm directly.

        Args:
            prompt: Complete prompt string (caller builds the ChatML).
            max_tokens: Maximum tokens in response.
            stop: Stop sequences (defaults to ChatML delimiters).
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.

        Returns:
            Raw generated text (stripped), or empty string on failure.
        """
        if stop is None:
            stop = ["<|im_end|>", "<|im_start|>"]

        with self._lock:
            self.load()
            self._reset_state()

            try:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response["choices"][0]["text"].strip()
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                return ""

    def generate(
        self,
        question: str,
        context: str,
        max_tokens: int = 200,
    ) -> str:
        """Generate an answer from question and grounded context.

        Args:
            question: The user's question.
            context: Pre-extracted/grounded context from retrieval.
            max_tokens: Maximum tokens in response (default 200).

        Returns:
            Generated answer string, or error message on failure.
        """
        truncated_context = context[: self.max_context_chars]
        truncated_question = question[: self.max_question_chars]

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=truncated_context,
            question=truncated_question,
        )

        answer = self.generate_text(
            prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|im_start|>", "\n\nQUESTION:", "\n\nCONTEXT:", "---"],
        )

        if not answer:
            return "[Unable to generate answer]"

        return self._clean_answer(answer)

    def _clean_answer(self, answer: str) -> str:
        """Clean up generated answer: trailing sentences, whitespace."""
        if not answer:
            return answer

        # Remove excessive whitespace
        answer = " ".join(answer.split())

        # If answer ends mid-sentence (no punctuation), try to truncate cleanly
        if answer and answer[-1] not in ".!?:":
            # Find last complete sentence
            for punct in [". ", "! ", "? "]:
                last_idx = answer.rfind(punct)
                if last_idx > len(answer) * 0.5:  # Keep at least half
                    answer = answer[: last_idx + 1]
                    break

        return answer


def test_rag_generator() -> None:
    """Test the RAG generator with sample context."""
    import os

    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    model_path = models_dir / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    if not model_path.exists():
        # Fallback to legacy model
        model_path = models_dir / "LFM2-1.2B-RAG-Q4_K_M.gguf"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print("Loading RAG generator...")
    gen = RAGAnswerGenerator(model_path)
    gen.load()
    print("Model loaded!")

    # Test with sample context
    context = """## The LEAP Platform Components

LEAP stands for Liquid Edge AI Platform and transforms cutting-edge AI research
into deployable business solutions. The platform includes a Model Library with
models ranging from LFM2-350M through 8.3B with optimized variants. The Fine-Tuning
CLI provides LoRA adapters, data pipeline tools, training infrastructure, and an
evaluation suite. The Edge SDK is cross-platform supporting macOS, Windows, Linux,
iOS, and Android."""

    question = "What is LEAP?"

    print(f"\nQuestion: {question}")
    print(f"Context: {context[:100]}...")
    print("\nGenerating answer...")

    answer = gen.generate(question, context)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    test_rag_generator()
