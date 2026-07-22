"""Local distiller backend — on-device model, no egress (F-702, ADR-001 step 3).

v1 uses the config-driven generation model (LFM2.5-2.6B per D-03) prompted for
the section → grounded-answer-unit task, reusing RAGAnswerGenerator's thread-safe
llama.cpp runtime (Metal, KV reset, <think> stripping). The forged fine-tuned
specialist (ADR-001 path step 2) will replace the prompt with trained behavior
behind this same interface.

Unlike the heuristic backend, the model reads the RAW section — tables and code
intact — and reshapes them into speakable prose (the "three levels" class of
answer lives in tables that clean_markdown strips).

Robustness: a section whose generation fails or comes back empty/refused falls
back to the heuristic unit for that section — the local backend can only add
quality over the heuristic floor, never lose content.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from lib.corpus.text import clean_markdown

logger = logging.getLogger(__name__)

MAX_SECTION_CHARS = 6000  # token budget guard for the prompt
MAX_ANSWER_TOKENS = 400  # one consolidated, speakable answer

# ChatML, mirrors the cloud consolidated instruction (lib/corpus/distiller.py).
# The empty <think></think> prefill keeps the reasoning 2.6B from stalling in
# chain-of-thought (same treatment as RAG_PROMPT_TEMPLATE, see D-03).
_PROMPT = """<|im_start|>system
You write ONE complete, self-contained answer that captures everything borrowable in a documentation section — a speaker could read it aloud to answer questions about this topic.

Rules:
- Ground it ONLY in the SECTION text. Do not add outside knowledge.
- Cover ALL the key facts: every item in a list, every level/option, every number and when-to-use — don't drop any.
- If the section contains a table, express the table's content as prose.
- Make it fully self-contained: name the subject, resolve pronouns, no "this"/"it" that refers outside the answer.
- Reply with just the answer prose. If the section is pure heading/navigation with no borrowable content, reply with exactly: NONE<|im_end|>
<|im_start|>user
SECTION: {heading}

{text}<|im_end|>
<|im_start|>assistant
<think></think>
"""

_REFUSAL_MARKERS = ("NONE", "I don't have")


def default_model_path() -> Path:
    """The configured generation model (config-driven; MODELS_DIR-resolved)."""
    from lib.config import load_config
    from lib.paths import get_models_dir

    model_file = load_config().models.generation.model_file
    return get_models_dir() / model_file


class LocalDistiller:
    """Prompted on-device section → answer-unit distiller (lazy model load)."""

    def __init__(self, model_path: Optional[Path] = None, n_ctx: int = 4096) -> None:
        from lib.rag_generator import RAGAnswerGenerator

        self.model_path = model_path or default_model_path()
        self._generator = RAGAnswerGenerator(self.model_path, n_ctx=n_ctx)

    def available(self) -> bool:
        """True when the model file exists on disk."""
        return self.model_path.exists()

    def distill_section(self, heading: str, text: str) -> list[str]:
        """One consolidated answer-unit for a section, or [] for navigation stubs.

        Falls back to the heuristic unit if generation fails or refuses, so the
        local backend never yields less than the heuristic floor.
        """
        from lib.corpus.distiller import MIN_SECTION_WORDS, _distill_heuristic

        if len(text.split()) < MIN_SECTION_WORDS:
            return []
        prompt = _PROMPT.format(heading=heading, text=text[:MAX_SECTION_CHARS])
        answer = self._generator.generate_text(prompt, max_tokens=MAX_ANSWER_TOKENS)
        # Model output is for reading aloud — strip any markdown it leaked
        # (leading heading hashes, emphasis) down to plain prose.
        answer = clean_markdown(answer)
        if not answer or any(answer.startswith(m) for m in _REFUSAL_MARKERS):
            if not clean_markdown(text).split():
                return []  # genuinely empty after cleaning (e.g. pure table the model refused)
            logger.warning("local distill empty for %r — heuristic fallback", heading)
            return _distill_heuristic(heading, text, "consolidated")
        return [answer]


_instance: Optional[LocalDistiller] = None


def get_local_distiller() -> LocalDistiller:
    """Process-wide instance so the model loads once, not once per document."""
    global _instance
    if _instance is None:
        override = os.environ.get("CORPUS_LOCAL_MODEL", "")
        _instance = LocalDistiller(Path(override) if override else None)
    return _instance
