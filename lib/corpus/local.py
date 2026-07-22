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
import re
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

# The 2.6B is a reasoning model: on ~58% of sections it narrated its plan ("Okay,
# I need to write a complete, self-contained answer...") instead of answering, and
# an empty <think></think> prefill does not reliably suppress it (prompt hardening
# was tried and produced different meta-text, not less). Those strings are not
# answers, and a corpus full of them silently poisons retrieval and coverage — so
# a unit that reads as task-narration is REJECTED and the section falls back to
# the heuristic floor. The reject rate is reported in the distill stats; a high
# rate is the empirical case for forging a fine-tuned distiller (F-702 v2).
# This is a CONTRACT check, not just a style filter: an answer-unit must be
# self-contained (the prompt's own rule), so text that narrates the task ("I need
# to write..."), frames the artifact ("The following is a complete answer...") or
# points outside itself ("the section describes...") is never a valid unit — a
# speaker cannot read it aloud to answer anything.
_META_PATTERN = re.compile(
    r"\b(?:"
    r"i (?:need to|have to|am going to|want to|must|should|will|shall)\b|i'll|let me|okay,|"
    r"the user (?:wants|is asking)|my task|"  # task narration
    r"the following is|here'?s? (?:is )?(?:a|the) (?:complete|answer|self)|"
    r"this (?:is a|section)|a speaker-friendly|speaker could read|"  # artifact framing
    r"the(?:\s+\w+){0,2}\s+(?:section|text|passage|document|provided)\b"
    r"(?:\s+\w+){0,2}\s+(?:describes|explains|states|says|provides|contains|title)|"
    r"based on the (?:section|provided)|self-contained (?:answer|documentation|explanation)|"
    r"borrowable"
    r")",
    re.IGNORECASE,
)
META_SCAN_CHARS = 220  # narration and framing, when they happen, are in the opening


def looks_like_meta(text: str) -> bool:
    """True when the unit narrates/frames the task instead of being the answer.

    Enforces the self-containment contract: a unit that talks *about* an answer,
    or refers to "the section" it came from, is not borrowable.
    """
    return bool(_META_PATTERN.search(" ".join(text.split())[:META_SCAN_CHARS]))


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
        #: Per-run counts. `rejected` is the headline quality signal for the
        #: prompted model — a high rate is the case for forging a specialist.
        self.stats: dict[str, int] = {"model": 0, "rejected": 0, "empty": 0}

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
            self.stats["empty"] += 1
            logger.warning("local distill empty for %r — heuristic fallback", heading)
            return _distill_heuristic(heading, text, "consolidated")
        if looks_like_meta(answer):
            # The model narrated the task instead of answering it. Shipping this
            # would poison retrieval with text no speaker could ever read aloud.
            self.stats["rejected"] += 1
            logger.warning(
                "local distill rejected (task narration, not an answer) for %r: %r "
                "— heuristic fallback",
                heading,
                answer[:80],
            )
            return _distill_heuristic(heading, text, "consolidated")
        self.stats["model"] += 1
        return [answer]


_instance: Optional[LocalDistiller] = None


def get_local_distiller() -> LocalDistiller:
    """Process-wide instance so the model loads once, not once per document."""
    global _instance
    if _instance is None:
        override = os.environ.get("CORPUS_LOCAL_MODEL", "")
        _instance = LocalDistiller(Path(override) if override else None)
    return _instance
