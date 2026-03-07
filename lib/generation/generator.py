"""Mode-aware generator — routes triggers to appropriate prompt templates.

Replaces hybrid_answerer.py with trigger-type-aware generation:
- Selects prompt template by TriggerType
- Context budget: 30% conversation, 70% RAG context
- Keeps extraction fallback from answer_extractor.py
- Suppresses dead-end responses (F-202): empty/near-empty → "suppressed"
"""
import logging
import time
from pathlib import Path
from typing import Optional

from lib.answer_extractor import extract_answer, format_as_bullets
from lib.rag_generator import RAGAnswerGenerator
from lib.triggers.types import Trigger, TriggerType

from . import prompts
from .types import GenerationResult

logger = logging.getLogger(__name__)

# Context budget split: 30% conversation, 70% RAG
_CONVERSATION_RATIO = 0.30
_RAG_RATIO = 0.70

# Prompt config per trigger type
_PROMPT_CONFIG = {
    TriggerType.QUESTION: {
        "system": prompts.QUESTION_SYSTEM,
        "template": prompts.QUESTION_PROMPT,
        "max_tokens": prompts.QUESTION_MAX_TOKENS,
    },
    TriggerType.TOPIC_MATCH: {
        "system": prompts.TOPIC_SYSTEM,
        "template": prompts.TOPIC_PROMPT,
        "max_tokens": prompts.TOPIC_MAX_TOKENS,
    },
    TriggerType.FOLLOW_UP: {
        "system": prompts.FOLLOWUP_SYSTEM,
        "template": prompts.FOLLOWUP_PROMPT,
        "max_tokens": prompts.FOLLOWUP_MAX_TOKENS,
    },
    TriggerType.ALERT: {
        "system": prompts.ALERT_SYSTEM,
        "template": prompts.ALERT_PROMPT,
        "max_tokens": prompts.ALERT_MAX_TOKENS,
    },
}


class ModeAwareGenerator:
    """Trigger-routed generation with extraction fallback.

    For QUESTION triggers, runs the two-stage hybrid pipeline:
      1. Extract relevant sentences (grounding)
      2. Generate fluent answer with prompt template

    For other trigger types, generates directly from RAG context
    with type-specific prompts and token limits.

    Args:
        model_path: Path to the GGUF model file.
        max_context_chars: Total character budget for context (default 6000).
        min_extraction_confidence: Minimum extraction score to proceed (default 0.25).
        use_generation: Enable LLM generation (False = extraction-only mode).
    """

    def __init__(
        self,
        model_path: Path,
        max_context_chars: int = 6000,
        min_extraction_confidence: float = 0.25,
        use_generation: bool = True,
        min_answer_length: int = 10,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.min_extraction_confidence = min_extraction_confidence
        self.use_generation = use_generation
        self.min_answer_length = min_answer_length
        self._generator: Optional[RAGAnswerGenerator] = None

        if use_generation:
            if not model_path.exists():
                logger.warning("Model not found at %s, using extraction-only", model_path)
                self.use_generation = False
            else:
                self._generator = RAGAnswerGenerator(model_path)

    def process_trigger(
        self,
        trigger: Trigger,
        rag_context: str,
        conversation_context: str = "",
    ) -> GenerationResult:
        """Generate a response for a trigger using the appropriate prompt.

        Args:
            trigger: The trigger that fired.
            rag_context: Retrieved context from ColBERT/RAG.
            conversation_context: Recent transcript text.

        Returns:
            GenerationResult with answer, method, confidence, and timing.
        """
        start = time.time()

        # Budget context chars between conversation and RAG
        conv_budget = int(self.max_context_chars * _CONVERSATION_RATIO)
        rag_budget = int(self.max_context_chars * _RAG_RATIO)
        conv_text = conversation_context[-conv_budget:] if conversation_context else ""
        rag_text = rag_context[:rag_budget]

        # For questions, run extraction grounding first
        if trigger.type == TriggerType.QUESTION:
            result = self._process_question(trigger, rag_text, conv_text, start)
        else:
            result = self._process_other(trigger, rag_text, conv_text, start)

        # F-202: Suppress dead-end responses — silence is better than "I can't help"
        if not result.answer or len(result.answer.strip()) < self.min_answer_length:
            return GenerationResult(
                answer="",
                trigger_type=trigger.type,
                confidence=result.confidence,
                method="suppressed",
                latency_ms=_elapsed_ms(start),
            )

        return result

    def _process_question(
        self,
        trigger: Trigger,
        rag_text: str,
        conv_text: str,
        start: float,
    ) -> GenerationResult:
        """Two-stage pipeline for questions: extraction then generation.

        If extraction grounding succeeds, uses extracted text as context for
        a more focused generation. If extraction confidence is too low, falls
        through to direct generation against full RAG context (same approach
        as topic triggers) rather than silently failing.
        """
        # Stage 1: Try extraction grounding
        extracted, extraction_confidence = extract_answer(
            rag_text, trigger.text, max_sentences=3,
        )

        context_for_generation = rag_text
        method = "generation"

        if extraction_confidence >= self.min_extraction_confidence and extracted:
            # Extraction succeeded — use extracted text as tighter context
            context_for_generation = extracted
            method = "hybrid"
            logger.debug(
                "Question extraction succeeded: conf=%.2f, %d chars",
                extraction_confidence, len(extracted),
            )
        else:
            logger.debug(
                "Question extraction low confidence (%.2f), using direct generation",
                extraction_confidence,
            )

        # Stage 2: Generate with prompt template
        if self.use_generation and self._generator:
            answer = self._generate(trigger, context_for_generation, conv_text)
            if answer:
                return GenerationResult(
                    answer=answer,
                    trigger_type=TriggerType.QUESTION,
                    confidence=max(extraction_confidence, trigger.confidence),
                    method=method,
                    latency_ms=_elapsed_ms(start),
                    source="ColBERT + " + method,
                )

        # Fallback: extraction bullets (only if extraction had results)
        if extracted:
            return GenerationResult(
                answer=format_as_bullets(extracted),
                trigger_type=TriggerType.QUESTION,
                confidence=extraction_confidence,
                method="extraction",
                latency_ms=_elapsed_ms(start),
                source="ColBERT + extraction",
            )

        return GenerationResult(
            answer="",
            trigger_type=TriggerType.QUESTION,
            confidence=extraction_confidence,
            method="no_match",
            latency_ms=_elapsed_ms(start),
        )

    def _process_other(
        self,
        trigger: Trigger,
        rag_text: str,
        conv_text: str,
        start: float,
    ) -> GenerationResult:
        """Direct generation for non-question triggers (topic, follow-up, alert)."""
        if not rag_text:
            return GenerationResult(
                answer="",
                trigger_type=trigger.type,
                confidence=trigger.confidence,
                method="no_context",
                latency_ms=_elapsed_ms(start),
            )

        if self.use_generation and self._generator:
            answer = self._generate(trigger, rag_text, conv_text)
            if answer:
                return GenerationResult(
                    answer=answer,
                    trigger_type=trigger.type,
                    confidence=trigger.confidence,
                    method="generation",
                    latency_ms=_elapsed_ms(start),
                    source="ColBERT + generation",
                )

        # Fallback: extract key sentences
        extracted, conf = extract_answer(rag_text, trigger.text, max_sentences=2)
        return GenerationResult(
            answer=format_as_bullets(extracted) if extracted else "",
            trigger_type=trigger.type,
            confidence=conf,
            method="extraction",
            latency_ms=_elapsed_ms(start),
            source="ColBERT + extraction",
        )

    def _generate(self, trigger: Trigger, context: str, conversation: str) -> Optional[str]:
        """Run LLM generation with trigger-specific prompt template."""
        if not self._generator:
            return None

        config = _PROMPT_CONFIG[trigger.type]
        prompt = config["template"].format(
            system=config["system"],
            conversation=conversation or "(no recent conversation)",
            context=context,
            text=trigger.text,
        )

        answer = self._generator.generate_text(
            prompt,
            max_tokens=config["max_tokens"],
            stop=prompts.STOP_TOKENS,
        )

        if not answer or answer.startswith("["):
            return None

        return self._generator._clean_answer(answer)


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start."""
    return (time.time() - start) * 1000.0
