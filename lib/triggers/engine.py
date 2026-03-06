"""Trigger engine orchestrator — runs all triggers and returns prioritized results.

Central coordination point for the four trigger types. Called by the
conversation buffer on each transcribed chunk.
"""
import logging
from typing import List, Optional, Protocol

from lib.config import TriggerConfig
from .types import Trigger, TriggerType
from .question_trigger import QuestionTrigger
from .alert_trigger import AlertTrigger
from .topic_trigger import TopicTrigger
from .followup_trigger import FollowUpTrigger

logger = logging.getLogger(__name__)


class RAGQueryable(Protocol):
    """Protocol for RAG engines that support querying."""

    def query(self, text: str) -> tuple:
        ...


class TriggerEngine:
    """Evaluates all trigger types and returns results sorted by priority.

    Usage:
        engine = TriggerEngine(config, rag_engine)
        triggers = engine.evaluate("What is the deployment timeline?", context)
        # → [Trigger(type=QUESTION, ...)]

        # For follow-ups, call on_pause when silence detected
        trigger = engine.on_pause(timestamp)

        # For non-question statements, record for follow-up tracking
        engine.on_statement("We're targeting Q2 for the beta", timestamp)
    """

    def __init__(self, config: TriggerConfig, rag_engine: RAGQueryable) -> None:
        self.question = QuestionTrigger(config)
        self.alert = AlertTrigger(
            watch_words=config.watch_words,
            cooldown_seconds=60.0,
        )
        self.topic = TopicTrigger(config, rag_engine)
        self.followup = FollowUpTrigger(config, rag_engine)

    def set_watch_words(self, words: List[str]) -> None:
        """Update alert trigger watch words (e.g. from meeting context)."""
        self.alert.set_watch_words(words)

    def evaluate(self, text: str, conversation_context: str = "") -> List[Trigger]:
        """Run all triggers against text, return results sorted by priority.

        Order: ALERT (1) > QUESTION (2) > TOPIC_MATCH (3).
        Follow-up triggers are checked separately via on_pause().
        """
        triggers: List[Trigger] = []

        # Check each trigger type (alert first — highest priority)
        for evaluator in [self.alert, self.question, self.topic]:
            try:
                result = evaluator.evaluate(text, conversation_context)
                if result is not None:
                    triggers.append(result)
            except Exception as e:
                logger.debug("Trigger evaluation error: %s", e)

        triggers.sort(key=lambda t: t.priority)
        return triggers

    def on_pause(self, timestamp: float) -> Optional[Trigger]:
        """Check follow-up trigger on audio pause."""
        try:
            return self.followup.on_pause(timestamp)
        except Exception as e:
            logger.debug("Follow-up trigger error: %s", e)
            return None

    def on_statement(self, text: str, timestamp: float) -> None:
        """Record non-triggering statements for follow-up detection."""
        self.followup.on_statement(text, timestamp)
