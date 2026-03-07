"""Follow-up trigger — suggests follow-up points during natural pauses.

Fires after a 3+ second pause following a substantive statement, when
RAG has related content the user might want to bring up.
"""
import logging
import time
from typing import Optional

from lib.config import TriggerConfig
from .types import RAGQueryable, Trigger, TriggerType

logger = logging.getLogger(__name__)


class FollowUpTrigger:
    """Suggests follow-up points when conversation pauses after a topic.

    Records non-question statements and checks for related RAG content
    when a pause occurs. Only fires if the docs have relevant information
    the user might want to raise.
    """

    def __init__(self, config: TriggerConfig, rag_engine: RAGQueryable) -> None:
        self.rag = rag_engine
        self.pause_threshold = config.followup_pause_threshold
        self.min_confidence = config.followup_rag_threshold
        self._last_statement: Optional[str] = None
        self._last_statement_time: float = 0.0
        self._last_fired_time: float = 0.0
        # Minimum gap between follow-up suggestions
        self._min_gap_seconds: float = 30.0

    def on_statement(self, text: str, timestamp: float) -> None:
        """Record a non-question statement for potential follow-up.

        Called by the conversation buffer when text is not a question
        but is substantive (not noise).
        """
        words = text.split()
        if len(words) >= 5:  # Only track substantive statements
            self._last_statement = text
            self._last_statement_time = timestamp

    def on_pause(self, timestamp: float) -> Optional[Trigger]:
        """Check if a pause warrants a follow-up suggestion.

        Called when silence is detected in the audio stream.
        """
        if not self._last_statement:
            return None

        pause_duration = timestamp - self._last_statement_time
        if pause_duration < self.pause_threshold:
            return None

        # Avoid rapid-fire follow-ups
        if timestamp - self._last_fired_time < self._min_gap_seconds:
            self._last_statement = None
            return None

        # Query RAG for related content
        try:
            _, confidence, source = self.rag.query(self._last_statement)
        except Exception:
            logger.debug("Follow-up RAG query failed")
            self._last_statement = None
            return None

        # Reset state regardless of whether we fire
        statement = self._last_statement
        self._last_statement = None

        if confidence >= self.min_confidence:
            self._last_fired_time = timestamp
            return Trigger(
                type=TriggerType.FOLLOW_UP,
                text=statement,
                confidence=confidence,
                metadata={"source": source},
                timestamp=timestamp,
            )

        return None

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]:
        """Not used for follow-up — fires via on_pause instead.

        Included for interface consistency with other triggers.
        """
        return None
