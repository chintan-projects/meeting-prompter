"""Trigger type definitions for multi-mode meeting intelligence."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class TriggerType(Enum):
    """Types of events that trigger RAG lookup and generation."""

    QUESTION = "question"        # Direct question detected in speech
    TOPIC_MATCH = "topic"        # Discussion topic matches document content
    FOLLOW_UP = "follow_up"      # Natural pause after topic — suggest follow-up
    ALERT = "alert"              # Watch word or key term detected

    @property
    def priority(self) -> int:
        """Priority ordering: lower = higher priority."""
        return {
            TriggerType.ALERT: 1,
            TriggerType.QUESTION: 2,
            TriggerType.TOPIC_MATCH: 3,
            TriggerType.FOLLOW_UP: 4,
        }[self]

    @property
    def label(self) -> str:
        """Human-readable label for display (coaching-oriented)."""
        return {
            TriggerType.ALERT: "HEADS UP",
            TriggerType.QUESTION: "ANSWER",
            TriggerType.TOPIC_MATCH: "FYI",
            TriggerType.FOLLOW_UP: "SUGGEST",
        }[self]

    @property
    def emoji(self) -> str:
        """Emoji prefix for display."""
        return {
            TriggerType.ALERT: "\u26a0\ufe0f",     # warning sign
            TriggerType.QUESTION: "\U0001f4a1",     # light bulb
            TriggerType.TOPIC_MATCH: "\U0001f4cc",  # pushpin
            TriggerType.FOLLOW_UP: "\U0001f4ac",    # speech bubble
        }[self]

    @property
    def persistence(self) -> str:
        """Auto-dismiss tier: persistent (user dismisses), standard (90s), ephemeral (45s)."""
        return {
            TriggerType.ALERT: "persistent",
            TriggerType.QUESTION: "persistent",
            TriggerType.TOPIC_MATCH: "ephemeral",
            TriggerType.FOLLOW_UP: "standard",
        }[self]


@dataclass
class Trigger:
    """A detected event that should prompt RAG lookup and response generation.

    Attributes:
        type: What kind of trigger fired.
        text: The text that triggered the event.
        confidence: Confidence score 0.0-1.0.
        source_context: Recent conversation context at time of trigger.
        metadata: Type-specific data (e.g. matched watch_word, topic).
        timestamp: When the trigger fired (time.time()).
    """

    type: TriggerType
    text: str
    confidence: float
    source_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def priority(self) -> int:
        """Priority derived from trigger type."""
        return self.type.priority
