"""Trigger engine orchestrator — runs all triggers and returns prioritized results.

Central coordination point for the four trigger types. Called by the
conversation buffer on each transcribed chunk.

The turn-evaluated triggers (alert / question / topic) run through the
``EncoderIntelligenceLayer`` as ``Head`` implementations (F-501). This engine is
a thin adapter that preserves the public API (``evaluate`` / ``on_pause`` /
``on_statement`` / ``set_watch_words``) and the exact prior behavior; the
follow-up trigger keeps its pause-driven lifecycle.
"""

import logging
from typing import List, Optional

from lib.config import TriggerConfig
from lib.intelligence.encoder import EncoderBackbone
from lib.intelligence.encoder_layer import EncoderIntelligenceLayer
from lib.intelligence.heads.heuristic_heads import HeuristicHead
from lib.intelligence.heads.linear_probe import LinearProbeHead
from lib.intelligence.turn_state import TurnState

from .alert_trigger import AlertTrigger
from .followup_trigger import FollowUpTrigger
from .question_trigger import QuestionTrigger
from .topic_trigger import TopicTrigger
from .types import RAGQueryable, Trigger, TriggerType

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        config: TriggerConfig,
        rag_engine: RAGQueryable,
        encoder: Optional[EncoderBackbone] = None,
    ) -> None:
        self.question = QuestionTrigger(config)
        self.alert = AlertTrigger(
            watch_words=config.watch_words,
            cooldown_seconds=60.0,
        )
        self.topic = TopicTrigger(config, rag_engine)
        self.followup = FollowUpTrigger(config, rag_engine)

        # Turn-evaluated triggers become heads behind the intelligence layer.
        # Order preserved: ALERT (1) > QUESTION (2) > TOPIC_MATCH (3).
        # alert + question are HOT (cheap regex); topic is COLD (RAG-backed) and
        # only runs on the cold path for substantive turns (F-506).
        heads: List[object] = [
            HeuristicHead("alert", TriggerType.ALERT, self.alert),
            HeuristicHead("question", TriggerType.QUESTION, self.question),
            HeuristicHead("topic", TriggerType.TOPIC_MATCH, self.topic, cold=True),
        ]

        # F-510: frozen-encoder linear-probe router, wired-but-OFF. It stays
        # disabled (returns None, never loads the encoder) because on the current
        # synthetic set it regresses the question class the heuristic owns — see
        # evaluate_probe_gate. Enabling it is a future step (real labeled data)
        # that also removes the heuristic heads (delete-as-you-replace).
        self.probe_head: Optional[LinearProbeHead] = None
        if encoder is not None:
            self.probe_head = LinearProbeHead(encoder, enabled=False)
            heads.append(self.probe_head)

        self.layer = EncoderIntelligenceLayer(
            heads,
            encoder=encoder,
            cold_path_min_words=getattr(config, "cold_path_min_words", 3),
        )

    def set_watch_words(self, words: List[str]) -> None:
        """Update alert trigger watch words (e.g. from meeting context)."""
        self.alert.set_watch_words(words)

    def evaluate(self, text: str, conversation_context: str = "") -> List[Trigger]:
        """Run all turn-evaluated triggers, return results sorted by priority.

        Order: ALERT (1) > QUESTION (2) > TOPIC_MATCH (3).
        Follow-up triggers are checked separately via on_pause().
        """
        state = TurnState(text=text, conversation_context=conversation_context)
        return self.layer.process(state)

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
