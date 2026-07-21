"""EncoderIntelligenceLayer — runs the heads over a turn (F-501).

The layer is the intelligence brain's coordinator: given a ``TurnState`` it
optionally computes the shared mean-pooled encoder embedding once, then runs each
``Head`` and returns the priority-sorted triggers. This is a pure structural
refactor of the old ``TriggerEngine.evaluate`` loop — same heads, same order,
same per-head error isolation, same priority sort — so behavior is unchanged.

Embedding computation is off by default (``compute_embedding=False``): the
heuristic heads don't need it, so the model never loads on that path. Encoder-
backed heads (F-510+) flip it on and read ``state.embedding``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from lib.intelligence.encoder import EncoderBackbone
from lib.intelligence.heads.base import Head
from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger

logger = logging.getLogger(__name__)


class EncoderIntelligenceLayer:
    """Coordinates the shared encoder backbone and the intelligence heads."""

    def __init__(
        self,
        heads: Sequence[Head],
        encoder: Optional[EncoderBackbone] = None,
        compute_embedding: bool = False,
    ) -> None:
        self._heads: List[Head] = list(heads)
        self._encoder = encoder
        self._compute_embedding = compute_embedding

    @property
    def heads(self) -> List[Head]:
        return self._heads

    def ensure_embedding(self, state: TurnState) -> None:
        """Populate ``state.embedding`` from the shared encoder (best-effort).

        One mean-pooled forward per turn. Failures degrade to ``None`` — heads
        fall back to their heuristic behavior rather than the turn erroring out.
        """
        if state.embedding is not None or self._encoder is None:
            return
        try:
            state.embedding = self._encoder.embed(state.text)
        except Exception as exc:  # model missing / load failure — degrade honestly
            logger.debug("Encoder embedding unavailable, degrading: %s", exc)

    def process(self, state: TurnState) -> List[Trigger]:
        """Run all heads over the turn; return priority-sorted triggers."""
        if self._compute_embedding:
            self.ensure_embedding(state)

        triggers: List[Trigger] = []
        for head in self._heads:
            try:
                result = head.evaluate(state)
                if result is not None:
                    triggers.append(result)
            except Exception as exc:
                logger.debug("Head '%s' evaluation error: %s", getattr(head, "name", "?"), exc)

        triggers.sort(key=lambda t: t.priority)
        state.triggers = triggers
        return triggers
