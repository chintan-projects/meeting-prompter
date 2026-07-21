"""Typed shared turn-state for the encoder intelligence layer (F-501).

``TurnState`` is the single object that flows through the intelligence layer for
one speech turn: the text, its conversation context, an optional mean-pooled
encoder embedding, and the accumulated head outputs. It replaces passing loose
``(text, context)`` tuples around and gives heads (heuristic today, encoder-
backed later) one typed surface to read from and write to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from lib.triggers.types import Trigger


@dataclass
class TurnState:
    """Mutable state for one turn as it passes through the intelligence layer.

    Attributes:
        text: The transcribed text for this turn.
        conversation_context: Recent rolling transcript window.
        timestamp: When the turn occurred (time.time()).
        embedding: Mean-pooled encoder vector, populated by the layer only when
            the encoder is wired and enabled; ``None`` on the pure-heuristic path.
        triggers: Head outputs collected for this turn (priority-sorted).
    """

    text: str
    conversation_context: str = ""
    timestamp: float = 0.0
    embedding: Optional[List[float]] = None
    triggers: List[Trigger] = field(default_factory=list)
    # Route-first (F-506): whether the expensive RAG-backed cold path ran.
    ran_cold: bool = False
