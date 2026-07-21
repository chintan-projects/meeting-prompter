"""The Head interface (F-501).

A ``Head`` is any component that reads a ``TurnState`` and optionally emits a
``Trigger``. The current heuristics (question / alert / topic detection) are the
first head implementations; encoder-backed heads (linear probe in F-510, forge
LoRA in F-503+) drop in behind the same interface with no engine changes.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger, TriggerType


@runtime_checkable
class Head(Protocol):
    """A turn-evaluated intelligence head."""

    #: Stable identifier for logs / routing (e.g. "question", "alert").
    name: str
    #: The trigger type this head produces.
    trigger_type: TriggerType

    def evaluate(self, state: TurnState) -> Optional[Trigger]:
        """Inspect the turn and return a Trigger, or None to stay silent."""
        ...
