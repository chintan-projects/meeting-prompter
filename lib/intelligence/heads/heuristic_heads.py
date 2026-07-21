"""Heuristic heads — the first Head implementations (F-501).

These adapt the existing rule-based triggers (question / alert / topic) to the
``Head`` interface with zero behavior change. They read only ``state.text`` and
``state.conversation_context`` — exactly the inputs the legacy engine passed —
so the encoder embedding stays unused on this path. Encoder-backed heads replace
these one at a time (delete-as-you-replace) once they beat the heuristic.
"""

from __future__ import annotations

from typing import Optional, Protocol

from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger, TriggerType


class _TextEvaluator(Protocol):
    """The legacy trigger surface these heads wrap."""

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]: ...


class HeuristicHead:
    """Wraps a legacy ``(text, context) -> Optional[Trigger]`` evaluator as a Head."""

    def __init__(self, name: str, trigger_type: TriggerType, evaluator: _TextEvaluator) -> None:
        self.name = name
        self.trigger_type = trigger_type
        self._evaluator = evaluator

    def evaluate(self, state: TurnState) -> Optional[Trigger]:
        return self._evaluator.evaluate(state.text, state.conversation_context)
