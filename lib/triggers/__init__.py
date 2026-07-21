"""Trigger engine for multi-mode meeting intelligence.

``TriggerEngine`` is exposed lazily (PEP 562) so that importing lightweight
``lib.triggers.types`` from the intelligence layer does not pull in ``engine``
(which imports ``lib.intelligence``) — that would form an import cycle.
"""

from typing import TYPE_CHECKING

from .types import Trigger, TriggerType

if TYPE_CHECKING:
    from .engine import TriggerEngine

__all__ = ["Trigger", "TriggerType", "TriggerEngine"]


def __getattr__(name: str) -> object:
    if name == "TriggerEngine":
        from .engine import TriggerEngine

        return TriggerEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
