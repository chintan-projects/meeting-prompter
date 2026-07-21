"""Speaker-attribution hierarchy (F-601).

Public surface:
    AttributionResolver — composes L1 channel / L3 acoustic / L4 roster + regime
    AttributionResult   — a resolved label with provenance + confidence
    AttributionLayer    — which layer produced a label
    Regime              — solo-endpoint vs conference-room
"""

from __future__ import annotations

from lib.attribution.resolver import AttributionResolver
from lib.attribution.types import (
    AttributionLayer,
    AttributionResult,
    Regime,
)

__all__ = [
    "AttributionResolver",
    "AttributionResult",
    "AttributionLayer",
    "Regime",
]
