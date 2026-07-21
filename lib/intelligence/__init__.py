"""Encoder intelligence layer (F-501).

Public surface:
    TurnState                — typed per-turn state carrier
    Head                     — turn-evaluated intelligence head interface
    HeuristicHead            — adapter wrapping legacy rule-based triggers
    EncoderBackbone          — warm LFM2.5-Encoder-350M (mean-pooled)
    EncoderIntelligenceLayer — runs heads over a turn, priority-sorted
"""

from __future__ import annotations

from lib.intelligence.encoder import EncoderBackbone
from lib.intelligence.encoder_layer import EncoderIntelligenceLayer
from lib.intelligence.heads.base import Head
from lib.intelligence.heads.heuristic_heads import HeuristicHead
from lib.intelligence.turn_state import TurnState

__all__ = [
    "TurnState",
    "Head",
    "HeuristicHead",
    "EncoderBackbone",
    "EncoderIntelligenceLayer",
]
