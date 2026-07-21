"""Intelligence heads — heuristic today, encoder-backed later (F-501)."""

from __future__ import annotations

from lib.intelligence.heads.base import Head
from lib.intelligence.heads.heuristic_heads import HeuristicHead

__all__ = ["Head", "HeuristicHead"]
