"""
Ranker protocol.

Defines the interface for re-ranking retrieval results.
Implementations: HeuristicRanker (v0), future ModelRanker (v2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lib.rag.config import RAGConfig
from lib.rag.types import FusedHit


@runtime_checkable
class Ranker(Protocol):
    """Protocol for re-rankers. Implement to add new ranking strategies."""

    def rank(
        self, query: str, hits: list[FusedHit], config: RAGConfig
    ) -> list[FusedHit]:
        """Re-rank fused hits. Returns hits in new order with adjusted scores."""
        ...
