"""
Heuristic re-ranker (v0).

Applies lightweight boosts based on title match, keyword match,
and section locality. No ML model required.
"""

from __future__ import annotations

import sqlite3

from lib.rag.config import RAGConfig
from lib.rag.types import FusedHit


class HeuristicRanker:
    """Re-rank results using heuristic signal boosts."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def rank(
        self, query: str, hits: list[FusedHit], config: RAGConfig
    ) -> list[FusedHit]:
        """Apply heuristic boosts and re-sort."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        boosted: list[FusedHit] = []
        for hit in hits:
            boost = 0.0

            # Title match boost
            title = self._get_title(hit.chunk_id)
            if title and _terms_overlap(query_terms, title.lower()):
                boost += config.title_match_boost

            # Keyword match boost
            keywords = self._get_keywords(hit.chunk_id)
            if keywords and _terms_overlap(query_terms, keywords.lower()):
                boost += config.keyword_match_boost

            boosted.append(
                FusedHit(
                    chunk_id=hit.chunk_id,
                    fused_score=round(hit.fused_score + boost, 6),
                    lexical_score=hit.lexical_score,
                    semantic_score=hit.semantic_score,
                )
            )

        boosted.sort(key=lambda h: h.fused_score, reverse=True)
        return boosted

    def _get_title(self, chunk_id: int) -> str | None:
        """Get the document filename for a chunk."""
        row = self._conn.execute(
            """
            SELECT d.filename FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        return row[0] if row else None

    def _get_keywords(self, chunk_id: int) -> str | None:
        """Get manual keywords for a chunk."""
        row = self._conn.execute(
            "SELECT manual_keywords FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        return row[0] if row else None


def _terms_overlap(query_terms: set[str], text: str) -> bool:
    """Check if any query term appears in the text."""
    text_terms = set(text.split())
    return bool(query_terms & text_terms)
