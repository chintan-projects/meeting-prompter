"""
FTS5 lexical search.

Uses SQLite's built-in FTS5 extension with BM25 ranking and configurable
field boosts. The FTS5 index is kept in sync by triggers defined in
storage/schema.py.
"""

from __future__ import annotations

import sqlite3

from lib.rag.config import RAGConfig
from lib.rag.types import FTSHit


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    config: RAGConfig,
    filter_path: str | None = None,
) -> list[FTSHit]:
    """
    Search the FTS5 index with BM25 ranking and field boosts.

    Field boost weights (bm25 columns): content, title, heading_path, keywords.
    Higher weight = more important for ranking.
    """
    if not query.strip():
        return []

    # Sanitise query for FTS5: escape double quotes, wrap terms
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []

    # BM25 returns negative scores (more negative = better match).
    # Column weights are passed as arguments to bm25().
    bm25_weights = (
        f"{config.boost_content}, {config.boost_title}, "
        f"{config.boost_heading}, {config.boost_keywords}"
    )

    if filter_path:
        sql = f"""
            SELECT rowid, bm25(chunks_fts, {bm25_weights}) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
              AND rowid IN (
                  SELECT c.id FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                  WHERE d.path LIKE ? || '%'
              )
            ORDER BY score
            LIMIT ?
        """
        rows = conn.execute(sql, (safe_query, filter_path, top_k)).fetchall()
    else:
        sql = f"""
            SELECT rowid, bm25(chunks_fts, {bm25_weights}) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """
        rows = conn.execute(sql, (safe_query, top_k)).fetchall()

    return [
        FTSHit(
            chunk_id=row[0],
            # Negate BM25 score so higher = better match
            score=-row[1] if row[1] is not None else 0.0,
        )
        for row in rows
    ]


def _sanitize_fts_query(query: str) -> str:
    """Sanitise user input for FTS5 MATCH syntax.

    Escapes special characters and wraps individual terms in double quotes.
    """
    # Remove FTS5 operators that could cause syntax errors
    cleaned = query.replace('"', "").replace("*", "").replace("(", "").replace(")", "")
    # Split into words and wrap each in quotes for exact matching
    terms = [f'"{t}"' for t in cleaned.split() if t.strip()]
    return " ".join(terms)
