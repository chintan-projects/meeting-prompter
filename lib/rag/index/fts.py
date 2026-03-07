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

# Common English stop words to strip from FTS5 queries.
# Removing these improves recall (OR of content words only).
_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "you", "we", "they", "he", "she",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "so", "if", "than", "then", "there", "their", "our",
    "my", "your", "his", "her", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "all", "each",
    "some", "any", "very", "just", "also", "more", "most", "other",
])


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    config: RAGConfig,
    filter_path: str | None = None,
) -> list[FTSHit]:
    """Search the FTS5 index with BM25 ranking and field boosts.

    Field boost weights (bm25 columns): content, title, heading_path, keywords.
    Higher weight = more important for ranking.
    """
    if not query.strip():
        return []

    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []

    # BM25 returns negative scores (more negative = better match).
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

    Strips stop words for better recall, joins remaining content words
    with OR so chunks matching any term are returned. BM25 naturally
    ranks chunks with more matching terms higher.
    """
    # Remove FTS5 operators that could cause syntax errors
    cleaned = query.replace('"', "").replace("*", "").replace("(", "").replace(")", "")
    cleaned = cleaned.replace("?", "").replace("!", "").replace(".", "")

    terms: list[str] = []
    for word in cleaned.split():
        word_lower = word.lower().strip()
        if word_lower and word_lower not in _STOP_WORDS and len(word_lower) >= 2:
            terms.append(f'"{word_lower}"')

    if not terms:
        return ""

    return " OR ".join(terms)
