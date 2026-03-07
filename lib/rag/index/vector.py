"""
Vector cosine search.

Brute-force cosine similarity search across all chunk embeddings.
Centralises the search logic that was duplicated across three tools.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from typing import Sequence

from lib.rag.types import VectorHit


def vector_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int,
    filter_path: str | None = None,
) -> list[VectorHit]:
    """Search chunks by cosine similarity to the query embedding."""
    if not query_embedding:
        return []

    dim = len(query_embedding)
    pack_fmt = f"<{dim}f"

    if filter_path:
        rows = conn.execute(
            """
            SELECT c.id, c.embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL AND d.path LIKE ? || '%'
            """,
            (filter_path,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT c.id, c.embedding
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            """
        ).fetchall()

    hits: list[VectorHit] = []
    for row in rows:
        chunk_emb = list(struct.unpack(pack_fmt, row[1]))
        score = _cosine_similarity(query_embedding, chunk_emb)
        hits.append(VectorHit(chunk_id=row[0], score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
