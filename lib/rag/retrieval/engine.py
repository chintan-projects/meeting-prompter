"""
Retrieval engine — orchestrates hybrid search.

Runs FTS5 lexical search and vector cosine search in parallel,
fuses results via weighted sum, builds citations, and returns
structured RetrievalResult objects.
"""

from __future__ import annotations

import sqlite3

from lib.rag.config import RAGConfig
from lib.rag.index.fts import fts_search
from lib.rag.index.protocol import Embedder
from lib.rag.index.vector import vector_search
from lib.rag.retrieval.fusion import weighted_fusion
from lib.rag.types import Citation, FusedHit, RetrievalResult


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder,
    config: RAGConfig,
    top_k: int = 5,
    filter_path: str | None = None,
) -> list[RetrievalResult]:
    """Full hybrid retrieval pipeline.

    1. FTS5 lexical search (field-boosted BM25)
    2. Vector cosine search (embedding similarity)
    3. Weighted fusion (default 5% lexical / 95% semantic)
    4. Build citations for each result
    """
    # Lexical search via FTS5
    lexical_hits = fts_search(
        conn, query, config.lexical_top_k, config, filter_path
    )

    # Semantic search via embeddings
    query_emb = embedder.embed(query)
    semantic_hits = vector_search(
        conn, query_emb, config.semantic_top_k, filter_path
    )

    # Fuse results
    fused = weighted_fusion(
        lexical_hits,
        semantic_hits,
        lexical_weight=config.lexical_weight,
        semantic_weight=config.semantic_weight,
        top_k=top_k,
    )

    # Build retrieval results with citations
    return _build_results(conn, fused)


def _build_results(
    conn: sqlite3.Connection, fused_hits: list[FusedHit]
) -> list[RetrievalResult]:
    """Enrich fused hits with chunk content and citations."""
    if not fused_hits:
        return []

    results: list[RetrievalResult] = []
    for hit in fused_hits:
        row = conn.execute(
            """
            SELECT c.content, c.chunk_index, c.section_id,
                   d.path, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (hit.chunk_id,),
        ).fetchone()

        if row is None:
            continue

        # Fetch section info if available
        section_heading = ""
        heading_path = ""
        page_range: tuple[int | None, int | None] = (None, None)

        if row[2] is not None:  # section_id
            sec_row = conn.execute(
                "SELECT heading, heading_path, start_page, end_page "
                "FROM sections WHERE id = ?",
                (row[2],),
            ).fetchone()
            if sec_row:
                section_heading = sec_row[0] or ""
                heading_path = sec_row[1] or ""
                page_range = (sec_row[2], sec_row[3])

        citation = Citation(
            document_path=row[3],
            document_name=row[4],
            section_heading=section_heading,
            heading_path=heading_path,
            page_range=page_range,
            chunk_id=hit.chunk_id,
            chunk_index=row[1],
        )

        results.append(
            RetrievalResult(
                chunk_id=hit.chunk_id,
                document_path=row[3],
                section_heading=section_heading,
                heading_path=heading_path,
                chunk_text=row[0],
                chunk_index=row[1],
                score=hit.fused_score,
                lexical_score=hit.lexical_score,
                semantic_score=hit.semantic_score,
                citation=citation,
            )
        )

    return results
