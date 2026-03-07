"""
Weighted score fusion.

Combines lexical (FTS5) and semantic (vector) search results using
weighted sum. Lexical BM25 scores are min-max normalised (arbitrary
range). Semantic cosine similarity is used raw (already [0, 1] for
normalised vectors), preserving natural score discrimination.
"""

from __future__ import annotations

from lib.rag.types import FTSHit, FusedHit, VectorHit


def weighted_fusion(
    lexical_hits: list[FTSHit],
    semantic_hits: list[VectorHit],
    lexical_weight: float = 0.05,
    semantic_weight: float = 0.95,
    top_k: int = 10,
) -> list[FusedHit]:
    """Fuse lexical and semantic scores via weighted sum.

    Lexical (BM25) scores are min-max normalised to [0, 1] because BM25
    produces arbitrary-range scores. Semantic (cosine) scores are used
    raw — they are already in [0, 1] for normalised vectors, and
    min-max would destroy discrimination (top always becomes 1.0).
    """
    # Build score maps
    lex_scores = {h.chunk_id: h.score for h in lexical_hits}
    sem_scores = {h.chunk_id: h.score for h in semantic_hits}

    # Min-max normalise lexical (BM25 has arbitrary range)
    lex_norm = _min_max_normalise(lex_scores)
    # Semantic cosine is already [0, 1] — use raw scores
    sem_raw = sem_scores

    # Union of all chunk IDs
    all_ids = set(lex_norm.keys()) | set(sem_raw.keys())

    results: list[FusedHit] = []
    for chunk_id in all_ids:
        lex = lex_norm.get(chunk_id, 0.0)
        sem = sem_raw.get(chunk_id, 0.0)
        fused = lexical_weight * lex + semantic_weight * sem
        results.append(
            FusedHit(
                chunk_id=chunk_id,
                fused_score=round(fused, 6),
                lexical_score=round(lex, 6),
                semantic_score=round(sem, 6),
            )
        )

    results.sort(key=lambda h: h.fused_score, reverse=True)
    return results[:top_k]


def _min_max_normalise(scores: dict[int, float]) -> dict[int, float]:
    """Normalise scores to [0, 1] via min-max scaling."""
    if not scores:
        return {}

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val

    if span == 0.0:
        # All scores identical — normalise to 1.0 if non-zero, else 0.0
        norm_val = 1.0 if max_val > 0.0 else 0.0
        return {k: norm_val for k in scores}

    return {k: (v - min_val) / span for k, v in scores.items()}
