"""
RAG pipeline configuration.

All tunables for the hybrid retrieval pipeline live here.
No magic numbers in library code — everything references RAGConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# Default file types for indexing
DEFAULT_FILE_TYPES: Final[list[str]] = [
    ".txt", ".md", ".py", ".ts", ".js", ".html", ".css", ".json",
]


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline. Immutable after creation."""

    # ─── Chunking ─────────────────────────────────────────────────────────
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # ─── Retrieval ────────────────────────────────────────────────────────
    lexical_weight: float = 0.05
    semantic_weight: float = 0.95
    lexical_top_k: int = 20
    semantic_top_k: int = 20

    # ─── FTS5 field boosts (bm25 column weights) ─────────────────────────
    boost_content: float = 1.0
    boost_title: float = 10.0
    boost_heading: float = 5.0
    boost_keywords: float = 20.0

    # ─── Ranking ──────────────────────────────────────────────────────────
    title_match_boost: float = 0.1
    keyword_match_boost: float = 0.15
    locality_boost: float = 0.05

    # ─── Indexing ─────────────────────────────────────────────────────────
    file_types: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_FILE_TYPES)
    )
