"""
RAG pipeline shared types.

Dataclasses used across multiple modules. No business logic here —
just data containers with clear field documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─── Parser types ─────────────────────────────────────────────────────────────


@dataclass
class ParsedSection:
    """A section extracted from a document (e.g. a markdown heading block)."""

    heading: str
    heading_level: int  # 0 = no heading (root), 1-6 = h1-h6
    heading_path: str  # "Chapter > Section > Subsection"
    content: str
    start_page: int | None = None
    end_page: int | None = None


@dataclass
class ParsedDocument:
    """Result of parsing a single document."""

    path: str
    filename: str
    mime_type: str
    sections: list[ParsedSection]
    full_text: str
    token_count: int


# ─── Chunker types ────────────────────────────────────────────────────────────


@dataclass
class ChunkOutput:
    """A single chunk produced by the chunker."""

    content: str
    token_count: int
    section_index: int  # which ParsedSection this chunk belongs to
    parent_chunk_index: int | None = None  # for hierarchical chunking (Phase 2)


# ─── Index types ──────────────────────────────────────────────────────────────


@dataclass
class FTSHit:
    """A hit from FTS5 lexical search."""

    chunk_id: int
    score: float  # BM25 score (negative by convention in FTS5, normalised to positive)


@dataclass
class VectorHit:
    """A hit from vector cosine search."""

    chunk_id: int
    score: float  # cosine similarity in [0, 1] for normalised vectors


@dataclass
class FusedHit:
    """A hit after weighted fusion of lexical and semantic scores."""

    chunk_id: int
    fused_score: float
    lexical_score: float
    semantic_score: float


# ─── Citation types ───────────────────────────────────────────────────────────


@dataclass
class Citation:
    """Source anchor for a retrieval result."""

    document_path: str
    document_name: str
    section_heading: str
    heading_path: str
    page_range: tuple[int | None, int | None]
    chunk_id: int
    chunk_index: int


# ─── Retrieval types ─────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A single result from the hybrid retrieval pipeline."""

    chunk_id: int
    document_path: str
    section_heading: str
    heading_path: str
    chunk_text: str
    chunk_index: int
    score: float  # final fused score
    lexical_score: float
    semantic_score: float
    citation: Citation


# ─── Index result ─────────────────────────────────────────────────────────────


@dataclass
class IndexResult:
    """Summary of an indexing operation."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_updated: int = 0
    documents_removed: int = 0
    chunks_created: int = 0
    sections_created: int = 0
    errors: list[str] = field(default_factory=list)
