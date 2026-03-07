"""
RAG pipeline error hierarchy.

Typed exceptions for each pipeline stage. Every error includes context
about what operation failed and why. Follows the catch-log-recover pattern.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base error for all RAG pipeline operations."""


class ParseError(RAGError):
    """Document parsing failed (unreadable file, unsupported format)."""


class ChunkError(RAGError):
    """Chunking operation failed (empty document, invalid config)."""


class IndexError_(RAGError):
    """Indexing (FTS5 or vector) failed.

    Named IndexError_ to avoid shadowing the builtin IndexError.
    """


class RetrievalError(RAGError):
    """Retrieval operation failed (query error, schema mismatch)."""
