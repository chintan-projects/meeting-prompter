"""
Chunker protocol.

Defines the interface for splitting sections into chunks.
Implementations: TokenChunker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lib.rag.config import RAGConfig
from lib.rag.types import ChunkOutput, ParsedSection


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunkers. Implement to add new chunking strategies."""

    def chunk(
        self, sections: list[ParsedSection], config: RAGConfig
    ) -> list[ChunkOutput]:
        """Split sections into chunks respecting token limits."""
        ...
