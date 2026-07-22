"""
Embedder protocol (dependency injection point).

Defines the interface for embedding text into vectors. The RAG pipeline
never imports a specific embedding implementation — it receives one
via this protocol. Consumers swap in mock, sentence-transformers, or
OpenAI embeddings without touching the library.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a float vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: loop over embed()."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    # Optional: embedders may add ``embed_query(text) -> list[float]`` for
    # asymmetric query/passage prompts. Retrieval falls back to ``embed`` when
    # it is absent, so implementing it is not required by this protocol.
