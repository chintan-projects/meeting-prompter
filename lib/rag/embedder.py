"""Sentence-transformers embedder for the RAG pipeline.

Uses all-MiniLM-L6-v2 (384 dimensions, ~80MB). Lazy-loads the model
on first embed() call to keep import time low. Thread-safe via the
model's internal batch handling.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384


class SentenceTransformerEmbedder:
    """Embedder implementation using sentence-transformers.

    Satisfies the lib.rag.index.protocol.Embedder protocol:
    - embed(text: str) -> list[float]
    - embed_batch(texts: list[str]) -> list[list[float]]
    - dimension property -> int
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model ready (dim=%d)", _DIMENSION)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a float vector."""
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True)  # type: ignore[union-attr]
        return embedding.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch."""
        if not texts:
            return []
        self._load_model()
        assert self._model is not None
        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts, convert_to_numpy=True, batch_size=32,
        )
        return [e.tolist() for e in embeddings]  # type: ignore[union-attr]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return _DIMENSION
