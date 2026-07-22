"""Sentence-transformers embedder for the RAG pipeline.

Default model is the Liquid retriever ``LFM2.5-Embedding-350M`` (1024-dim),
loaded from the local model registry via ``trust_remote_code``. The legacy
``all-MiniLM-L6-v2`` (384-dim, HuggingFace hub) remains selectable by name for
comparison and offline fallback. Lazy-loads the model on first ``embed()`` call
to keep import time low; thread-safe via the model's internal batch handling.

Model selection is config-driven — no hardcoded model in library call sites.
A bare local directory name (e.g. ``LFM2.5-Embedding-350M``) is resolved against
the model registry (``MODELS_DIR`` / ``~/Projects/_models``); anything else is
treated as a HuggingFace hub id.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default retriever (F-502 swap): Liquid embedding model, local + trust_remote_code.
_DEFAULT_MODEL = "LFM2.5-Embedding-350M"
_LIQUID_DIMENSION = 1024
_MINILM_DIMENSION = 384


def _resolve_model(model_name: str) -> tuple[str, bool]:
    """Resolve a model name to (load_target, is_local_registry_model).

    A path that already exists, or a bare name found under the model registry,
    resolves to a local directory (``is_local=True`` → needs trust_remote_code).
    Otherwise the name is passed through as a HuggingFace hub id.
    """
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate), True

    # Only treat bare names (no path separator) as registry lookups.
    if "/" not in model_name and "\\" not in model_name:
        from lib.paths import get_models_dir

        registry = get_models_dir() / model_name
        if registry.exists():
            return str(registry), True

    return model_name, False


class SentenceTransformerEmbedder:
    """Embedder implementation using sentence-transformers.

    Satisfies the lib.rag.index.protocol.Embedder protocol:
    - embed(text: str) -> list[float]
    - embed_batch(texts: list[str]) -> list[list[float]]
    - dimension property -> int
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dimension: Optional[int] = None,
        trust_remote_code: Optional[bool] = None,
        query_prompt: str = "",
        document_prompt: str = "",
    ) -> None:
        load_target, is_local = _resolve_model(model_name)
        self._model_name = load_target
        self._trust_remote_code = is_local if trust_remote_code is None else trust_remote_code
        self._dimension = (
            dimension
            if dimension is not None
            else (_LIQUID_DIMENSION if is_local else _MINILM_DIMENSION)
        )
        # Asymmetric query/passage prompts (E5-style). Empty = symmetric encoding.
        # LFM2.5-Embedding documents "query: " / "document: " in its ST config.
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        self._model: Optional[object] = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model (thread-safe).

        Two threads racing here corrupts the load — torch materialises weights
        from meta device during construction, and a concurrent second
        construction fails with "Cannot copy out of meta tensor; no data!".
        The race went live when session start began pre-warming the embedder on
        a background thread (F-705) while the pipeline queried on another.

        Double-checked locking: the unlocked fast path keeps steady-state
        embedding lock-free, and the model is published to ``self._model`` only
        once fully constructed, so the fast path can never see a half-built one.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model: %s (dim=%d, trust_remote_code=%s)",
                self._model_name,
                self._dimension,
                self._trust_remote_code,
            )
            model = SentenceTransformer(self._model_name, trust_remote_code=self._trust_remote_code)
            self._model = model
            logger.info("Embedding model ready (dim=%d)", self._dimension)

    def embed(self, text: str) -> list[float]:
        """Embed a single passage/document string into a float vector."""
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(  # type: ignore[union-attr]
            self._document_prompt + text, convert_to_numpy=True
        )
        return embedding.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple passages/documents in a single batch."""
        if not texts:
            return []
        self._load_model()
        assert self._model is not None
        prefixed = [self._document_prompt + t for t in texts]
        embeddings = self._model.encode(  # type: ignore[union-attr]
            prefixed,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [e.tolist() for e in embeddings]  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (applies the query prompt for asymmetric models).

        Retrieval calls this instead of ``embed`` so queries and passages can use
        distinct instruction prompts. With empty prompts this equals ``embed``.
        """
        self._load_model()
        assert self._model is not None
        embedding = self._model.encode(  # type: ignore[union-attr]
            self._query_prompt + text, convert_to_numpy=True
        )
        return embedding.tolist()  # type: ignore[union-attr]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (1024 for LFM2.5-Embedding-350M)."""
        return self._dimension
