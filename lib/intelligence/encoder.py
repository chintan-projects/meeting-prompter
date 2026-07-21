"""Shared encoder backbone — LFM2.5-Encoder-350M (F-501).

One warm, frozen, bidirectional encoder that produces a mean-pooled sentence
vector per turn (~14 ms measured on MPS in the Stage-0 smoke spike). Loaded via
``AutoModelForMaskedLM`` → ``.lfm2`` backbone with ``trust_remote_code``; the
model stays resident after the first call. Mean-pooling (never last-token) is
required — the final encoder layer is a 3-token conv (Liquid architecture rule).

The backbone is optional: it lazy-loads on first ``embed()`` and is only invoked
when a head needs embeddings (F-510+). The heuristic path never loads it, so the
test suite runs without the model present.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # heavy deps stay out of import time
    import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIRNAME = "LFM2.5-Encoder-350M"
_DIMENSION = 1024
_MAX_TOKENS = 64


class EncoderBackbone:
    """Warm, frozen LFM2.5-Encoder-350M with mean-pooled embeddings."""

    def __init__(
        self,
        model_dirname: str = _MODEL_DIRNAME,
        device: Optional[str] = None,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self._model_dirname = model_dirname
        self._device = device
        self._max_tokens = max_tokens
        # Dynamically-typed HF objects (untyped third-party); Any is intentional.
        self._tok: Any = None
        self._backbone: Any = None
        self._lock = threading.Lock()

    def resolve_path(self) -> Path:
        """Locate the model directory in the local registry."""
        from lib.paths import get_models_dir

        return get_models_dir() / self._model_dirname

    def is_available(self) -> bool:
        """True if the encoder weights are present on disk."""
        return self.resolve_path().exists()

    @property
    def dimension(self) -> int:
        return _DIMENSION

    def _load(self) -> None:
        """Lazy-load tokenizer + backbone once; thread-safe and warm thereafter."""
        if self._backbone is not None:
            return
        with self._lock:
            if self._backbone is not None:
                return
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            path = self.resolve_path()
            if not path.exists():
                raise FileNotFoundError(f"Encoder model not found: {path}")

            device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            logger.info("Loading encoder backbone: %s (device=%s)", path.name, device)
            tok = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                str(path), trust_remote_code=True
            )
            backbone = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True).lfm2
            backbone.eval().to(device)
            self._tok = tok
            self._backbone = backbone
            self._device = device
            logger.info("Encoder backbone ready (dim=%d)", _DIMENSION)

    def embed(self, text: str) -> List[float]:
        """Mean-pooled embedding for a single string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Mean-pooled embeddings for a batch of strings (frozen, no grad)."""
        if not texts:
            return []
        self._load()
        import torch

        assert self._backbone is not None and self._tok is not None
        out: List[List[float]] = []
        with torch.no_grad():
            for t in texts:
                enc = self._tok(
                    t, return_tensors="pt", truncation=True, max_length=self._max_tokens
                ).to(self._device)
                hs = self._backbone(**enc).last_hidden_state  # [1, T, H]
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                vec: "np.ndarray" = pooled.squeeze(0).cpu().numpy()
                out.append(vec.tolist())
        return out
