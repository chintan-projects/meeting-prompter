"""Regression test: concurrent embedder load (F-705 pre-warm race)."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import pytest

from lib.rag.embedder import SentenceTransformerEmbedder


class _SlowModel:
    """Stand-in for SentenceTransformer with a construction window wide enough
    for a second thread to race into it."""

    constructions = 0
    lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        with _SlowModel.lock:
            _SlowModel.constructions += 1
        time.sleep(0.05)  # the window the real torch meta-tensor load leaves open

    def encode(self, text: str, **kwargs: Any) -> Any:
        import numpy as np

        return np.zeros(8)


def test_concurrent_load_constructs_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session start pre-warms the embedder on a background thread (F-705) while
    the pipeline queries on another. Before the lock, both entered _load_model and
    torch failed with 'Cannot copy out of meta tensor; no data!'."""
    _SlowModel.constructions = 0
    fake = type(sys)("sentence_transformers")
    fake.SentenceTransformer = _SlowModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    emb = SentenceTransformerEmbedder("stub-model", dimension=8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            emb.embed_query("what is the deployment timeline?")
        except BaseException as e:  # noqa: BLE001 — the race surfaced as NotImplementedError
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent load raised: {errors[:2]}"
    assert _SlowModel.constructions == 1, (
        f"model constructed {_SlowModel.constructions}x — the load is racing again"
    )
