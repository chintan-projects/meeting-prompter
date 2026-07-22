"""Unit tests for lib.corpus.active (F-704) — live-corpus activation state."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.corpus import active


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    f = tmp_path / "corpus_active.json"
    monkeypatch.setattr(active, "ACTIVE_FILE", f)
    return f


def test_set_and_get_roundtrip(state: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "distilled"
    corpus.mkdir()
    active.set_active_dir(corpus, state)
    assert active.get_active_dir(state) == corpus


def test_set_none_deactivates(state: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "distilled"
    corpus.mkdir()
    active.set_active_dir(corpus, state)
    active.set_active_dir(None, state)
    assert active.get_active_dir(state) is None


def test_set_missing_dir_rejected(state: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus dir not found"):
        active.set_active_dir(tmp_path / "nope", state)


def test_get_ignores_corrupt_state(state: Path) -> None:
    state.write_text("{not json", encoding="utf-8")
    assert active.get_active_dir(state) is None


def test_get_ignores_vanished_dir(state: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "distilled"
    corpus.mkdir()
    active.set_active_dir(corpus, state)
    corpus.rmdir()
    assert active.get_active_dir(state) is None


def test_resolve_corpus_prefers_active(state: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "distilled"
    corpus.mkdir()
    active.set_active_dir(corpus, state)
    docs, db = active.resolve_corpus("context", "data/rag.db")
    assert docs == corpus and db == active.ACTIVE_DB


def test_resolve_corpus_falls_back_to_configured(state: Path, tmp_path: Path) -> None:
    configured = tmp_path / "context"
    configured.mkdir()
    docs, db = active.resolve_corpus(str(configured), "data/rag.db")
    assert docs == configured and db == Path("data/rag.db")
