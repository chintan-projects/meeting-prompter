"""Unit tests for lib.corpus.incremental (F-706) — change-driven re-distill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.corpus.incremental import MANIFEST_NAME, distill_dir

DOC_A = (
    "# A\n\n## Section One\n\nQuantization trades numeric precision for a smaller "
    "memory footprint on device, which is the core on-device lever.\n"
)
DOC_B = (
    "# B\n\n## Section Two\n\nSpeculative decoding is provably lossless because the "
    "rejection rule preserves the target model's output distribution exactly.\n"
)


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "a.md").write_text(DOC_A, encoding="utf-8")
    (src / "b.md").write_text(DOC_B, encoding="utf-8")
    return src, out


def test_first_run_distills_everything(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    res = distill_dir(src, out, backend="heuristic")
    assert res["distilled"] == ["a.md", "b.md"] and res["skipped"] == []
    assert (out / "a.distilled.md").exists() and (out / "b.distilled.md").exists()
    assert (out / MANIFEST_NAME).exists()


def test_unchanged_docs_are_skipped(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic")
    res = distill_dir(src, out, backend="heuristic")
    assert res["distilled"] == [] and res["skipped"] == ["a.md", "b.md"]


def test_only_changed_doc_redistills(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic")
    (src / "a.md").write_text(DOC_A + "\nAn extra fact about INT4 quantization appears here.\n")
    res = distill_dir(src, out, backend="heuristic")
    assert res["distilled"] == ["a.md"] and res["skipped"] == ["b.md"]
    assert "extra fact" in (out / "a.distilled.md").read_text()


def test_deleted_source_removes_output(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic")
    (src / "b.md").unlink()
    res = distill_dir(src, out, backend="heuristic")
    assert res["removed"] == ["b.distilled.md"]
    assert not (out / "b.distilled.md").exists()
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert "b.md" not in manifest["docs"]


def test_backend_change_invalidates_manifest(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic", mode="consolidated")
    res = distill_dir(src, out, backend="heuristic", mode="atomic")
    assert res["distilled"] == ["a.md", "b.md"]  # recipe changed → full rebuild


def test_force_redistills_everything(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic")
    res = distill_dir(src, out, backend="heuristic", force=True)
    assert res["distilled"] == ["a.md", "b.md"]


def test_corrupt_manifest_falls_back_to_full(corpus: tuple[Path, Path]) -> None:
    src, out = corpus
    distill_dir(src, out, backend="heuristic")
    (out / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    res = distill_dir(src, out, backend="heuristic")
    assert res["distilled"] == ["a.md", "b.md"]


def test_missing_src_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source dir not found"):
        distill_dir(tmp_path / "nope", tmp_path / "out")
