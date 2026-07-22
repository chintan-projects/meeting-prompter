"""Route tests for the Prepare-corpus flow (F-704)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.routes import corpus as corpus_route


@pytest.fixture()
def distilled_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "distilled"
    d.mkdir()
    monkeypatch.setattr(corpus_route, "DISTILLED_DIR", d)
    return d


def test_distilled_state_empty_when_no_corpus(distilled_dir: Path) -> None:
    assert corpus_route._distilled_state()["exists"] is False


def test_distilled_state_reads_wizard_manifest(distilled_dir: Path) -> None:
    (distilled_dir / "a.distilled.md").write_text("x", encoding="utf-8")
    (distilled_dir / corpus_route.MANIFEST_NAME).write_text(
        json.dumps({"backend": "local", "mode": "consolidated",
                    "docs": {"a.md": {"units": 12}}}), encoding="utf-8")
    s = corpus_route._distilled_state()
    assert (s["exists"], s["backend"], s["units"], s["source"]) == (True, "local", 12, "manifest")


def test_distilled_state_falls_back_to_cli_sidecar(distilled_dir: Path) -> None:
    """The lab CLI writes a per-file .meta.json, not the wizard's manifest.
    Reading only the manifest made the wizard report 'no distilled corpus' while
    one sat on disk, blocking its own distill/readiness/activate steps."""
    f = distilled_dir / "a.distilled.md"
    f.write_text("x", encoding="utf-8")
    (distilled_dir / f".{f.name}.meta.json").write_text(
        json.dumps({"backend": "cloud", "units": 88}), encoding="utf-8")
    s = corpus_route._distilled_state()
    assert (s["exists"], s["backend"], s["units"], s["source"]) == (True, "cloud", 88, "sidecar")


def test_distilled_state_detects_corpus_with_no_provenance(distilled_dir: Path) -> None:
    (distilled_dir / "a.distilled.md").write_text("x", encoding="utf-8")
    s = corpus_route._distilled_state()
    assert s["exists"] is True and s["backend"] == "unknown" and s["source"] == "files"
