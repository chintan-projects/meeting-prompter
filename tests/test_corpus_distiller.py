"""Unit tests for lib.corpus — text cleaning + the heuristic distiller (F-701).

Covers the deterministic, no-network pieces. The cloud backend is offline/opt-in
(ADR-001) and is exercised via the lab, not here; its guard rails (credential
fail-fast, unknown backend) are tested without any network call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.corpus import distiller
from lib.corpus.text import clean_markdown


# --- clean_markdown --------------------------------------------------------
def test_clean_markdown_strips_headers_emphasis_and_inline_code() -> None:
    assert clean_markdown("## Heading\n**bold** and `code` here.") == "Heading bold and code here."


def test_clean_markdown_drops_code_fences_and_tables() -> None:
    src = "Intro line.\n```\nprint('x')\n```\n| a | b |\n|---|---|\n| 1 | 2 |\nOutro line."
    out = clean_markdown(src)
    assert "print" not in out and "|" not in out
    assert "Intro line." in out and "Outro line." in out


def test_clean_markdown_rewrites_links_to_text() -> None:
    assert clean_markdown("see [the docs](https://x.example/y)") == "see the docs"


# --- heuristic distiller ---------------------------------------------------
def test_distill_heuristic_skips_thin_sections() -> None:
    assert distiller._distill_heuristic("Part 1", "too short") == []


def test_distill_consolidated_keeps_whole_section_atomic_truncates() -> None:
    # A long section: atomic caps at MAX_UNIT_WORDS; consolidated keeps it all.
    text = "Fact one is important. " * 40  # ~160 words
    atomic = distiller._distill_heuristic("Topic", text, mode="atomic")[0]
    consolidated = distiller._distill_heuristic("Topic", text, mode="consolidated")[0]
    assert len(consolidated.split()) > len(atomic.split())
    assert len(atomic.split()) <= distiller.MAX_UNIT_WORDS + 20  # cap (plus topic prefix)


def test_distill_heuristic_produces_self_contained_unit() -> None:
    text = (
        "Speculative decoding is provably lossless. The rejection rule makes the "
        "output distribution identical to the target model's, only faster."
    )
    units = distiller._distill_heuristic("5.5 Provably Lossless", text)
    assert len(units) == 1
    assert "lossless" in units[0].lower()


def test_distill_emits_topic_unit_for_multisection_part(tmp_path: Path) -> None:
    # A Part with two sub-sections whose answers are split → a topic unit merges them.
    src = tmp_path / "doc.md"
    src.write_text(
        "# Doc\n\n# Part 1 — Quantization\n\n"
        "## 1.3 INT4 cost\n\nINT4 quantization costs about one to three percent of "
        "model quality for a four times smaller footprint.\n\n"
        "## 1.9 Where it degrades\n\nQuantization degrades most on multi-step math "
        "and reasoning tasks, far less on factual recall.\n",
        encoding="utf-8",
    )
    out = tmp_path / "doc.distilled.md"
    stats = distiller.distill(src, out, backend="heuristic", mode="consolidated")
    assert stats["topic_units"] >= 1
    body = out.read_text(encoding="utf-8")
    # the topic unit for Part 1 should carry BOTH the cost and the degrade content
    topic = [b for b in body.split("## ") if b.startswith("Topic — Part 1")][0]
    assert "percent" in topic and "degrades" in topic


def test_distill_writes_markdown_with_provenance(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text(
        "# Doc\n\n## Section A\n\nThis section explains that quantization trades "
        "precision for a smaller memory footprint on device.\n",
        encoding="utf-8",
    )
    out = tmp_path / "doc.distilled.md"
    stats = distiller.distill(src, out, backend="heuristic")
    assert stats["units"] >= 1
    body = out.read_text(encoding="utf-8")
    assert "_Source: doc.md ›" in body  # provenance pointer present
    assert "## Section A" in body


# --- backend guard rails ---------------------------------------------------
def test_distill_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown distiller backend"):
        distiller.distill(tmp_path / "x.md", tmp_path / "y.md", backend="quantum")


def test_distill_cloud_fails_fast_without_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No credential → distill() must abort before parsing/making any API call.
    monkeypatch.setattr(distiller.cloud, "credential_hint", lambda: "no key")
    with pytest.raises(RuntimeError, match="cloud backend needs a credential"):
        distiller.distill(tmp_path / "x.md", tmp_path / "y.md", backend="cloud")


def test_lab_wrapper_reexports_distiller() -> None:
    # scripts/lab/distiller.py is a thin wrapper — same objects, not copies.
    from scripts.lab import distiller as lab_distiller

    assert lab_distiller.distill is distiller.distill
    assert lab_distiller.MAX_UNIT_WORDS == distiller.MAX_UNIT_WORDS
