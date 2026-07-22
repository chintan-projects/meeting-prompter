"""Unit tests for the Corpus & Retrieval Lab's pure logic.

Covers the deterministic, no-network pieces: markdown cleaning, coverage /
calibration aggregation, and the heuristic distiller. The cloud judge and RAG
retrieval paths are integration concerns and are exercised live in the lab, not here.
"""

from __future__ import annotations

from pathlib import Path

from scripts.lab import distiller
from scripts.lab.pipeline import aggregate_calibration, aggregate_coverage, clean_markdown


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


# --- aggregate_coverage ----------------------------------------------------
def _rec(span: str, chunk_id: int, rating: str, source: str = "human") -> dict:
    return {"span": span, "chunk_id": chunk_id, "rating": rating, "doc": "d.md", "source": source}


def test_coverage_counts_good_partial_gap_by_best_per_question() -> None:
    records = [
        _rec("q1", 1, "wrong"),
        _rec("q1", 2, "good"),  # best for q1 → good
        _rec("q2", 1, "partial"),
        _rec("q3", 1, "noise"),  # best for q3 → gap
    ]
    cov = aggregate_coverage(records, "human")
    assert cov["questions"] == 3
    assert (cov["good"], cov["partial"], cov["gap"]) == (1, 1, 1)


def test_coverage_filters_by_source() -> None:
    records = [_rec("q1", 1, "good", "human"), _rec("q1", 1, "noise", "judge")]
    assert aggregate_coverage(records, "human")["good"] == 1
    assert aggregate_coverage(records, "judge")["good"] == 0


def test_coverage_latest_write_wins_on_tie() -> None:
    # same span+chunk re-rated: the later record should win
    records = [_rec("q1", 1, "good"), _rec("q1", 1, "wrong")]
    cov = aggregate_coverage(records, "human")
    assert cov["good"] == 0 and cov["gap"] == 1


def test_coverage_empty_is_zero() -> None:
    cov = aggregate_coverage([], "human")
    assert cov["questions"] == 0 and cov["rows"] == []


# --- aggregate_calibration -------------------------------------------------
def test_calibration_agreement_only_over_pairs_rated_by_both() -> None:
    records = [
        _rec("q1", 1, "good", "human"),
        _rec("q1", 1, "good", "judge"),  # match
        _rec("q1", 2, "partial", "human"),
        _rec("q1", 2, "wrong", "judge"),  # mismatch
        _rec("q2", 1, "good", "human"),  # judge never rated → excluded
    ]
    cal = aggregate_calibration(records)
    assert cal["pairs"] == 2
    assert cal["agree"] == 1
    assert cal["agreement_pct"] == 50


def test_calibration_empty_is_zero() -> None:
    cal = aggregate_calibration([])
    assert cal["pairs"] == 0 and cal["agreement_pct"] == 0


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
