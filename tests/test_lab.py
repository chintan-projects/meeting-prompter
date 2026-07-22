"""Unit tests for the Corpus & Retrieval Lab's pure logic.

Covers the deterministic, no-network pieces: coverage / calibration aggregation,
plus the lab's re-export surface (clean_markdown and the distiller moved to
lib/corpus/ — tested in tests/test_corpus_distiller.py). The cloud judge and RAG
retrieval paths are integration concerns and are exercised live in the lab, not here.
"""

from __future__ import annotations

from lib.corpus.text import clean_markdown as lib_clean_markdown
from scripts.lab.pipeline import aggregate_calibration, aggregate_coverage, clean_markdown


def test_lab_reexports_clean_markdown_from_lib_corpus() -> None:
    assert clean_markdown is lib_clean_markdown


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
