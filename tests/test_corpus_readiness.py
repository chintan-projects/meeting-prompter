"""Unit tests for lib.corpus.readiness (F-703) — aggregation, rater, scorer, API.

Everything here is local and deterministic: retrieval is stubbed with synthetic
RetrievalResult objects, and the rater under test is the shipped heuristic (plus
a stub rater for the aggregation contract). The cloud judge is calibration-only
and is not exercised here.
"""

from __future__ import annotations

from typing import Any

from lib.corpus.readiness import (
    BORROWABLE_MIN_WORDS,
    GOOD_COSINE,
    PARTIAL_COSINE,
    aggregate_coverage,
    borrowable_card,
    heuristic_rater,
    readiness,
    score_question,
)
from lib.rag.types import Citation, RetrievalResult


def _result(text: str, cosine: float, doc: str = "doc.md", chunk_id: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_path=f"context/{doc}",
        section_heading="Section",
        heading_path="Part > Section",
        chunk_text=text,
        chunk_index=0,
        score=cosine,
        lexical_score=0.0,
        semantic_score=cosine,
        citation=Citation(
            document_path=f"context/{doc}",
            document_name=doc,
            section_heading="Section",
            heading_path="Part > Section",
            page_range=(None, None),
            chunk_id=chunk_id,
            chunk_index=0,
        ),
    )


PROSE = (
    "INT4 quantization costs about one to three percent accuracy and degrades most "
    "on multi-step reasoning tasks while factual recall is barely affected."
)


# --- aggregate_coverage (moved from scripts/lab/pipeline.py) ----------------
def _rec(span: str, chunk_id: int, rating: str, source: str = "human") -> dict[str, Any]:
    return {"span": span, "chunk_id": chunk_id, "rating": rating, "doc": "d.md", "source": source}


def test_aggregate_coverage_best_per_question() -> None:
    records = [
        _rec("q1", 1, "wrong"),
        _rec("q1", 2, "good"),
        _rec("q2", 1, "partial"),
        _rec("q3", 1, "noise"),
    ]
    cov = aggregate_coverage(records, "human")
    assert (cov["good"], cov["partial"], cov["gap"]) == (1, 1, 1)


# --- borrowable_card --------------------------------------------------------
def test_borrowable_card_cleans_and_flags_shape() -> None:
    card = borrowable_card(_result(PROSE, 0.7))
    assert card["answer_shaped"] and card["doc"] == "doc.md" and card["cosine"] == 0.7

    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert not borrowable_card(_result(table, 0.9))["answer_shaped"]


# --- heuristic_rater --------------------------------------------------------
def test_rater_noise_for_non_answer_shaped() -> None:
    card = {"answer_shaped": False, "cosine": 0.9, "text": "x"}
    assert heuristic_rater("any question", card)["rating"] == "noise"


def test_rater_good_needs_confidence_and_term_coverage() -> None:
    card = borrowable_card(_result(PROSE, GOOD_COSINE + 0.05))
    verdict = heuristic_rater(
        "How much does INT4 quantization hurt accuracy and where does it degrade?", card
    )
    assert verdict["rating"] == "good"


def test_rater_partial_when_terms_partly_covered() -> None:
    # High cosine but the card only touches half the question's terms → partial.
    card = borrowable_card(
        _result("INT4 quantization costs about one to three percent accuracy.", 0.7)
    )
    verdict = heuristic_rater(
        "How much does INT4 quantization hurt accuracy, and where does it degrade most?",
        card,
    )
    assert verdict["rating"] == "partial"


def test_rater_wrong_below_confidence_floor() -> None:
    card = borrowable_card(_result(PROSE, PARTIAL_COSINE - 0.1))
    assert heuristic_rater("Unrelated question about pricing tiers?", card)["rating"] == "wrong"


# --- score_question / readiness --------------------------------------------
def test_score_question_picks_best_card() -> None:
    cards = [
        {"answer_shaped": False, "cosine": 0.9, "text": "", "doc": "a.md", "heading": "h1"},
        borrowable_card(_result(PROSE, 0.8, doc="b.md")),
    ]
    best = score_question(
        "How much does INT4 quantization hurt accuracy, where degrade?", cards, heuristic_rater
    )
    assert best["best"] == "good" and best["doc"] == "b.md"


def test_score_question_empty_cards_is_gap() -> None:
    row = score_question("q", [], heuristic_rater)
    assert row["best"] == "gap" and row["reason"] == "nothing retrieved"


def test_readiness_shape_and_score_with_stub_retrieval() -> None:
    def retrieve(query: str, top_k: int) -> list[RetrievalResult]:
        if "INT4" in query:
            return [_result(PROSE, 0.8)]
        return []

    def stub_rater(question: str, card: dict[str, Any]) -> dict[str, str]:
        return {"rating": "good", "reason": "stub"}

    out = readiness(
        retrieve,
        ["How much does INT4 hurt?", "What is our refund policy?"],
        rater=stub_rater,
    )
    assert out["questions"] == 2
    assert out["good"] == 1 and out["gap"] == 1
    assert out["score_pct"] == 50
    assert [g["question"] for g in out["gaps"]] == ["What is our refund policy?"]
    assert {"question", "best", "reason", "doc", "heading"} <= set(out["rows"][0])


def test_readiness_empty_questions_scores_zero() -> None:
    out = readiness(lambda q, k: [], [])
    assert out["score_pct"] == 0 and out["questions"] == 0


def test_min_words_constant_matches_lab() -> None:
    # The lab re-exports these — keep one source of truth.
    from scripts.lab.pipeline import BORROWABLE_MIN_WORDS as lab_min

    assert lab_min == BORROWABLE_MIN_WORDS
