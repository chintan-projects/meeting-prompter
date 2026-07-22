"""Corpus readiness score — the fit-for-purpose onboarding gate (F-703, D-11).

Given a corpus and a set of likely meeting questions, answer: *can this corpus
answer these meetings?* Each question runs through retrieval → borrowable cards →
a rater; the aggregate is a readiness score plus a gap list the user can act on
before relying on the corpus live.

The shipped rater is LOCAL and heuristic (answer-shapedness + retrieval
confidence + question-term overlap) per ADR-001 — the cloud judge remains an
offline calibration instrument only (scripts/lab/judge.py). The question set
must stay independent of the distiller so coverage reflects a real corpus, not
a gamed one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Sequence, Union

from lib.corpus.text import clean_markdown
from lib.rag.types import RetrievalResult

RATING_RANK = {"good": 3, "partial": 2, "wrong": 1, "noise": 0}
BORROWABLE_MIN_WORDS = 8  # below this, a cleaned chunk isn't an answer-shaped unit
DEFAULT_TOP_K = 5

# Heuristic rater thresholds. Cosine bands follow the live config's semantics
# (rag_confidence_minimum=0.35 is the floor below which retrieval stays silent);
# overlap asks whether the card actually touches the question's terms.
GOOD_COSINE = 0.60
PARTIAL_COSINE = 0.35
GOOD_OVERLAP = 0.50
PARTIAL_OVERLAP = 0.34

_STOP_WORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or should "
    "that the to we what when where which who why will with you your".split()
)

#: A rater maps (question, card) → {"rating": good|partial|wrong|noise, "reason": str}.
Rater = Callable[[str, dict[str, Any]], dict[str, str]]
#: Retrieval callable: (query, top_k) → results. RAGEngine.retrieve satisfies this.
Retrieve = Callable[[str, int], list[RetrievalResult]]


def aggregate_coverage(records: list[dict[str, Any]], source: str = "human") -> dict[str, Any]:
    """Pure: coverage counts for one rating source. A question is `good` if any of
    its chunks has a borrowable answer, `partial` if the best is partial, else `gap`.

    Collapses to the LATEST rating per (span, chunk) first, so re-rating a chunk
    downward correctly lowers coverage — then takes the best chunk per question.
    """
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        if rec.get("source", "human") != source:
            continue
        span = str(rec.get("span", "")).strip()
        if not span:
            continue
        latest[(span, int(rec.get("chunk_id", -1)))] = rec  # later write wins
    best: dict[str, tuple[int, str, str]] = {}
    for (span, _cid), rec in latest.items():
        rank = RATING_RANK.get(rec.get("rating", ""), 0)
        if span not in best or rank > best[span][0]:  # best chunk wins for the question
            best[span] = (rank, rec.get("rating", ""), rec.get("doc", ""))
    good = partial = gap = 0
    rows: list[dict[str, Any]] = []
    for span, (rank, rating, doc) in best.items():
        rows.append({"span": span, "best": rating, "doc": doc})
        if rank >= 3:
            good += 1
        elif rank == 2:
            partial += 1
        else:
            gap += 1
    rows.sort(key=lambda r: RATING_RANK.get(r["best"], 0))  # gaps first
    return {
        "source": source,
        "questions": len(best),
        "good": good,
        "partial": partial,
        "gap": gap,
        "rows": rows,
    }


def is_heading_only(text: str, heading: str = "") -> bool:
    """True when a unit is a section title rather than an answer.

    Sectioning leaves title-only chunks in the index ("Part 1 — Quantization:
    Fewer Bits Per Weight"). A long title clears a word-count floor, so word
    count alone cannot reject it — and it then wins retrieval on exactly the
    queries it names, putting a card on screen that repeats the question back
    as a heading. Structure is the discriminator: an answer is prose, so it
    ends a sentence somewhere.
    """
    body = text.strip()
    if not body:
        return True
    # A single line with no sentence-ending punctuation is a title, not prose.
    if "\n" not in body and not re.search(r"[.!?](\s|$)", body):
        return True
    # Or the unit is literally (a prefix of) its own heading.
    h = heading.split(" > ")[-1].strip().lower()
    return bool(h) and body.lower().rstrip(".") == h.rstrip(".")


def borrowable_card(result: RetrievalResult) -> dict[str, Any]:
    """One retrieval hit as a borrowable card: cleaned prose + provenance + scores."""
    cleaned = clean_markdown(result.chunk_text)
    words = len(cleaned.split())
    heading = result.heading_path or result.section_heading or ""
    return {
        "chunk_id": result.chunk_id,
        "doc": Path(result.document_path).name,
        "heading": heading,
        "cosine": round(result.semantic_score, 4),
        "fused": round(result.score, 4),
        "text": cleaned,
        "words": words,
        "answer_shaped": words >= BORROWABLE_MIN_WORDS and not is_heading_only(cleaned, heading),
    }


def _question_terms(question: str) -> set[str]:
    """Content-bearing question terms (lowercased, stop words removed)."""
    return {
        w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOP_WORDS and len(w) > 2
    }


def _term_overlap(question: str, text: str) -> float:
    """Fraction of the question's content terms present in the card text."""
    terms = _question_terms(question)
    if not terms:
        return 0.0
    body = text.lower()
    return sum(1 for t in terms if t in body) / len(terms)


def heuristic_rater(question: str, card: dict[str, Any]) -> dict[str, str]:
    """Local, no-LLM rater: answer-shapedness + retrieval confidence + term overlap.

    A v1 proxy for the cloud judge (which stays offline, for calibration). It is
    deliberately conservative about `good`: high semantic confidence alone is not
    enough — the card must also touch most of the question's content terms, which
    is what separates "answers THIS question" from "same topic".
    """
    if not card.get("answer_shaped"):
        return {
            "rating": "noise",
            "reason": "mostly table/code/heading — not answer-shaped (corpus gap)",
        }
    cosine = float(card.get("cosine") or 0.0)
    overlap = _term_overlap(question, str(card.get("text", "")))
    if cosine >= GOOD_COSINE and overlap >= GOOD_OVERLAP:
        return {
            "rating": "good",
            "reason": f"high confidence (cos {cosine:.2f}) and covers the question terms",
        }
    if cosine >= PARTIAL_COSINE and overlap >= PARTIAL_OVERLAP:
        return {
            "rating": "partial",
            "reason": f"relevant (cos {cosine:.2f}) but only partly covers the question "
            f"(term overlap {overlap:.0%})",
        }
    return {
        "rating": "wrong",
        "reason": f"low match for this question (cos {cosine:.2f}, overlap {overlap:.0%})",
    }


def merged_card(cards: Sequence[dict[str, Any]], max_units: int = 2) -> Union[dict[str, Any], None]:
    """Merge the top answer-shaped cards into one multi-unit answer candidate.

    Compound questions whose answer spans sections (the INT4 failure mode) can be
    answered live by showing two short borrowable units together — a legit UX.
    Provenance is kept per unit in ``parts``; the merged confidence is the MIN of
    the parts (both must actually be relevant), so a strong card can't smuggle a
    weak one into a `good`.
    """
    shaped = [c for c in cards if c.get("answer_shaped")][:max_units]
    if len(shaped) < 2:
        return None
    return {
        "chunk_id": shaped[0]["chunk_id"],
        "merged": True,
        "parts": [
            {"doc": c.get("doc", ""), "heading": c.get("heading", ""), "chunk_id": c["chunk_id"]}
            for c in shaped
        ],
        "doc": " + ".join(dict.fromkeys(str(c.get("doc", "")) for c in shaped)),
        "heading": " + ".join(str(c.get("heading", "")) for c in shaped),
        "cosine": min(float(c.get("cosine") or 0.0) for c in shaped),
        "fused": min(float(c.get("fused") or 0.0) for c in shaped),
        "text": "\n\n".join(str(c.get("text", "")) for c in shaped),
        "words": sum(int(c.get("words") or 0) for c in shaped),
        "answer_shaped": True,
    }


def score_question(question: str, cards: Sequence[dict[str, Any]], rater: Rater) -> dict[str, Any]:
    """Rate every card for one question and return the best verdict row.

    A single unit beats a merged answer at equal rank (strictly-greater
    comparison, single cards first), so multi-unit only wins when it genuinely
    upgrades the rating — one glanceable card is the better live UX.
    """
    best_rank = -1
    best: dict[str, Any] = {
        "question": question,
        "best": "gap",
        "reason": "nothing retrieved",
        "doc": "",
        "heading": "",
        "merged": False,
    }
    for card in cards:
        verdict = rater(question, card)
        rank = RATING_RANK.get(verdict.get("rating", ""), 0)
        if rank > best_rank:
            best_rank = rank
            best = {
                "question": question,
                "best": verdict.get("rating", "noise"),
                "reason": verdict.get("reason", ""),
                "doc": card.get("doc", ""),
                "heading": card.get("heading", ""),
                "merged": bool(card.get("merged", False)),
            }
            if card.get("merged"):
                best["parts"] = card.get("parts", [])
    return best


def _candidates(cards: list[dict[str, Any]], multi_unit: bool) -> list[dict[str, Any]]:
    """Single cards first (preferred at equal rank), then the merged candidate."""
    if not multi_unit:
        return cards
    merged = merged_card(cards)
    return cards + [merged] if merged else cards


def readiness(
    corpus: Union[Path, Retrieve],
    questions: Sequence[str],
    *,
    rater: Rater = heuristic_rater,
    top_k: int = DEFAULT_TOP_K,
    db_path: Union[Path, None] = None,
    multi_unit: bool = True,
) -> dict[str, Any]:
    """Score a corpus against a question set → readiness % + gap list.

    Args:
        corpus: either a docs directory (a throwaway index is built for it) or a
            retrieval callable ``(query, top_k) -> list[RetrievalResult]`` over an
            already-indexed corpus (e.g. ``RAGEngine.retrieve``).
        questions: the readiness question set. Keep it independent of the
            distiller — this gate must measure the corpus, not train it.
        rater: card rater; defaults to the local heuristic (ADR-001).
        top_k: retrieval depth per question.
        db_path: index path when ``corpus`` is a directory (default: a fresh
            ``data/rag_readiness.db``, stale copies removed first).
        multi_unit: also consider a merged top-2 answer per question (compound
            questions whose answer spans sections; shown live as two snippets).

    Returns:
        ``{score_pct, questions, good, partial, gap, gaps: [...], rows: [...]}``
        where ``gaps`` are the non-good rows, worst first, each with a reason and
        the best card's provenance so the user can act on it.
    """
    if callable(corpus):
        rows = [
            score_question(
                q, _candidates([borrowable_card(r) for r in corpus(q, top_k)], multi_unit), rater
            )
            for q in questions
        ]
    else:
        engine = _build_engine(corpus, db_path)
        try:
            rows = [
                score_question(
                    q,
                    _candidates(
                        [borrowable_card(r) for r in engine.retrieve(q, top_k)], multi_unit
                    ),
                    rater,
                )
                for q in questions
            ]
        finally:
            engine.close()
    good = sum(1 for r in rows if RATING_RANK.get(r["best"], 0) >= 3)
    partial = sum(1 for r in rows if RATING_RANK.get(r["best"], 0) == 2)
    gap = len(rows) - good - partial
    gaps = sorted(
        (r for r in rows if RATING_RANK.get(r["best"], 0) < 3),
        key=lambda r: RATING_RANK.get(r["best"], 0),
    )
    return {
        "score_pct": round(100 * good / len(rows)) if rows else 0,
        "questions": len(rows),
        "good": good,
        "partial": partial,
        "gap": gap,
        "gaps": gaps,
        "rows": rows,
    }


def live_borrowable(
    results: Sequence[RetrievalResult],
    query: str,
    min_confidence: float,
    min_answer_length: int = 10,
) -> Union[dict[str, Any], None]:
    """The retrieval-first live answer (F-705/D-08): the best borrowable unit
    for a live trigger, glanceable, with expand-to-source provenance. No LLM.

    Returns ``{answer, full_text, doc, heading, confidence}`` or None when
    nothing answer-shaped clears the confidence floor — silence beats noise.
    ``answer`` is the most question-relevant sentence(s) of the unit
    (glanceable); ``full_text`` is the whole cleaned unit for expansion.
    """
    from lib.answer_extractor import extract_answer

    best = next((c for c in map(borrowable_card, results) if c["answer_shaped"]), None)
    if best is None:
        return None
    confidence = float(best.get("fused") or best.get("cosine") or 0.0)
    if confidence < min_confidence:
        return None
    full = str(best["text"])
    heading = str(best["heading"])
    glance, _ = extract_answer(full, query, max_sentences=2)
    answer = (glance or full).strip()
    # The unit is prose, but the most query-relevant line inside it can still be
    # a sub-heading — which reads as the card repeating the question back.
    if is_heading_only(answer, heading):
        answer = full.strip()
    if len(answer) < min_answer_length or is_heading_only(answer, heading):
        return None
    return {
        "answer": answer,
        "full_text": full,
        "doc": str(best["doc"]),
        "heading": str(best["heading"]),
        "confidence": confidence,
    }


def _build_engine(docs_dir: Path, db_path: Union[Path, None]) -> Any:
    """Index a docs directory into a throwaway readiness DB and return the engine."""
    from lib.config import load_config
    from lib.rag import RAGConfig
    from lib.rag_engine import RAGEngine

    db = db_path or Path("data/rag_readiness.db")
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    c = load_config()
    rc = RAGConfig(
        max_chunk_tokens=c.rag.max_chunk_tokens,
        chunk_overlap_tokens=c.rag.chunk_overlap_tokens,
        lexical_weight=c.rag.lexical_weight,
        semantic_weight=c.rag.semantic_weight,
        lexical_top_k=c.rag.lexical_top_k,
        semantic_top_k=c.rag.semantic_top_k,
        embedding_model=c.rag.embedding_model,
        embedding_dimension=c.rag.embedding_dimension,
    )
    return RAGEngine(docs_dir, db_path=db, config=rc)
