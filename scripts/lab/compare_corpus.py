"""Before/after corpus comparison — does distilling move coverage?

Retrieves the same questions from the ORIGINAL corpus and the DISTILLED corpus,
judges the top borrowable card from each (cloud judge), and prints per-corpus
coverage side by side. This is the acceptance test for the distiller: the same
instrument that diagnosed the corpus as unfit proves whether reshaping fixed it.

    python -m scripts.lab.compare_corpus                 # sample questions
    python -m scripts.lab.compare_corpus "q1" "q2" ...   # your questions

Needs ANTHROPIC_API_KEY for the judge (retrieval-only preview runs without it).
Run the distiller first:  python -m scripts.lab.distiller <src.md> [--backend cloud]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from lib.config import load_config
from lib.paths import get_docs_dir
from lib.rag import RAGConfig
from lib.rag_engine import RAGEngine
from scripts.lab.pipeline import (
    BORROWABLE_MIN_WORDS,
    RATING_RANK,
    SAMPLE_SPANS,
    TOP_K,
    clean_markdown,
)

DISTILLED_DIR = Path("data/distilled")
DISTILLED_DB = Path("data/rag_distilled.db")


def _mk(docs_dir: Path, db: Path) -> RAGEngine:
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


def _best_rating(eng: RAGEngine, span: str, use_judge: bool) -> dict[str, Any]:
    """Retrieve top cards, judge each (or preview), return the best rating."""
    from scripts.lab import judge as _judge

    results = eng._pipeline.retrieve(span, top_k=TOP_K)
    best_rank, best = -1, {"rating": "gap", "doc": "", "reason": "no cards"}
    for r in results:
        cleaned = clean_markdown(r.chunk_text)
        if len(cleaned.split()) < BORROWABLE_MIN_WORDS:
            continue
        if not use_judge:
            # retrieval-only preview: report the top doc/heading, no rating
            return {
                "rating": "?",
                "doc": Path(r.document_path).name,
                "heading": (r.heading_path or r.section_heading or "")[-60:],
                "cosine": round(r.semantic_score, 3),
            }
        v = _judge.judge(span, cleaned)
        if "error" in v:
            return {"rating": "error", "reason": v["error"]}
        rank = RATING_RANK.get(v.get("rating", ""), 0)
        if rank > best_rank:
            best_rank, best = rank, {
                "rating": v["rating"],
                "doc": Path(r.document_path).name,
                "reason": v.get("reason", ""),
            }
    return best


def main() -> None:
    questions = sys.argv[1:] or SAMPLE_SPANS
    if not (DISTILLED_DIR.exists() and any(DISTILLED_DIR.glob("*.md"))):
        print(f"No distilled corpus in {DISTILLED_DIR}. Run scripts.lab.distiller first.")
        return

    from scripts.lab import judge as _judge

    use_judge = _judge.credential_hint() is None
    if not use_judge:
        print(f"(no credential — retrieval-only preview; {_judge.credential_hint()})\n")

    cfg = load_config()
    for f in ("data/rag_distilled.db", "data/rag_distilled.db-wal", "data/rag_distilled.db-shm"):
        Path(f).unlink(missing_ok=True)
    orig = _mk(get_docs_dir(cfg.paths.docs_dir), Path("data/rag.db"))
    dist = _mk(DISTILLED_DIR, DISTILLED_DB)

    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for q in questions:
        rows.append((q, _best_rating(orig, q, use_judge), _best_rating(dist, q, use_judge)))

    def _cov(idx: int) -> str:
        goods = sum(1 for _, *bs in rows if RATING_RANK.get(bs[idx]["rating"], 0) >= 3)
        return f"{round(100 * goods / len(rows))}%  ({goods}/{len(rows)} good)"

    print(f"{'QUESTION':<52}  {'ORIGINAL':<22}  DISTILLED")
    print("-" * 100)
    for q, bo, bd in rows:
        print(f"{q[:50]:<52}  {bo.get('rating', '?'):<22}  {bd.get('rating', '?')}")
    if use_judge:
        print("-" * 100)
        print(f"{'COVERAGE (borrowable good answer)':<52}  {_cov(0):<22}  {_cov(1)}")


if __name__ == "__main__":
    main()
