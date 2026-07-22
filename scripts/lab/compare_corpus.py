"""Before/after corpus comparison — does distilling move coverage?

Runs the same question set against the ORIGINAL corpus and the DISTILLED corpus
through the readiness scorer (retrieval → borrowable cards → merged top-2
candidate → a rater) and prints per-corpus coverage side by side. This is the
acceptance test for the distiller: the same instrument that diagnosed the corpus
as unfit proves whether reshaping fixed it.

Raters:
  - judge (default when ANTHROPIC_API_KEY is present): cloud Opus — offline
    validation only (ADR-001).
  - local: the shipped heuristic rater (lib.corpus.readiness) — no network.

    python -m scripts.lab.compare_corpus                          # sample questions
    python -m scripts.lab.compare_corpus --questions tests/eval/corpus_questions.yaml
    python -m scripts.lab.compare_corpus --rater local "q1" "q2"

Run the distiller first:  python -m scripts.lab.distiller <src.md> [--backend cloud]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from lib.config import load_config
from lib.corpus.readiness import Rater, heuristic_rater, readiness
from lib.paths import get_docs_dir
from lib.rag import RAGConfig
from lib.rag_engine import RAGEngine
from scripts.lab.pipeline import SAMPLE_SPANS, TOP_K

DISTILLED_DIR = Path("data/distilled")
# NOTE: no module-level DB paths. Each _score() call indexes into its own temp
# directory — shared fixed paths made concurrent runs corrupt each other.


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


def judge_rater(question: str, card: dict[str, Any]) -> dict[str, str]:
    """Adapt the cloud judge to the Rater protocol. Errors abort the run —
    a silently degraded judge would misreport coverage."""
    from scripts.lab import judge as _judge

    v = _judge.judge(question, str(card.get("text", "")))
    if "error" in v:
        raise RuntimeError(f"judge failed: {v['error']}")
    return {"rating": str(v.get("rating", "")), "reason": str(v.get("reason", ""))}


def load_questions(path: Path) -> list[str]:
    """Question texts from a corpus_questions.yaml (list of {id, text, ...})."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(q["text"]) for q in data["questions"]]


def _score(docs_dir: Path, label: str, questions: list[str], rater: Rater) -> dict[str, Any]:
    """Index `docs_dir` into a PRIVATE throwaway DB and score it.

    The index must be built fresh (the corpus changes between runs), but it must
    not live at a fixed shared path: this used to delete `data/rag_*.db*` and
    rebuild in place, so two concurrent runs — or a background job and a terminal
    — silently destroyed each other's SQLite files mid-query ("disk I/O error").
    A per-run temp directory makes concurrent comparisons safe by construction.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"corpus_cmp_{label}_"))
    engine = _mk(docs_dir, tmp / "index.db")
    try:
        return readiness(engine.retrieve, questions, rater=rater, top_k=TOP_K)
    finally:
        engine.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Original vs distilled corpus coverage.")
    ap.add_argument("questions", nargs="*", help="questions (default: --questions file or samples)")
    ap.add_argument("--questions-file", default="", help="corpus_questions.yaml path")
    ap.add_argument("--rater", choices=["auto", "judge", "local"], default="auto")
    args = ap.parse_args()

    if args.questions:
        questions = list(args.questions)
    elif args.questions_file:
        questions = load_questions(Path(args.questions_file))
    else:
        questions = SAMPLE_SPANS

    if not (DISTILLED_DIR.exists() and any(DISTILLED_DIR.glob("*.md"))):
        print(f"No distilled corpus in {DISTILLED_DIR}. Run scripts.lab.distiller first.")
        return

    from scripts.lab import judge as _judge

    rater: Rater
    if args.rater == "judge" or (args.rater == "auto" and _judge.credential_hint() is None):
        rater, rater_name = judge_rater, f"judge ({_judge.JUDGE_MODEL})"
    else:
        rater, rater_name = heuristic_rater, "local heuristic (shipped)"
        if args.rater == "auto":
            print(f"(no credential — using local rater; {_judge.credential_hint()})")

    cfg = load_config()
    orig = _score(get_docs_dir(cfg.paths.docs_dir), "orig", questions, rater)
    dist = _score(DISTILLED_DIR, "dist", questions, rater)

    print(f"\nRater: {rater_name}    Questions: {len(questions)}")
    print(f"{'QUESTION':<58}  {'ORIGINAL':<10}  DISTILLED")
    print("-" * 92)
    for ro, rd in zip(orig["rows"], dist["rows"]):
        mark = " (merged)" if rd.get("merged") else ""
        print(f"{ro['question'][:56]:<58}  {ro['best']:<10}  {rd['best']}{mark}")
    print("-" * 92)
    for name, res in (("ORIGINAL", orig), ("DISTILLED", dist)):
        print(
            f"{name:<10} coverage: {res['score_pct']}% good "
            f"({res['good']}/{res['questions']}; partial {res['partial']}, gap {res['gap']})"
        )


if __name__ == "__main__":
    main()
