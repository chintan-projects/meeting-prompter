"""E-01 — Select-driven model + retrieval comparison harness.

Given a transcript span (as if you selected it in the Tauri app), show:
  1. Retrieval introspection — which chunks the hybrid engine pulls, with the
     per-stage breakdown (BM25 lexical / vector semantic / fused final).
  2. Answer 3-way — the SAME retrieved context answered by each candidate
     generator (350M-Extract, 1.2B-Instruct, 2.6B), with latency.

Feeds decision D-03 (see docs/architecture/open-decisions-log.md). This is the
CLI/experiment form; the winner + the select flow later move into the Tauri UI.

Usage:
    python scripts/exp_model_retrieval_compare.py "your selected transcript text"
    python scripts/exp_model_retrieval_compare.py            # uses a sample span
"""

import gc
import sys
import time
from pathlib import Path

from lib.config import load_config
from lib.paths import get_docs_dir, get_models_dir
from lib.rag import RAGConfig
from lib.rag_engine import RAGEngine
from lib.rag_generator import RAGAnswerGenerator

# Resolve exactly like the app (handles ${HOME}-style MODELS_DIR via expandvars).
MODELS_DIR = get_models_dir()

# Candidate generators — all GGUF, same llama.cpp path, swapped by model path.
CANDIDATES = [
    ("350M-Extract", MODELS_DIR / "LFM2.5-350M-Extract-023-v1" / "extract-023-v1.gguf"),
    ("1.2B-Instruct", MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
    ("2.6B", MODELS_DIR / "LFM2.5-2.6B-Q4_K_M.gguf"),
]

SAMPLE_SPAN = "How should we generate synthetic data to fine-tune a small model without a GPU?"

TOP_K = 5
MAX_TOKENS = 160


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_engine() -> RAGEngine:
    cfg = load_config()
    docs = get_docs_dir(cfg.paths.docs_dir)
    rc = RAGConfig(
        max_chunk_tokens=cfg.rag.max_chunk_tokens,
        chunk_overlap_tokens=cfg.rag.chunk_overlap_tokens,
        lexical_weight=cfg.rag.lexical_weight,
        semantic_weight=cfg.rag.semantic_weight,
        lexical_top_k=cfg.rag.lexical_top_k,
        semantic_top_k=cfg.rag.semantic_top_k,
        embedding_model=cfg.rag.embedding_model,
        embedding_dimension=cfg.rag.embedding_dimension,
    )
    return RAGEngine(docs, db_path=Path(cfg.rag.db_path), config=rc)


def main() -> None:
    span = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_SPAN

    _rule("E-01  SELECTED TRANSCRIPT SPAN")
    print(f"  {span!r}")

    eng = build_engine()

    # --- Panel 1: retrieval introspection -----------------------------------
    results = eng._pipeline.retrieve(span, top_k=TOP_K)
    _rule(
        f"PANEL 1 — RETRIEVAL (hybrid: BM25 {eng._config.lexical_weight} + "
        f"vector {eng._config.semantic_weight}, then re-rank)"
    )
    if not results:
        print("  (no chunks retrieved)")
    for i, r in enumerate(results, 1):
        doc = Path(r.document_path).name
        head = r.heading_path or r.section_heading or "(root)"
        print(
            f"\n  #{i}  fused={r.score:.3f}   bm25={r.lexical_score:.3f}   "
            f"cosine={r.semantic_score:.3f}"
        )
        print(f"      {doc}  ›  {head}")
        snippet = " ".join(r.chunk_text.split())[:160]
        print(f"      “{snippet}…”")

    # Context handed to every generator = the same top chunks (as query() joins them).
    context = "\n\n---\n\n".join(r.chunk_text for r in results)

    # --- Panel 2: answer, 3-way ---------------------------------------------
    _rule("PANEL 2 — ANSWER, SAME CONTEXT, 3 MODELS")
    print(
        "  Note: 350M-Extract is trained for field-extraction with a YAML-schema\n"
        "  prompt; here it runs the standard RAG prompt for an apples-to-apples\n"
        "  baseline. A fair extract test needs its own prompt (follow-up).\n"
    )
    for name, path in CANDIDATES:
        if not path.exists():
            print(f"\n  [{name}] SKIP — not found at {path}")
            continue
        print(f"\n  ── {name} ──  ({path.name})")
        try:
            gen = RAGAnswerGenerator(path)
            t0 = time.time()
            answer = gen.generate(span, context, max_tokens=MAX_TOKENS)
            dt = (time.time() - t0) * 1000
            print(f"  latency: {dt:.0f} ms")
            print(f"  answer : {answer.strip()}")
        except Exception as e:  # noqa: BLE001 — experiment harness, report and continue
            print(f"  ERROR: {e}")
        finally:
            gen = None  # noqa: F841 — drop llama handle before loading the next model
            gc.collect()

    _rule("DONE — record findings against D-03 in open-decisions-log.md")


if __name__ == "__main__":
    main()
