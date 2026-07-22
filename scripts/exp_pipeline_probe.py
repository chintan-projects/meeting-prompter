"""E-01 pipeline probe — detached, one selected span through the WHOLE pipeline.

Walks a single transcript span through every stage the app would, but standalone
(no FastAPI/Tauri): reuses the library pieces directly so we can see each stage
in isolation and compare model choices.

    STAGE 1  classify   — encoder trigger router (LFM2.5-Encoder-350M + adapter)
                          vs the heuristic question score
    STAGE 2  retrieve   — hybrid fusion (BM25 + vector), pre-rerank order
    STAGE 3  rerank     — heuristic re-ranker, post-rerank order + movement
    STAGE 4  answer     — 1.2B + 2.6B (llama.cpp) and 350M-Extract (transformers)

Feeds D-03 / E-01 (docs/architecture/open-decisions-log.md). Each stage is
defensive: a failure degrades that stage, the rest still runs.

Usage:
    python scripts/exp_pipeline_probe.py "your selected transcript text"
    python scripts/exp_pipeline_probe.py          # uses a sample span
"""

import gc
import sys
import time
from pathlib import Path

from lib.config import load_config
from lib.paths import get_docs_dir, get_models_dir
from lib.rag import RAGConfig
from lib.rag.rank.heuristic import HeuristicRanker
from lib.rag.retrieval.engine import retrieve as rag_retrieve
from lib.rag_engine import RAGEngine
from lib.rag_generator import RAGAnswerGenerator

MODELS_DIR = get_models_dir()
SAMPLE = "How should we generate synthetic data to fine-tune a small model without a GPU?"
TOP_K = 5
MAX_TOKENS = 140

LLAMA_CANDIDATES = [
    ("1.2B-Instruct", MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
    ("2.6B", MODELS_DIR / "LFM2.5-2.6B-Q4_K_M.gguf"),
]
EXTRACT_DIR = MODELS_DIR / "LFM2.5-350M-Extract-023-v1"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_engine() -> RAGEngine:
    cfg = load_config()
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
    return RAGEngine(get_docs_dir(cfg.paths.docs_dir), db_path=Path(cfg.rag.db_path), config=rc)


def stage_classify(span: str) -> None:
    rule("STAGE 1 — CLASSIFY (encoder trigger router vs heuristic)")
    try:
        from lib.intelligence.heads.trigger_router import TriggerRouterHead

        head = TriggerRouterHead(models_dir=MODELS_DIR, enabled=True)
        if not head.can_load():
            print("  encoder router: not loadable (weights or peft missing) — skipped")
        else:
            pred = head.predict(span)
            print(
                f"  encoder router → {pred[0]}  (conf={pred[1]:.3f})"
                if pred
                else "  encoder router → (no prediction)"
            )
    except Exception as e:  # noqa: BLE001 — harness: degrade, keep going
        print(f"  encoder router ERROR: {e!r}")
    try:
        from lib.triggers.question_trigger import score_question

        print(f"  heuristic question-score → {score_question(span):.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"  heuristic ERROR: {e!r}")


def _fmt(results: list) -> None:
    for i, r in enumerate(results, 1):
        doc = Path(r.document_path).name
        print(
            f"    {i}. fused={r.score:.3f}  bm25={r.lexical_score:.3f}  "
            f"cos={r.semantic_score:.3f}   {doc}"
        )


def stage_retrieve_rerank(eng: RAGEngine, span: str) -> str:
    rule("STAGE 2/3 — RETRIEVE (fusion) → RERANK (heuristic)")
    pre = rag_retrieve(eng._conn, span, eng._embedder, eng._config, top_k=TOP_K, ranker=None)
    post = rag_retrieve(
        eng._conn, span, eng._embedder, eng._config, top_k=TOP_K, ranker=HeuristicRanker(eng._conn)
    )
    print("  PRE-rerank (fusion order):")
    _fmt(pre)
    print("\n  POST-rerank (heuristic):")
    _fmt(post)

    pre_ids = [r.chunk_id for r in pre]
    deltas = []
    for new_rank, r in enumerate(post, 1):
        old_rank = pre_ids.index(r.chunk_id) + 1 if r.chunk_id in pre_ids else None
        if old_rank is not None and old_rank != new_rank:
            deltas.append(f"chunk {old_rank}→{new_rank}")
    print("\n  rerank movement:", "; ".join(deltas) if deltas else "order unchanged")

    return "\n\n---\n\n".join(r.chunk_text for r in post)


def answer_llama(name: str, path: Path, span: str, context: str) -> None:
    if not path.exists():
        print(f"\n  ── {name} ──  SKIP (not found)")
        return
    print(f"\n  ── {name} ──  (llama.cpp)")
    gen = None
    try:
        gen = RAGAnswerGenerator(path)
        t0 = time.time()
        ans = gen.generate(span, context, max_tokens=MAX_TOKENS)
        print(f"  {(time.time() - t0) * 1000:.0f} ms | {ans.strip()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {e!r}")
    finally:
        del gen
        gc.collect()


def answer_extract(span: str, context: str) -> None:
    print("\n  ── 350M-Extract ──  (transformers / safetensors)")
    if not EXTRACT_DIR.exists():
        print("  SKIP (dir not found)")
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(str(EXTRACT_DIR), trust_remote_code=True)
        except Exception:
            # transformers 4.56 doesn't know this model's custom tokenizer class;
            # load the fast tokenizer straight from tokenizer.json + its template.
            from transformers import PreTrainedTokenizerFast

            tok = PreTrainedTokenizerFast(
                tokenizer_file=str(EXTRACT_DIR / "tokenizer.json"),
                bos_token="<|startoftext|>",
                eos_token="<|im_end|>",
                pad_token="<|pad|>",
                unk_token="<|unk|>",
            )
            tok.chat_template = (EXTRACT_DIR / "chat_template.jinja").read_text(encoding="utf-8")
        model = AutoModelForCausalLM.from_pretrained(
            str(EXTRACT_DIR), trust_remote_code=True, torch_dtype=torch.float32
        )
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(dev).eval()
        msgs = [
            {
                "role": "system",
                "content": "Answer the question using ONLY the context. "
                "Be concise. If the context lacks the answer, say so.",
            },
            {"role": "user", "content": f"CONTEXT:\n{context[:6000]}\n\nQUESTION: {span}"},
        ]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(dev) for k, v in enc.items() if k != "token_type_ids"}
        n_in = enc["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False)
        dt = (time.time() - t0) * 1000
        txt = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        print(f"  {dt:.0f} ms | {txt}")
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {e!r}")


def main() -> None:
    span = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    rule("E-01 PIPELINE PROBE — SELECTED SPAN")
    print(f"  {span!r}")

    eng = build_engine()
    stage_classify(span)
    context = stage_retrieve_rerank(eng, span)

    rule("STAGE 4 — ANSWER (same reranked context, 3 models)")
    for name, path in LLAMA_CANDIDATES:
        answer_llama(name, path, span, context)
    answer_extract(span, context)

    rule("DONE — record against D-03 in open-decisions-log.md")


if __name__ == "__main__":
    main()
