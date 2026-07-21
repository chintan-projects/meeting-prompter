"""Model & retrieval comparison lab — analysis backend.

Reuses the real library pieces (no reimplementation) but exposes each pipeline
stage *separately* so a human can look at all the options side-by-side and make
the model call themselves:

    CLASSIFY   encoder trigger router + heuristic question score
    RETRIEVE   BM25 arm | vector arm | fused | reranked  (each shown on its own)
    ANSWER     1.2B | 2.6B (llama.cpp) | 350M-Extract (subprocess, newer runtime)

This is deliberately a *harness*, not a decision: it surfaces scores and outputs;
the judgement (which answer model, whether BM25/rerank earn their place) is the
operator's. Feeds D-03 / E-01 (docs/architecture/open-decisions-log.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

from lib.config import AppConfig, load_config
from lib.paths import get_docs_dir, get_models_dir
from lib.rag import RAGConfig
from lib.rag.index.fts import _sanitize_fts_query, fts_search
from lib.rag.index.vector import vector_search
from lib.rag.rank.heuristic import HeuristicRanker
from lib.rag.retrieval.fusion import weighted_fusion
from lib.rag_engine import RAGEngine
from lib.rag_generator import RAGAnswerGenerator

MODELS_DIR = get_models_dir()

# Answer candidates. GGUF ones run in-process via llama.cpp; Extract runs in a
# subprocess against a newer transformers runtime (its GGUF is llama.cpp-incompatible).
LLAMA_MODELS: dict[str, Path] = {
    "1.2B-Instruct": MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    "2.6B": MODELS_DIR / "LFM2.5-2.6B-Q4_K_M.gguf",
}
EXTRACT_DIR = MODELS_DIR / "LFM2.5-350M-Extract-023-v1"
_EXTRACT_RUNNER = Path(__file__).with_name("extract_runner.py")
# Interpreter with transformers>=5 + torch. Override for the isolated overlay venv.
EXTRACT_PYTHON = os.environ.get("LAB_EXTRACT_PYTHON", sys.executable)

# How many candidates to show per retrieval arm (arms fetch more; fused keeps TOP_K).
ARM_DISPLAY_K = 8
TOP_K = 5
DEFAULT_MAX_TOKENS = 160

SAMPLE_SPANS = [
    "How should we generate synthetic data to fine-tune a small model without a GPU?",
    "What teacher model should we use for distillation, and what are the licensing risks?",
    "Can we run LFM2 on device, and what are the memory and latency numbers?",
]


class LabEngine:
    """Holds the warm RAG engine + cached answer generators for the lab server."""

    def __init__(self) -> None:
        self.cfg: AppConfig = load_config()
        self.engine: RAGEngine = self._build_engine()
        self._conn: sqlite3.Connection = self.engine._conn
        self._generators: dict[str, RAGAnswerGenerator] = {}
        self._router: Any = None
        self._router_tried = False

    def _build_engine(self) -> RAGEngine:
        c = self.cfg
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
        return RAGEngine(get_docs_dir(c.paths.docs_dir), db_path=Path(c.rag.db_path), config=rc)

    # --- meta lookup --------------------------------------------------------
    def _chunk_meta(self, chunk_id: int) -> dict[str, Any]:
        row = self._conn.execute(
            """SELECT d.filename, c.content, c.section_id
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.id = ?""",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return {"doc": "(missing)", "heading": "", "snippet": ""}
        heading = ""
        if row[2] is not None:
            sec = self._conn.execute(
                "SELECT heading_path, heading FROM sections WHERE id = ?", (row[2],)
            ).fetchone()
            if sec:
                heading = sec[0] or sec[1] or ""
        snippet = " ".join((row[1] or "").split())[:200]
        return {"doc": row[0], "heading": heading, "snippet": snippet}

    def _row(self, chunk_id: int, **scores: float) -> dict[str, Any]:
        meta = self._chunk_meta(chunk_id)
        return {"chunk_id": chunk_id, **meta, **scores}

    # --- stage 1: classify --------------------------------------------------
    def classify(self, span: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not self._router_tried:
            self._router_tried = True
            try:
                from lib.intelligence.heads.trigger_router import TriggerRouterHead

                head = TriggerRouterHead(models_dir=MODELS_DIR, enabled=True)
                self._router = head if head.can_load() else None
            except Exception as e:  # noqa: BLE001 — harness: degrade, keep going
                out["router_error"] = repr(e)
                self._router = None
        if self._router is not None:
            try:
                pred = self._router.predict(span)
                if pred:
                    out["router_label"] = pred[0]
                    out["router_confidence"] = round(float(pred[1]), 4)
            except Exception as e:  # noqa: BLE001
                out["router_error"] = repr(e)
        elif "router_error" not in out:
            out["router_note"] = "encoder router not loadable (weights/peft absent)"
        try:
            from lib.triggers.question_trigger import score_question

            out["heuristic_question_score"] = round(float(score_question(span)), 4)
        except Exception as e:  # noqa: BLE001
            out["heuristic_error"] = repr(e)
        return out

    # --- stage 2/3: retrieve + rerank --------------------------------------
    def retrieve_stages(self, span: str) -> dict[str, Any]:
        cfg = self.engine._config
        conn = self._conn
        emb = self.engine._embedder

        sanitized = _sanitize_fts_query(span)
        lexical = fts_search(conn, span, cfg.lexical_top_k, cfg)
        embed_query = getattr(emb, "embed_query", None)
        query_emb = embed_query(span) if callable(embed_query) else emb.embed(span)
        semantic = vector_search(conn, query_emb, cfg.semantic_top_k)
        fused = weighted_fusion(
            lexical,
            semantic,
            lexical_weight=cfg.lexical_weight,
            semantic_weight=cfg.semantic_weight,
            top_k=TOP_K,
        )
        reranked = HeuristicRanker(conn).rank(span, fused, cfg)

        bm25_panel = [
            self._row(h.chunk_id, score=round(h.score, 4)) for h in lexical[:ARM_DISPLAY_K]
        ]
        vec_panel = [
            self._row(h.chunk_id, score=round(h.score, 4)) for h in semantic[:ARM_DISPLAY_K]
        ]
        fused_panel = [
            self._row(
                h.chunk_id,
                fused=round(h.fused_score, 4),
                bm25=round(h.lexical_score, 4),
                cosine=round(h.semantic_score, 4),
            )
            for h in fused
        ]
        rerank_panel = [
            self._row(
                h.chunk_id,
                fused=round(h.fused_score, 4),
                bm25=round(h.lexical_score, 4),
                cosine=round(h.semantic_score, 4),
            )
            for h in reranked
        ]

        # Observations the operator can verify at a glance.
        bm25_idle = all(h.lexical_score == 0.0 for h in fused) if fused else True
        order_changed = [h.chunk_id for h in fused] != [h.chunk_id for h in reranked]

        context = "\n\n---\n\n".join(self._chunk_text(h.chunk_id) for h in reranked)
        return {
            "sanitized_fts_query": sanitized or "(empty after stop-word removal)",
            "weights": {"lexical": cfg.lexical_weight, "semantic": cfg.semantic_weight},
            "bm25": bm25_panel,
            "vector": vec_panel,
            "fused": fused_panel,
            "reranked": rerank_panel,
            "bm25_idle": bm25_idle,
            "rerank_changed_order": order_changed,
            "context": context,
        }

    def _chunk_text(self, chunk_id: int) -> str:
        row = self._conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return row[0] if row else ""

    # --- stage 4: answers ---------------------------------------------------
    def answer_llama(self, key: str, span: str, context: str, max_tokens: int) -> dict[str, Any]:
        path = LLAMA_MODELS[key]
        if not path.exists():
            return {"model": key, "available": False, "note": f"not found: {path.name}"}
        gen = self._generators.get(key)
        if gen is None:
            gen = RAGAnswerGenerator(path)
            self._generators[key] = gen
        try:
            t0 = time.time()
            text = gen.generate(span, context, max_tokens=max_tokens)
            return {
                "model": key,
                "available": True,
                "latency_ms": round((time.time() - t0) * 1000),
                "text": text.strip(),
            }
        except Exception as e:  # noqa: BLE001
            return {"model": key, "available": True, "error": repr(e)}

    def answer_extract(self, span: str, context: str, max_tokens: int) -> dict[str, Any]:
        key = "350M-Extract"
        if not EXTRACT_DIR.exists():
            return {"model": key, "available": False, "note": f"dir not found: {EXTRACT_DIR.name}"}
        payload = json.dumps(
            {
                "model_dir": str(EXTRACT_DIR),
                "question": span,
                "context": context,
                "max_tokens": max_tokens,
            }
        )
        try:
            proc = subprocess.run(
                [EXTRACT_PYTHON, str(_EXTRACT_RUNNER)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except Exception as e:  # noqa: BLE001
            return {"model": key, "available": True, "error": f"subprocess failed: {e!r}"}
        if proc.returncode != 0:
            return {
                "model": key,
                "available": True,
                "error": (proc.stderr or proc.stdout or "unknown error").strip()[-800:],
                "runtime": EXTRACT_PYTHON,
            }
        try:
            data: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            return {"model": key, "available": True, "error": f"bad runner output: {e!r}"}
        data.update({"model": key, "available": True, "runtime": EXTRACT_PYTHON})
        return data

    def answers(self, span: str, context: str, max_tokens: int) -> list[dict[str, Any]]:
        return [
            self.answer_llama("1.2B-Instruct", span, context, max_tokens),
            self.answer_llama("2.6B", span, context, max_tokens),
            self.answer_extract(span, context, max_tokens),
        ]

    def analyze(self, span: str, max_tokens: Optional[int] = None) -> dict[str, Any]:
        mt = max_tokens or DEFAULT_MAX_TOKENS
        classification = self.classify(span)
        retrieval = self.retrieve_stages(span)
        answers = self.answers(span, retrieval["context"], mt)
        return {
            "span": span,
            "docs_dir": self.cfg.paths.docs_dir,
            "answer_model_in_config": self.cfg.models.generation.model_file,
            "classification": classification,
            "retrieval": retrieval,
            "answers": answers,
        }
