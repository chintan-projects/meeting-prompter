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
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
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

# Generation (the 3-model answer panel) is OFF by default — the lab is a corpus /
# retrieval instrument now (D-08), and loading 1.2B/2.6B/Extract per analyze is slow.
# Re-enable for a model comparison with:  LAB_GENERATE=1 uvicorn scripts.lab.server:app
GENERATE_ANSWERS = os.environ.get("LAB_GENERATE", "") == "1"

# Sample questions matched to the on-device-capability-playbook corpus (D-08 loop).
# Replace with the actual questions from a real meeting to get a true coverage baseline.
SAMPLE_SPANS = [
    "How much does INT4 quantization hurt accuracy, and where does it degrade most?",
    "What are the three levels of distillation and when do we use each?",
    "Is speculative decoding lossless, and how much speedup does it give?",
    "When should we prune versus quantize — do they stack?",
]

# Ratings persist here so corpus coverage can be aggregated across questions/sessions.
RATINGS_PATH = Path("data/corpus_ratings.jsonl")
# A question is "covered" only if some retrieved chunk is a genuinely borrowable answer.
RATING_RANK = {"good": 3, "partial": 2, "wrong": 1, "noise": 0}
BORROWABLE_MIN_WORDS = 8  # below this, a cleaned chunk isn't an answer-shaped unit


def clean_markdown(text: str) -> str:
    """Reduce a raw chunk to readable, borrowable prose.

    Drops what you would never *say* out loud — fenced code, table rows, heading
    hashes, blockquote/list markers, inline emphasis/link syntax — and collapses
    whitespace. Tables/code are removed on purpose: a chunk that's mostly table is
    not an answer-shaped unit, and the near-empty result is itself a corpus signal
    (surfaced via BORROWABLE_MIN_WORDS).
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced code blocks
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|") or re.fullmatch(r"[-|:\s]+", line):
            continue  # table row / separator
        line = re.sub(r"^#{1,6}\s*", "", line)  # heading hashes
        line = re.sub(r"^>\s*", "", line)  # blockquote
        line = re.sub(r"^[-*+]\s+", "", line)  # bullet
        line = re.sub(r"^\d+\.\s+", "", line)  # ordered list
        kept.append(line)
    out = " ".join(kept)
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)  # bold
    out = re.sub(r"\*(.+?)\*", r"\1", out)  # italic
    out = re.sub(r"`([^`]+)`", r"\1", out)  # inline code
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)  # links → text
    return re.sub(r"\s+", " ", out).strip()


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
        t0 = time.time()
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
        retrieval_ms = round((time.time() - t0) * 1000)

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
            "retrieval_ms": retrieval_ms,
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

    # --- borrowable answers: retrieve-big, clean, rate ----------------------
    def build_borrowable(
        self, span: str, retrieval: dict[str, Any], top_k: int = TOP_K
    ) -> dict[str, Any]:
        """The judging surface: each top chunk as a full, cleaned, *borrowable*
        answer — markdown stripped to readable prose — with source and score.

        This is what the user reads/says verbatim, and it doubles as the corpus
        instrument: a chunk that cleans down to almost nothing is a table/code blob,
        i.e. not answer-shaped — a corpus gap, flagged inline.
        """
        t0 = time.time()
        cards: list[dict[str, Any]] = []
        for row in retrieval["reranked"][:top_k]:
            raw = self._chunk_text(row["chunk_id"])
            cleaned = clean_markdown(raw)
            words = len(cleaned.split())
            shaped = words >= BORROWABLE_MIN_WORDS
            cards.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc": row["doc"],
                    "heading": row["heading"],
                    "cosine": row.get("cosine"),
                    "fused": row.get("fused"),
                    "text": cleaned if shaped else (cleaned or "(empty after cleaning)"),
                    "words": words,
                    "answer_shaped": shaped,
                    "note": (
                        ""
                        if shaped
                        else "mostly table/code/heading — not answer-shaped (corpus gap)"
                    ),
                }
            )
        clean_ms = round((time.time() - t0) * 1000)
        return {
            "retrieval_ms": int(retrieval.get("retrieval_ms") or 0),
            "clean_ms": clean_ms,
            "cards": cards,
        }

    # --- ratings + corpus coverage -----------------------------------------
    def record_rating(
        self, span: str, chunk_id: int, doc: str, rating: str, source: str = "human"
    ) -> None:
        """Append one judgement (good/partial/wrong/noise) to the log.

        source is "human" (operator click) or "judge" (LLM). Both are kept so the
        judge can be calibrated against the human ground truth.
        """
        RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "span": span.strip(),
            "chunk_id": chunk_id,
            "doc": doc,
            "rating": rating,
            "source": source,
        }
        with open(RATINGS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _best_by_span(self, source: str) -> dict[str, tuple[int, str, str]]:
        """For one source, the best rating per question (latest write wins ties)."""
        best: dict[str, tuple[int, str, str]] = {}
        if not RATINGS_PATH.exists():
            return best
        for line in RATINGS_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source", "human") != source:
                continue
            span = str(rec.get("span", "")).strip()
            if not span:
                continue
            rank = RATING_RANK.get(rec.get("rating", ""), 0)
            if span not in best or rank >= best[span][0]:
                best[span] = (rank, rec.get("rating", ""), rec.get("doc", ""))
        return best

    def coverage(self, source: str = "human") -> dict[str, Any]:
        """Fit-for-purpose score: fraction of questions with a borrowable ('good')
        answer in the corpus, from one rating source (human or judge).
        """
        best = self._best_by_span(source)
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

    # --- LLM-as-judge (cloud) + calibration --------------------------------
    def judge_span(self, span: str) -> dict[str, Any]:
        """Run the cloud judge over the borrowable cards for a span, record its
        verdicts (source="judge"), and return them plus refreshed calibration.
        """
        from scripts.lab import judge as _judge

        retrieval = self.retrieve_stages(span)
        borrowable = self.build_borrowable(span, retrieval)
        verdicts: list[dict[str, Any]] = []
        for c in borrowable["cards"]:
            v = _judge.judge(span, c["text"])
            v.update({"chunk_id": c["chunk_id"], "doc": c["doc"]})
            if "rating" in v:
                self.record_rating(span, c["chunk_id"], c["doc"], v["rating"], source="judge")
            verdicts.append(v)
        return {
            "model": _judge.JUDGE_MODEL,
            "verdicts": verdicts,
            "judge_coverage": self.coverage("judge"),
            "calibration": self.calibration(),
        }

    def calibration(self) -> dict[str, Any]:
        """Judge-vs-human agreement over (span, chunk) pairs rated by BOTH.

        This is the trust gate: only lean on the judge's coverage once it agrees
        with the human ground truth. Reports exact-match agreement and the rows.
        """
        human: dict[tuple[str, int], str] = {}
        judge_r: dict[tuple[str, int], str] = {}
        if RATINGS_PATH.exists():
            for line in RATINGS_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(rec.get("span", "")).strip(), int(rec.get("chunk_id", -1)))
                target = human if rec.get("source", "human") == "human" else judge_r
                target[key] = rec.get("rating", "")  # latest write wins
        pairs = sorted(set(human) & set(judge_r))
        rows = [
            {
                "span": k[0],
                "chunk_id": k[1],
                "human": human[k],
                "judge": judge_r[k],
                "match": human[k] == judge_r[k],
            }
            for k in pairs
        ]
        agree = sum(1 for r in rows if r["match"])
        pct = round(100 * agree / len(rows)) if rows else 0
        return {"pairs": len(rows), "agree": agree, "agreement_pct": pct, "rows": rows}

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
        borrowable = self.build_borrowable(span, retrieval)
        answers = self.answers(span, retrieval["context"], mt) if GENERATE_ANSWERS else []
        return {
            "span": span,
            "docs_dir": self.cfg.paths.docs_dir,
            "answer_model_in_config": self.cfg.models.generation.model_file,
            "classification": classification,
            "borrowable": borrowable,
            "retrieval": retrieval,
            "answers": answers,
        }
