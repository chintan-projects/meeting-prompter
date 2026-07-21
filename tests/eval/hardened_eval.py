"""Hardened chunk-level RAG eval (F-509).

The doc-level harness (test_rag_eval.py) saturates on the small ``context/``
corpus — MiniLM and LFM2.5-Embedding tie to three decimals, so it cannot prove
one retriever is better, only that neither regresses. This harness adds the
discriminating power that was missing:

  * a CONFUSABLE fixture corpus (tests/eval/fixtures/docs) — three near-identical
    router model cards + two near-identical deployment tiers,
  * CHUNK-LEVEL relevance — each query names a distinctive fact
    (``expected_contains``) that appears only in the correct chunk, so the metric
    rewards retrieving the right passage, not just the right document,
  * QUERY / PASSAGE prompts — retrievers can be run symmetric or with the
    E5-style ``query: `` / ``document: `` instruction prompts the model documents.

``compare_retrievers()`` runs several retriever configurations over the same
corpus + queries and returns their chunk-level metrics so the MiniLM-vs-LFM
margin can be recorded. Reused by both the pytest gate and a CLI A/B runner.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from lib.rag import RAGConfig, RAGPipeline
from lib.rag.embedder import SentenceTransformerEmbedder
from lib.rag.parser.composite_parser import CompositeParser
from lib.rag.types import RetrievalResult

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "docs"
HARDENED_DATASET = Path(__file__).parent / "rag_eval_hardened.yaml"

# LFM2.5-Embedding documents these asymmetric prompts in its ST config.
LFM_QUERY_PROMPT = "query: "
LFM_DOCUMENT_PROMPT = "document: "


# ─── Data types ──────────────────────────────────────────────────────────


@dataclass
class HardenedQuery:
    """A query with both doc-level and chunk-level expectations."""

    query: str
    expected_documents: list[str]
    expected_contains: Optional[str]
    relevance: str


@dataclass
class ChunkEvalReport:
    """Chunk-level aggregate metrics for one retriever configuration."""

    label: str
    positive_queries: int = 0
    # Doc-level (top result document matches)
    doc_hit_at_1: float = 0.0
    doc_hit_at_3: float = 0.0
    # Chunk-level (top result document matches AND contains the labeled fact)
    chunk_hit_at_1: float = 0.0
    chunk_hit_at_3: float = 0.0
    chunk_mrr: float = 0.0
    negative_max_confidence: float = 0.0
    per_query: list[dict[str, object]] = field(default_factory=list)


# ─── Loading ─────────────────────────────────────────────────────────────


def load_hardened_dataset(path: Path = HARDENED_DATASET) -> list[HardenedQuery]:
    """Load hardened eval queries from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        HardenedQuery(
            query=q["query"],
            expected_documents=q.get("expected_documents", []),
            expected_contains=q.get("expected_contains"),
            relevance=q.get("relevance", "none"),
        )
        for q in data["queries"]
    ]


# ─── Retriever configurations ────────────────────────────────────────────


@dataclass
class RetrieverSpec:
    """A retriever configuration to evaluate."""

    label: str
    model_name: str
    dimension: int
    query_prompt: str = ""
    document_prompt: str = ""

    def build_embedder(self) -> SentenceTransformerEmbedder:
        return SentenceTransformerEmbedder(
            model_name=self.model_name,
            dimension=self.dimension,
            query_prompt=self.query_prompt,
            document_prompt=self.document_prompt,
        )


def default_specs() -> list[RetrieverSpec]:
    """The three configurations the F-509 margin is reported over."""
    return [
        RetrieverSpec("all-MiniLM-L6-v2", "all-MiniLM-L6-v2", 384),
        RetrieverSpec("LFM2.5-Embedding (symmetric)", "LFM2.5-Embedding-350M", 1024),
        RetrieverSpec(
            "LFM2.5-Embedding (query/passage prompts)",
            "LFM2.5-Embedding-350M",
            1024,
            query_prompt=LFM_QUERY_PROMPT,
            document_prompt=LFM_DOCUMENT_PROMPT,
        ),
    ]


# ─── Core eval ───────────────────────────────────────────────────────────


def run_chunk_eval(
    spec: RetrieverSpec,
    docs_dir: Path = FIXTURES_DIR,
    dataset_path: Path = HARDENED_DATASET,
    db_path: Optional[Path] = None,
) -> ChunkEvalReport:
    """Index the confusable corpus with ``spec`` and score chunk-level retrieval."""
    queries = load_hardened_dataset(dataset_path)
    _db_path = db_path or Path(tempfile.mkdtemp()) / "hardened_eval.db"
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    config = RAGConfig(
        file_types=(".md", ".txt", ".markdown"),
        max_chunk_tokens=120,
        chunk_overlap_tokens=20,
    )
    embedder = spec.build_embedder()
    conn = sqlite3.connect(str(_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    pipeline = RAGPipeline(conn=conn, embedder=embedder, config=config, parser=CompositeParser())
    pipeline.setup()
    pipeline.index([docs_dir])

    report = ChunkEvalReport(label=spec.label)
    doc_h1 = doc_h3 = chunk_h1 = chunk_h3 = chunk_rr = 0.0
    positives = 0
    neg_max = 0.0

    for q in queries:
        results: list[RetrievalResult] = pipeline.retrieve(q.query, top_k=5)
        top_docs = [Path(r.document_path).name for r in results]
        top_texts = [r.chunk_text for r in results]
        top_conf = results[0].score if results else 0.0

        if not q.expected_documents:
            neg_max = max(neg_max, top_conf)
            report.per_query.append(
                {"query": q.query, "negative": True, "top_conf": round(top_conf, 4)}
            )
            continue

        positives += 1
        expected = set(q.expected_documents)

        doc_rank = next((i for i, d in enumerate(top_docs) if d in expected), None)
        # Chunk-level: correct doc AND the distinctive fact present in that chunk.
        chunk_rank = next(
            (
                i
                for i, (d, t) in enumerate(zip(top_docs, top_texts))
                if d in expected and (q.expected_contains is None or q.expected_contains in t)
            ),
            None,
        )

        if doc_rank is not None:
            doc_h1 += 1.0 if doc_rank == 0 else 0.0
            doc_h3 += 1.0 if doc_rank < 3 else 0.0
        if chunk_rank is not None:
            chunk_h1 += 1.0 if chunk_rank == 0 else 0.0
            chunk_h3 += 1.0 if chunk_rank < 3 else 0.0
            chunk_rr += 1.0 / (chunk_rank + 1)

        report.per_query.append(
            {
                "query": q.query,
                "expected": q.expected_documents,
                "top_doc": top_docs[0] if top_docs else "none",
                "doc_rank": doc_rank,
                "chunk_rank": chunk_rank,
            }
        )

    pipeline.close()
    conn.close()

    report.positive_queries = positives
    if positives:
        report.doc_hit_at_1 = doc_h1 / positives
        report.doc_hit_at_3 = doc_h3 / positives
        report.chunk_hit_at_1 = chunk_h1 / positives
        report.chunk_hit_at_3 = chunk_h3 / positives
        report.chunk_mrr = chunk_rr / positives
    report.negative_max_confidence = neg_max
    return report


def compare_retrievers(
    specs: Optional[list[RetrieverSpec]] = None,
) -> list[ChunkEvalReport]:
    """Run several retriever configs over the confusable corpus."""
    return [run_chunk_eval(s) for s in (specs or default_specs())]


def format_comparison(reports: list[ChunkEvalReport]) -> str:
    """Human-readable A/B table for STATUS / logs."""
    lines = [
        f"{'retriever':<44} {'doc@1':>6} {'chk@1':>6} {'chk@3':>6} " f"{'chkMRR':>7} {'neg':>6}"
    ]
    for r in reports:
        lines.append(
            f"{r.label:<44} {r.doc_hit_at_1:>6.1%} {r.chunk_hit_at_1:>6.1%} "
            f"{r.chunk_hit_at_3:>6.1%} {r.chunk_mrr:>7.3f} "
            f"{r.negative_max_confidence:>6.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    reps = compare_retrievers()
    print(format_comparison(reps))
