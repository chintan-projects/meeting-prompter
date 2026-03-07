"""RAG retrieval evaluation harness.

Runs queries from rag_eval_dataset.yaml against the real RAG pipeline
with real embeddings and real context documents. Reports Hit@1, Hit@3,
MRR, and confidence distribution.

Marked @pytest.mark.slow — skipped in normal `pytest` runs.
Run with: pytest tests/eval/test_rag_eval.py -v -m slow
"""
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import yaml

from lib.rag import RAGConfig, RAGPipeline
from lib.rag.embedder import SentenceTransformerEmbedder
from lib.rag.parser.composite_parser import CompositeParser
from lib.rag.types import RetrievalResult

DATASET_PATH = Path(__file__).parent / "rag_eval_dataset.yaml"
DOCS_DIR = Path(__file__).parents[2] / "context"

# ─── Data types ──────────────────────────────────────────────────────────


@dataclass
class EvalQuery:
    """A single evaluation query with expected results."""

    query: str
    expected_documents: list[str]
    relevance: str  # high, medium, none


@dataclass
class EvalResult:
    """Metrics for a single query."""

    query: str
    expected_docs: list[str]
    top_k_docs: list[str]
    top_k_scores: list[float]
    top_k_lexical: list[float]
    top_k_semantic: list[float]
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    top_confidence: float


@dataclass
class EvalReport:
    """Aggregated evaluation metrics."""

    results: list[EvalResult] = field(default_factory=list)
    total_queries: int = 0
    positive_queries: int = 0  # queries with expected docs
    negative_queries: int = 0  # queries with no expected docs

    # Aggregate metrics (computed for positive queries only)
    mean_hit_at_1: float = 0.0
    mean_hit_at_3: float = 0.0
    mean_mrr: float = 0.0

    # Confidence distribution (all queries)
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    confidence_mean: float = 0.0
    confidence_stdev: float = 0.0
    confidence_histogram: dict[str, int] = field(default_factory=dict)

    # Negative query metrics
    negative_max_confidence: float = 0.0


# ─── Core eval function ──────────────────────────────────────────────────


def load_dataset(path: Path) -> list[EvalQuery]:
    """Load eval queries from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        EvalQuery(
            query=q["query"],
            expected_documents=q.get("expected_documents", []),
            relevance=q.get("relevance", "none"),
        )
        for q in data["queries"]
    ]


def run_eval(
    docs_dir: Path,
    dataset_path: Path,
    config: Optional[RAGConfig] = None,
    db_path: Optional[Path] = None,
) -> EvalReport:
    """Run the full RAG evaluation suite.

    Args:
        docs_dir: Directory with source documents.
        dataset_path: Path to rag_eval_dataset.yaml.
        config: Optional RAGConfig overrides for A/B testing.
        db_path: SQLite DB path (defaults to temp).

    Returns:
        EvalReport with per-query and aggregate metrics.
    """
    import sqlite3
    import tempfile

    queries = load_dataset(dataset_path)
    _db_path = db_path or Path(tempfile.mkdtemp()) / "eval_rag.db"
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    rag_config = config or RAGConfig(
        file_types=(".pdf", ".md", ".txt", ".markdown"),
        max_chunk_tokens=400,
        chunk_overlap_tokens=50,
    )

    embedder = SentenceTransformerEmbedder()
    conn = sqlite3.connect(str(_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    pipeline = RAGPipeline(
        conn=conn, embedder=embedder, config=rag_config, parser=CompositeParser()
    )
    pipeline.setup()
    result = pipeline.index([docs_dir])
    print(
        f"\nIndexed: {result.documents_indexed} docs, "
        f"{result.chunks_created} chunks, "
        f"{result.documents_skipped} skipped"
    )
    if result.errors:
        for err in result.errors:
            print(f"  Error: {err}")

    report = EvalReport()
    report.total_queries = len(queries)

    for eq in queries:
        results: list[RetrievalResult] = pipeline.retrieve(eq.query, top_k=5)

        top_k_docs = [Path(r.document_path).name for r in results]
        top_k_scores = [r.score for r in results]
        top_k_lexical = [r.lexical_score for r in results]
        top_k_semantic = [r.semantic_score for r in results]

        hit_at_1 = False
        hit_at_3 = False
        rr = 0.0
        top_confidence = top_k_scores[0] if top_k_scores else 0.0

        if eq.expected_documents:
            report.positive_queries += 1
            expected_set = set(eq.expected_documents)

            for i, doc in enumerate(top_k_docs):
                if doc in expected_set:
                    if i == 0:
                        hit_at_1 = True
                    if i < 3:
                        hit_at_3 = True
                    if rr == 0.0:
                        rr = 1.0 / (i + 1)
                    break
        else:
            report.negative_queries += 1

        report.results.append(
            EvalResult(
                query=eq.query,
                expected_docs=eq.expected_documents,
                top_k_docs=top_k_docs,
                top_k_scores=top_k_scores,
                top_k_lexical=top_k_lexical,
                top_k_semantic=top_k_semantic,
                hit_at_1=hit_at_1,
                hit_at_3=hit_at_3,
                reciprocal_rank=rr,
                top_confidence=top_confidence,
            )
        )

    pipeline.close()
    conn.close()

    _compute_aggregates(report)
    return report


def _compute_aggregates(report: EvalReport) -> None:
    """Fill in aggregate metrics on the report."""
    positive = [r for r in report.results if r.expected_docs]
    if positive:
        report.mean_hit_at_1 = sum(1 for r in positive if r.hit_at_1) / len(positive)
        report.mean_hit_at_3 = sum(1 for r in positive if r.hit_at_3) / len(positive)
        report.mean_mrr = sum(r.reciprocal_rank for r in positive) / len(positive)

    all_scores = [r.top_confidence for r in report.results if r.top_confidence > 0]
    if all_scores:
        report.confidence_min = min(all_scores)
        report.confidence_max = max(all_scores)
        report.confidence_mean = statistics.mean(all_scores)
        report.confidence_stdev = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    # Histogram: 10 buckets from 0.0-1.0
    buckets = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for s in all_scores:
        bucket_idx = min(int(s * 10), 9)
        key = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"
        buckets[key] += 1
    report.confidence_histogram = buckets

    # Negative query analysis
    negatives = [r for r in report.results if not r.expected_docs]
    if negatives:
        report.negative_max_confidence = max(r.top_confidence for r in negatives)


def print_report(report: EvalReport) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION REPORT")
    print("=" * 70)

    print(f"\nQueries: {report.total_queries} total "
          f"({report.positive_queries} positive, {report.negative_queries} negative)")

    print(f"\n--- Retrieval Quality (positive queries) ---")
    print(f"  Hit@1:  {report.mean_hit_at_1:.1%}")
    print(f"  Hit@3:  {report.mean_hit_at_3:.1%}")
    print(f"  MRR:    {report.mean_mrr:.3f}")

    print(f"\n--- Confidence Distribution ---")
    print(f"  Min:    {report.confidence_min:.4f}")
    print(f"  Max:    {report.confidence_max:.4f}")
    print(f"  Mean:   {report.confidence_mean:.4f}")
    print(f"  Stdev:  {report.confidence_stdev:.4f}")

    print(f"\n  Histogram:")
    for bucket, count in report.confidence_histogram.items():
        bar = "#" * count
        print(f"    {bucket}: {bar} ({count})")

    if report.negative_max_confidence > 0:
        print(f"\n--- Negative Query Analysis ---")
        print(f"  Max confidence on no-match query: {report.negative_max_confidence:.4f}")

    print(f"\n--- Per-Query Results ---")
    for r in report.results:
        status = "HIT" if r.hit_at_1 else ("hit@3" if r.hit_at_3 else "MISS")
        if not r.expected_docs:
            status = f"NEG({r.top_confidence:.3f})"
        top_doc = r.top_k_docs[0] if r.top_k_docs else "none"
        print(
            f"  [{status:>8}] score={r.top_confidence:.4f} "
            f"lex={r.top_k_lexical[0]:.3f} sem={r.top_k_semantic[0]:.3f} "
            f"doc={top_doc:30s} q={r.query[:50]}"
            if r.top_k_scores
            else f"  [{status:>8}] score=0.0000 doc=none q={r.query[:50]}"
        )

    print("=" * 70)


# ─── Pytest tests (marked slow) ─────────────────────────────────────────


@pytest.mark.slow
class TestRAGEval:
    """RAG evaluation tests — require real embedder and real docs."""

    @pytest.fixture(scope="class")
    def eval_report(self, tmp_path_factory: pytest.TempPathFactory) -> EvalReport:
        """Run eval once per test class, reuse results."""
        db_path = tmp_path_factory.mktemp("rag_eval") / "eval.db"
        report = run_eval(DOCS_DIR, DATASET_PATH, db_path=db_path)
        print_report(report)
        return report

    def test_hit_at_1_above_threshold(self, eval_report: EvalReport) -> None:
        """Top-1 result should match expected doc >= 60% of the time."""
        assert eval_report.mean_hit_at_1 >= 0.60, (
            f"Hit@1 = {eval_report.mean_hit_at_1:.1%}, expected >= 60%"
        )

    def test_hit_at_3_above_threshold(self, eval_report: EvalReport) -> None:
        """Top-3 results should contain expected doc >= 80% of the time."""
        assert eval_report.mean_hit_at_3 >= 0.80, (
            f"Hit@3 = {eval_report.mean_hit_at_3:.1%}, expected >= 80%"
        )

    def test_mrr_above_threshold(self, eval_report: EvalReport) -> None:
        """Mean reciprocal rank should be >= 0.60."""
        assert eval_report.mean_mrr >= 0.60, (
            f"MRR = {eval_report.mean_mrr:.3f}, expected >= 0.60"
        )

    def test_confidence_has_discrimination(self, eval_report: EvalReport) -> None:
        """Scores should not all cluster together (stdev > 0.02)."""
        assert eval_report.confidence_stdev > 0.02, (
            f"Confidence stdev = {eval_report.confidence_stdev:.4f}, "
            f"scores cluster too tightly for threshold tuning"
        )

    def test_negative_queries_score_lower(self, eval_report: EvalReport) -> None:
        """No-match queries should have lower confidence than positive matches."""
        positive_scores = [
            r.top_confidence for r in eval_report.results
            if r.expected_docs and r.top_confidence > 0
        ]
        if positive_scores and eval_report.negative_max_confidence > 0:
            positive_mean = statistics.mean(positive_scores)
            assert eval_report.negative_max_confidence < positive_mean, (
                f"Negative max ({eval_report.negative_max_confidence:.3f}) >= "
                f"positive mean ({positive_mean:.3f})"
            )


# ─── CLI runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    db_path = Path(tempfile.mkdtemp()) / "eval_rag.db"
    docs = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS_DIR
    report = run_eval(docs, DATASET_PATH, db_path=db_path)
    print_report(report)
