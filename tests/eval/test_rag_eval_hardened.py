"""Tests for the hardened chunk-level RAG eval (F-509).

Fast tests exercise the dataset/spec structure with no model load. The @slow
gate indexes the confusable corpus with the real Liquid retriever and asserts:
  * doc-level does NOT regress (top doc is correct),
  * the chunk-level metric is strictly harder than doc-level on this corpus —
    proving the eval gained the discriminating power the doc-level harness lacked,
  * no-match queries stay separated from positives.

Run the gate:  pytest tests/eval/test_rag_eval_hardened.py -v -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval.hardened_eval import (
    FIXTURES_DIR,
    LFM_DOCUMENT_PROMPT,
    LFM_QUERY_PROMPT,
    RetrieverSpec,
    default_specs,
    format_comparison,
    load_hardened_dataset,
    run_chunk_eval,
)

# ─── Fast structural tests (no model load) ───────────────────────────────


class TestHardenedDataset:
    def test_dataset_loads(self) -> None:
        queries = load_hardened_dataset()
        assert len(queries) >= 15
        positives = [q for q in queries if q.expected_documents]
        negatives = [q for q in queries if not q.expected_documents]
        assert positives and negatives

    def test_positives_carry_chunk_labels(self) -> None:
        """Every positive query names a distinctive chunk-level fact."""
        for q in load_hardened_dataset():
            if q.expected_documents:
                assert q.expected_contains, f"missing expected_contains: {q.query}"

    def test_expected_docs_exist_in_corpus(self) -> None:
        """Referenced fixture docs must exist on disk."""
        present = {p.name for p in FIXTURES_DIR.glob("*.md")}
        for q in load_hardened_dataset():
            for doc in q.expected_documents:
                assert doc in present, f"fixture missing: {doc}"

    def test_expected_facts_present_in_fixtures(self) -> None:
        """Each expected_contains fact really appears in its target document."""
        for q in load_hardened_dataset():
            if not (q.expected_documents and q.expected_contains):
                continue
            texts = [
                (FIXTURES_DIR / d).read_text()
                for d in q.expected_documents
                if (FIXTURES_DIR / d).exists()
            ]
            assert any(
                q.expected_contains in t for t in texts
            ), f"fact not in fixtures: {q.expected_contains!r}"

    def test_default_specs_shape(self) -> None:
        specs = default_specs()
        labels = [s.label for s in specs]
        assert any("MiniLM" in x for x in labels)
        assert any("symmetric" in x for x in labels)
        assert any("prompt" in x for x in labels)

    def test_format_comparison_smoke(self) -> None:
        from tests.eval.hardened_eval import ChunkEvalReport

        rep = ChunkEvalReport(label="x", doc_hit_at_1=1.0, chunk_hit_at_1=0.8)
        out = format_comparison([rep])
        assert "retriever" in out and "chk@1" in out


# ─── Slow gate (real Liquid retriever over the confusable corpus) ────────


@pytest.mark.slow
class TestHardenedGate:
    @pytest.fixture(scope="class")
    def lfm_report(self, tmp_path_factory: pytest.TempPathFactory):
        db = tmp_path_factory.mktemp("hardened") / "h.db"
        spec = RetrieverSpec(
            "LFM2.5-Embedding (query/passage prompts)",
            "LFM2.5-Embedding-350M",
            1024,
            query_prompt=LFM_QUERY_PROMPT,
            document_prompt=LFM_DOCUMENT_PROMPT,
        )
        return run_chunk_eval(spec, db_path=Path(db))

    def test_doc_level_no_regression(self, lfm_report) -> None:
        """Doc-level Hit@1 stays saturated — the swap does not regress retrieval."""
        assert lfm_report.doc_hit_at_1 >= 0.94

    def test_chunk_hit_at_3_strong(self, lfm_report) -> None:
        """Correct passage is in the top 3 for the large majority of queries."""
        assert lfm_report.chunk_hit_at_3 >= 0.85

    def test_eval_has_discriminating_power(self, lfm_report) -> None:
        """The chunk-level metric is strictly harder than doc-level on this
        confusable corpus — this is the discrimination the doc-level harness
        lacked (where MiniLM and LFM tied to three decimals)."""
        assert lfm_report.chunk_hit_at_1 < lfm_report.doc_hit_at_1

    def test_negatives_separated(self, lfm_report) -> None:
        """No-match queries stay below the production confidence floor (0.35)."""
        assert lfm_report.negative_max_confidence < 0.35
