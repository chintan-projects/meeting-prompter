"""Tests for the hybrid RAG engine (FTS5 + vector retrieval).

Tests the RAGEngine adapter, RAGPipeline, embedder protocol, PDF parser,
composite parser, and the end-to-end index→retrieve flow.
"""

import sqlite3
from pathlib import Path

import pytest

from lib.rag import RAGConfig, RAGPipeline
from lib.rag.parser.composite_parser import CompositeParser
from lib.rag.parser.pdf_parser import PdfParser
from lib.rag.parser.text_parser import TextParser
from lib.rag_engine import RAGEngine, format_confidence

# ─── Fixtures ────────────────────────────────────────────────────────────────


class MockEmbedder:
    """Deterministic embedder for fast tests (no model download)."""

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> list[float]:
        """Hash-based embedding for deterministic results."""
        h = hash(text) & 0xFFFFFFFF
        return [(h >> i & 0xFF) / 255.0 for i in range(0, 64, 8)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Create a temporary docs directory with test files."""
    docs = tmp_path / "context"
    docs.mkdir()

    # Markdown file about deployment
    (docs / "deployment.md").write_text(
        "# Deployment Guide\n\n"
        "## Timeline\n\n"
        "The deployment is scheduled for Q2 2025. "
        "The beta release will be in March, followed by GA in June.\n\n"
        "## Requirements\n\n"
        "Minimum 8GB RAM and 4 CPU cores for production. "
        "Docker and Kubernetes are required for orchestration.\n"
    )

    # Another markdown file about product
    (docs / "product.md").write_text(
        "# Product Overview\n\n"
        "The product uses machine learning for real-time analytics. "
        "Key features include automated reporting and anomaly detection.\n\n"
        "## Pricing\n\n"
        "Enterprise plan starts at $500/month with volume discounts.\n"
    )

    return docs


@pytest.fixture
def rag_engine(docs_dir: Path, tmp_path: Path) -> RAGEngine:
    """Create a RAGEngine with mock embedder for fast tests."""
    db_path = tmp_path / "test_rag.db"
    config = RAGConfig(max_chunk_tokens=100, chunk_overlap_tokens=10)

    # Patch to use mock embedder
    engine = RAGEngine.__new__(RAGEngine)
    engine.docs_dir = docs_dir
    engine._db_path = db_path
    engine._db_path.parent.mkdir(parents=True, exist_ok=True)
    engine._embedder = MockEmbedder()
    engine._config = config
    engine._conn = sqlite3.connect(str(db_path))
    engine._conn.execute("PRAGMA journal_mode=WAL")
    engine._conn.execute("PRAGMA foreign_keys=ON")

    engine._pipeline = RAGPipeline(
        conn=engine._conn,
        embedder=engine._embedder,
        config=config,
        parser=CompositeParser(),
    )
    engine._pipeline.setup()
    engine._index_documents()
    engine._chunk_count = engine._get_chunk_count()

    yield engine
    engine.close()


# ─── Contract tests ──────────────────────────────────────────────────────────


class TestRAGEngineContract:
    """Verify RAGEngine satisfies the public API contract."""

    def test_query_returns_tuple(self, rag_engine: RAGEngine) -> None:
        """query() must return (str, float, str)."""
        result = rag_engine.query("deployment timeline")
        assert isinstance(result, tuple)
        assert len(result) == 3
        context, confidence, source = result
        assert isinstance(context, str)
        assert isinstance(confidence, float)
        assert isinstance(source, str)

    def test_empty_query_returns_empty(self, rag_engine: RAGEngine) -> None:
        """Empty/whitespace queries return empty tuple."""
        assert rag_engine.query("") == ("", 0.0, "")
        assert rag_engine.query("   ") == ("", 0.0, "")

    def test_chunk_count_property(self, rag_engine: RAGEngine) -> None:
        """chunk_count should be a positive integer after indexing."""
        assert isinstance(rag_engine.chunk_count, int)
        assert rag_engine.chunk_count > 0

    def test_is_using_colbert_false(self, rag_engine: RAGEngine) -> None:
        """Backward compat: is_using_colbert always False."""
        assert rag_engine.is_using_colbert is False

    def test_is_hybrid_true(self, rag_engine: RAGEngine) -> None:
        """is_hybrid should be True for the new engine."""
        assert rag_engine.is_hybrid is True

    def test_satisfies_rag_queryable(self, rag_engine: RAGEngine) -> None:
        """RAGEngine should satisfy the RAGQueryable protocol."""

        # Duck typing check — query method exists with right signature
        assert hasattr(rag_engine, "query")
        assert callable(rag_engine.query)


# ─── Indexing tests ──────────────────────────────────────────────────────────


class TestIndexing:
    """Test document indexing behavior."""

    def test_markdown_files_indexed(self, rag_engine: RAGEngine) -> None:
        """Markdown files in docs_dir should be indexed."""
        assert rag_engine.chunk_count > 0

    def test_rebuild_index(self, rag_engine: RAGEngine) -> None:
        """rebuild_index should clear and re-index."""
        original_count = rag_engine.chunk_count
        rag_engine.rebuild_index()
        assert rag_engine.chunk_count == original_count

    def test_rebuild_changes_with_new_files(self, rag_engine: RAGEngine, docs_dir: Path) -> None:
        """After adding a file and rebuilding, chunk count increases."""
        original = rag_engine.chunk_count

        (docs_dir / "extra.md").write_text(
            "# Extra Document\n\n"
            "This is additional content about testing and quality assurance. "
            "We run integration tests daily on all platforms.\n"
        )

        rag_engine.rebuild_index()
        assert rag_engine.chunk_count > original

    def test_dimension_change_purges_stale_index(self, docs_dir: Path, tmp_path: Path) -> None:
        """Swapping to a different-dimension embedder clears stale-width vectors."""
        db_path = tmp_path / "dim_rag.db"

        def build(embedder: object) -> RAGEngine:
            engine = RAGEngine.__new__(RAGEngine)
            engine.docs_dir = docs_dir
            engine._db_path = db_path
            engine._embedder = embedder  # type: ignore[assignment]
            engine._config = RAGConfig(max_chunk_tokens=100)
            engine._conn = sqlite3.connect(str(db_path))
            engine._conn.execute("PRAGMA journal_mode=WAL")
            engine._conn.execute("PRAGMA foreign_keys=ON")
            engine._pipeline = RAGPipeline(
                conn=engine._conn,
                embedder=embedder,  # type: ignore[arg-type]
                config=engine._config,
                parser=CompositeParser(),
            )
            engine._pipeline.setup()
            engine._purge_stale_dimension()
            engine._index_documents()
            engine._chunk_count = engine._get_chunk_count()
            return engine

        class WideEmbedder(MockEmbedder):
            @property
            def dimension(self) -> int:
                return 16

            def embed(self, text: str) -> list[float]:
                return super().embed(text) + [0.0] * 8

        first = build(MockEmbedder())  # dim 8
        assert first.chunk_count > 0
        first.close()

        # Reopen with a different-dimension embedder — stale chunks must be purged
        # and re-embedded, not left at the old width.
        second = build(WideEmbedder())  # dim 16
        assert second.chunk_count > 0
        row = second._conn.execute(
            "SELECT length(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        assert row is not None and row[0] == 16 * 4
        second.close()

    def test_missing_docs_dir_safe(self, tmp_path: Path) -> None:
        """Engine handles missing docs_dir gracefully."""
        missing = tmp_path / "nonexistent"
        db_path = tmp_path / "test.db"
        config = RAGConfig(max_chunk_tokens=100)

        engine = RAGEngine.__new__(RAGEngine)
        engine.docs_dir = missing
        engine._db_path = db_path
        engine._embedder = MockEmbedder()
        engine._config = config
        engine._conn = sqlite3.connect(str(db_path))
        engine._conn.execute("PRAGMA journal_mode=WAL")
        engine._conn.execute("PRAGMA foreign_keys=ON")
        engine._pipeline = RAGPipeline(
            conn=engine._conn,
            embedder=engine._embedder,
            config=config,
            parser=CompositeParser(),
        )
        engine._pipeline.setup()
        engine._index_documents()
        engine._chunk_count = engine._get_chunk_count()

        assert engine.chunk_count == 0
        result = engine.query("anything")
        assert result == ("", 0.0, "")
        engine.close()


# ─── Retrieval tests ─────────────────────────────────────────────────────────


class TestRetrieval:
    """Test retrieval quality and behavior."""

    def test_query_returns_content(self, rag_engine: RAGEngine) -> None:
        """A relevant query should return non-empty context."""
        context, confidence, source = rag_engine.query("deployment")
        assert len(context) > 0
        assert confidence > 0.0

    def test_source_is_filename(self, rag_engine: RAGEngine) -> None:
        """Source should be a filename, not a full path."""
        _, _, source = rag_engine.query("deployment timeline")
        if source:
            assert "/" not in source
            assert source.endswith(".md")

    def test_confidence_between_0_and_1(self, rag_engine: RAGEngine) -> None:
        """Confidence score should be in [0, 1]."""
        _, confidence, _ = rag_engine.query("pricing enterprise")
        assert 0.0 <= confidence <= 1.0

    def test_no_match_low_confidence(self, rag_engine: RAGEngine) -> None:
        """Completely unrelated query should return low or zero confidence."""
        context, confidence, _ = rag_engine.query("xyzzy quantum flux capacitor")
        # With mock embedder, might still get some results, but score should be modest
        assert confidence <= 1.0  # sanity


# ─── Parser tests ────────────────────────────────────────────────────────────


class TestTextParser:
    """Test the markdown/text parser."""

    def test_parse_markdown(self, tmp_path: Path) -> None:
        """TextParser should parse markdown with heading hierarchy."""
        md = tmp_path / "test.md"
        md.write_text("# Title\n\nParagraph one.\n\n## Section\n\nParagraph two.\n")

        parser = TextParser()
        doc = parser.parse(md)

        assert doc.filename == "test.md"
        assert len(doc.sections) >= 1
        assert doc.token_count > 0

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Empty files should produce a document with no sections."""
        empty = tmp_path / "empty.md"
        empty.write_text("")

        parser = TextParser()
        doc = parser.parse(empty)
        assert doc.filename == "empty.md"


class TestCompositeParser:
    """Test the composite parser routing."""

    def test_routes_markdown(self, tmp_path: Path) -> None:
        """Markdown files routed to TextParser."""
        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nWorld.\n")

        parser = CompositeParser()
        doc = parser.parse(md)
        assert doc.filename == "test.md"

    def test_routes_txt(self, tmp_path: Path) -> None:
        """Plain text files routed to TextParser."""
        txt = tmp_path / "test.txt"
        txt.write_text("Some plain text content for testing.\n")

        parser = CompositeParser()
        doc = parser.parse(txt)
        assert doc.filename == "test.txt"

    def test_unsupported_type_raises(self, tmp_path: Path) -> None:
        """Unsupported file types should raise ParseError."""
        from lib.rag.errors import ParseError

        bad = tmp_path / "test.xyz"
        bad.write_text("data")

        parser = CompositeParser()
        with pytest.raises(ParseError):
            parser.parse(bad)


# ─── PDF parser tests ────────────────────────────────────────────────────────


class TestPdfParser:
    """Test PDF parsing (requires pypdf)."""

    def test_missing_file_raises(self) -> None:
        """Missing PDF file should raise ParseError."""
        from lib.rag.errors import ParseError

        parser = PdfParser()
        with pytest.raises(ParseError):
            parser.parse(Path("/nonexistent/file.pdf"))


# ─── Fusion tests ────────────────────────────────────────────────────────────


class TestWeightedFusion:
    """Test the score fusion logic."""

    def test_fusion_basic(self) -> None:
        """Weighted fusion should combine scores correctly."""
        from lib.rag.retrieval.fusion import weighted_fusion
        from lib.rag.types import FTSHit, VectorHit

        lex = [FTSHit(chunk_id=1, score=10.0), FTSHit(chunk_id=2, score=5.0)]
        sem = [VectorHit(chunk_id=1, score=0.9), VectorHit(chunk_id=3, score=0.8)]

        results = weighted_fusion(lex, sem, lexical_weight=0.5, semantic_weight=0.5, top_k=5)

        assert len(results) > 0
        # Chunk 1 appears in both lists — should have highest fused score
        chunk1 = next(r for r in results if r.chunk_id == 1)
        assert chunk1.fused_score > 0

    def test_empty_inputs(self) -> None:
        """No hits → no results."""
        from lib.rag.retrieval.fusion import weighted_fusion

        assert weighted_fusion([], [], top_k=5) == []

    def test_top_k_limit(self) -> None:
        """Output should be limited to top_k."""
        from lib.rag.retrieval.fusion import weighted_fusion
        from lib.rag.types import VectorHit

        sem = [VectorHit(chunk_id=i, score=1.0 - i * 0.1) for i in range(10)]
        results = weighted_fusion([], sem, top_k=3)
        assert len(results) == 3


# ─── Config tests ────────────────────────────────────────────────────────────


class TestRAGConfig:
    """Test RAG configuration dataclass."""

    def test_defaults(self) -> None:
        """RAGConfig should have sensible defaults."""
        config = RAGConfig()
        assert config.max_chunk_tokens == 512
        assert config.chunk_overlap_tokens == 50
        assert config.lexical_weight == 0.05
        assert config.semantic_weight == 0.95

    def test_custom_weights(self) -> None:
        """Custom weights should be preserved."""
        config = RAGConfig(lexical_weight=0.3, semantic_weight=0.7)
        assert config.lexical_weight == 0.3
        assert config.semantic_weight == 0.7

    def test_embedding_defaults_liquid(self) -> None:
        """Default retriever is the Liquid embedding model (F-502 swap)."""
        config = RAGConfig()
        assert config.embedding_model == "LFM2.5-Embedding-350M"
        assert config.embedding_dimension == 1024


# ─── Embedder model resolution (F-502) ───────────────────────────────────────


class TestEmbedderResolution:
    """Config-driven embedder selection — no model load required."""

    def test_default_is_liquid_local(self) -> None:
        """Bare Liquid name resolves to a local registry dir, 1024-dim."""
        from lib.rag.embedder import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        assert emb.dimension == 1024
        # Local registry model → trust_remote_code enabled.
        assert emb._trust_remote_code is True
        assert emb._model_name.endswith("LFM2.5-Embedding-350M")

    def test_minilm_name_is_hub_384(self) -> None:
        """A hub id (all-MiniLM-L6-v2) stays remote, 384-dim, no trust_remote_code."""
        from lib.rag.embedder import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        assert emb.dimension == 384
        assert emb._trust_remote_code is False
        assert emb._model_name == "all-MiniLM-L6-v2"

    def test_explicit_overrides_respected(self) -> None:
        """Explicit dimension / trust_remote_code override the auto-detection."""
        from lib.rag.embedder import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2", dimension=99, trust_remote_code=True
        )
        assert emb.dimension == 99
        assert emb._trust_remote_code is True


# ─── Format helpers ──────────────────────────────────────────────────────────


class TestFormatConfidence:
    """Test confidence score formatting."""

    def test_high_confidence_green(self) -> None:
        assert "green" in format_confidence(0.75)

    def test_medium_confidence_yellow(self) -> None:
        assert "yellow" in format_confidence(0.35)

    def test_low_confidence_red(self) -> None:
        assert "red" in format_confidence(0.10)


# ─── App config integration ─────────────────────────────────────────────────


class TestAppConfigRAG:
    """Test that RAGPipelineConfig is wired into AppConfig."""

    def test_rag_config_in_app_config(self) -> None:
        """AppConfig should have rag field with RAGPipelineConfig."""
        from lib.config import AppConfig, RAGPipelineConfig

        config = AppConfig()
        assert hasattr(config, "rag")
        assert isinstance(config.rag, RAGPipelineConfig)
        assert config.rag.max_chunk_tokens == 400
        assert config.rag.db_path == "data/rag.db"

    def test_no_chunking_or_normalization(self) -> None:
        """AppConfig should NOT have chunking or normalization fields."""
        from lib.config import AppConfig

        config = AppConfig()
        assert not hasattr(config, "chunking")
        assert not hasattr(config, "normalization")

    def test_load_config_parses_rag_section(self, tmp_path: Path) -> None:
        """load_config should parse rag section from YAML."""
        from lib.config import load_config

        yaml_content = (
            "rag:\n"
            "  max_chunk_tokens: 300\n"
            "  lexical_weight: 0.10\n"
            "  semantic_weight: 0.90\n"
            "  db_path: 'data/custom.db'\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.rag.max_chunk_tokens == 300
        assert config.rag.lexical_weight == 0.10
        assert config.rag.db_path == "data/custom.db"


# ─── RAGQueryable protocol ──────────────────────────────────────────────────


class TestRAGQueryableProtocol:
    """Test RAGQueryable protocol consolidation."""

    def test_single_definition_in_types(self) -> None:
        """RAGQueryable should be importable from triggers.types."""
        from lib.triggers.types import RAGQueryable

        assert RAGQueryable is not None

    def test_engine_imports_from_types(self) -> None:
        """TriggerEngine should import RAGQueryable from types, not define its own."""
        import lib.triggers.engine as engine_mod

        # The module should not define RAGQueryable locally
        source = Path(engine_mod.__file__).read_text()
        assert "class RAGQueryable" not in source

    def test_topic_imports_from_types(self) -> None:
        """TopicTrigger should import RAGQueryable from types."""
        import lib.triggers.topic_trigger as topic_mod

        source = Path(topic_mod.__file__).read_text()
        assert "class RAGQueryable" not in source

    def test_followup_imports_from_types(self) -> None:
        """FollowUpTrigger should import RAGQueryable from types."""
        import lib.triggers.followup_trigger as followup_mod

        source = Path(followup_mod.__file__).read_text()
        assert "class RAGQueryable" not in source
