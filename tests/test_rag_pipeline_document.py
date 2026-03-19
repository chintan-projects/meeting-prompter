"""Tests for RAGPipeline.index_document() — indexing pre-parsed documents."""

import sqlite3

import pytest

from lib.rag import RAGPipeline, RAGConfig
from lib.rag.types import ParsedDocument, ParsedSection

# ── Mock embedder ────────────────────────────────────────────────────────────

EMBED_DIM = 8


class _MockEmbedder:
    """Fixed-dimension embedder that returns deterministic vectors."""

    @property
    def dimension(self) -> int:
        return EMBED_DIM

    def embed(self, text: str) -> list[float]:
        return [0.1] * EMBED_DIM

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * EMBED_DIM for _ in texts]


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def pipeline() -> RAGPipeline:
    """In-memory RAGPipeline with mock embedder, temperature=0.0 conceptually."""
    conn = sqlite3.connect(":memory:")
    config = RAGConfig(max_chunk_tokens=100, chunk_overlap_tokens=10)
    pipe = RAGPipeline(conn=conn, embedder=_MockEmbedder(), config=config)
    pipe.setup()
    return pipe


def _make_doc(
    page_id: str = "abc123",
    title: str = "Test Doc",
    content: str = "Some content about deployment timelines and Q2 roadmap.",
) -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    full_text = f"# {title}\n\n{content}"
    return ParsedDocument(
        path=f"notion://{page_id}",
        filename=title,
        mime_type="text/notion",
        sections=[
            ParsedSection(
                heading=title,
                heading_level=1,
                heading_path=title,
                content=full_text,
            ),
        ],
        full_text=full_text,
        token_count=max(1, int(len(full_text.split()) * 1.3)),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestIndexNewDocument:
    """Indexing a brand-new document."""

    def test_creates_chunks_in_db(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc()
        result = pipeline.index_document(doc)
        assert result.documents_indexed == 1
        assert result.chunks_created > 0
        assert result.sections_created == 1

    def test_document_stored_in_db(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc(page_id="stored-check")
        pipeline.index_document(doc)
        row = pipeline._conn.execute(
            "SELECT path, filename FROM documents WHERE path = ?",
            ("notion://stored-check",),
        ).fetchone()
        assert row is not None
        assert row[0] == "notion://stored-check"
        assert row[1] == "Test Doc"

    def test_sections_stored(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc()
        pipeline.index_document(doc)
        sections = pipeline._conn.execute("SELECT heading FROM sections").fetchall()
        assert len(sections) == 1
        assert sections[0][0] == "Test Doc"


class TestHashBasedDedup:
    """Unchanged documents are skipped on re-index."""

    def test_unchanged_document_skipped(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc(page_id="dedup-test")
        result1 = pipeline.index_document(doc)
        assert result1.documents_indexed == 1

        result2 = pipeline.index_document(doc)
        assert result2.documents_skipped == 1
        assert result2.documents_indexed == 0
        assert result2.chunks_created == 0

    def test_changed_content_updates_document(self, pipeline: RAGPipeline) -> None:
        doc1 = _make_doc(page_id="update-test", content="Original content here")
        result1 = pipeline.index_document(doc1)
        assert result1.documents_indexed == 1

        doc2 = _make_doc(page_id="update-test", content="Updated content with new info")
        result2 = pipeline.index_document(doc2)
        assert result2.documents_updated == 1
        assert result2.documents_indexed == 0
        assert result2.chunks_created > 0


class TestMultipleDocuments:
    """Multiple documents indexed independently."""

    def test_different_documents_all_index(self, pipeline: RAGPipeline) -> None:
        docs = [
            _make_doc(page_id="doc-1", title="Doc One", content="First document content"),
            _make_doc(page_id="doc-2", title="Doc Two", content="Second document content"),
            _make_doc(page_id="doc-3", title="Doc Three", content="Third document content"),
        ]
        total_indexed = 0
        total_chunks = 0
        for doc in docs:
            result = pipeline.index_document(doc)
            total_indexed += result.documents_indexed
            total_chunks += result.chunks_created

        assert total_indexed == 3
        assert total_chunks > 0

        doc_count = pipeline._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count == 3


class TestNotionPaths:
    """Synthetic notion:// paths work correctly."""

    def test_notion_path_stored(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc(page_id="notion-path-test")
        pipeline.index_document(doc)
        row = pipeline._conn.execute(
            "SELECT path FROM documents WHERE path LIKE 'notion://%'"
        ).fetchone()
        assert row is not None
        assert row[0] == "notion://notion-path-test"

    def test_notion_path_with_dashes(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc(page_id="deadbeef-1234-5678-abcd")
        result = pipeline.index_document(doc)
        assert result.documents_indexed == 1

    def test_notion_and_file_paths_coexist(self, pipeline: RAGPipeline) -> None:
        """Notion documents and file-based documents don't collide."""
        notion_doc = _make_doc(page_id="notion-1")
        file_doc = ParsedDocument(
            path="/tmp/notes.md",
            filename="notes.md",
            mime_type="text/markdown",
            sections=[
                ParsedSection(
                    heading="Notes",
                    heading_level=1,
                    heading_path="Notes",
                    content="# Notes\n\nSome file content",
                ),
            ],
            full_text="# Notes\n\nSome file content",
            token_count=10,
        )
        r1 = pipeline.index_document(notion_doc)
        r2 = pipeline.index_document(file_doc)
        assert r1.documents_indexed == 1
        assert r2.documents_indexed == 1

        doc_count = pipeline._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count == 2


class TestIndexResultCounts:
    """IndexResult sections and chunk counts are correct."""

    def test_sections_count_matches(self, pipeline: RAGPipeline) -> None:
        doc = ParsedDocument(
            path="notion://multi-section",
            filename="Multi Section",
            mime_type="text/notion",
            sections=[
                ParsedSection(
                    heading="Intro",
                    heading_level=1,
                    heading_path="Intro",
                    content="# Intro\n\nIntroduction content goes here",
                ),
                ParsedSection(
                    heading="Details",
                    heading_level=2,
                    heading_path="Intro > Details",
                    content="## Details\n\nDetailed content goes here",
                ),
                ParsedSection(
                    heading="Conclusion",
                    heading_level=2,
                    heading_path="Intro > Conclusion",
                    content="## Conclusion\n\nClosing remarks go here",
                ),
            ],
            full_text="# Intro\n\nIntro\n\n## Details\n\nDetails\n\n## Conclusion\n\nClosing",
            token_count=20,
        )
        result = pipeline.index_document(doc)
        assert result.sections_created == 3
        assert result.chunks_created >= 3  # at least one chunk per section

    def test_no_errors_on_valid_document(self, pipeline: RAGPipeline) -> None:
        doc = _make_doc()
        result = pipeline.index_document(doc)
        assert result.errors == []
