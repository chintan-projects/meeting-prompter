"""RAG Engine — hybrid FTS5 + vector retrieval for document Q&A.

Wraps the standalone RAG pipeline library (lib/rag/) with the same
public API the orchestrator and triggers expect: query() returns
(context, confidence, source). Drop-in replacement for the old
ColBERT + Jaccard implementation.
"""
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from lib.rag import RAGPipeline, RAGConfig
from lib.rag.embedder import SentenceTransformerEmbedder
from lib.rag.parser.composite_parser import CompositeParser

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/rag.db")


class RAGEngine:
    """Hybrid RAG engine with FTS5 lexical + vector semantic retrieval.

    Public API unchanged from the old ColBERT engine:
    - query(text) -> (context, confidence, source)
    - rebuild_index()
    - chunk_count property
    """

    def __init__(
        self,
        docs_dir: Path,
        db_path: Optional[Path] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self._db_path = Path(db_path or _DEFAULT_DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._embedder = SentenceTransformerEmbedder()
        self._config = config or RAGConfig(
            file_types=(".pdf", ".md", ".txt", ".markdown"),
            max_chunk_tokens=400,
            chunk_overlap_tokens=50,
        )

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._pipeline = RAGPipeline(
            conn=self._conn,
            embedder=self._embedder,
            config=self._config,
            parser=CompositeParser(),
        )

        self._pipeline.setup()
        self._index_documents()

        self._chunk_count = self._get_chunk_count()
        logger.info(
            "RAG engine ready (hybrid): %d chunks from %s",
            self._chunk_count,
            self.docs_dir,
        )

    def query(self, text: str) -> Tuple[str, float, str]:
        """Find most relevant content for the query.

        Returns:
            Tuple of (context_text, confidence_score, source_filename).
            Satisfies the RAGQueryable protocol used by triggers.
        """
        if not text or not text.strip():
            return "", 0.0, ""

        try:
            results = self._pipeline.retrieve(text, top_k=3)
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return "", 0.0, ""

        if not results:
            return "", 0.0, ""

        top_score = results[0].score
        source_path = results[0].citation.document_path if results[0].citation else ""
        top_source = Path(source_path).name if source_path else ""

        # Combine top results (dedup within 80% of top score)
        seen_texts: set[str] = set()
        combined_chunks: list[str] = []

        for r in results:
            text_key = r.chunk_text[:100].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            if r.score >= top_score * 0.8:
                combined_chunks.append(r.chunk_text)

        combined_context = "\n\n---\n\n".join(combined_chunks)
        return combined_context, top_score, top_source

    def rebuild_index(self) -> None:
        """Force rebuild: clear all data and re-index documents."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM sections")
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()
        self._index_documents()
        self._chunk_count = self._get_chunk_count()
        logger.info("Index rebuilt: %d chunks", self._chunk_count)

    @property
    def is_using_colbert(self) -> bool:
        """Backward compat: always False (ColBERT removed)."""
        return False

    @property
    def is_hybrid(self) -> bool:
        """True: this engine uses hybrid FTS5 + vector retrieval."""
        return True

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return self._chunk_count

    def close(self) -> None:
        """Close the database connection."""
        self._pipeline.close()
        self._conn.close()

    def _index_documents(self) -> None:
        """Index all documents in docs_dir."""
        if not self.docs_dir.exists():
            logger.warning("Docs directory not found: %s", self.docs_dir)
            return
        result = self._pipeline.index([self.docs_dir])
        logger.info(
            "Indexed: %d new, %d updated, %d skipped, %d chunks",
            result.documents_indexed,
            result.documents_updated,
            result.documents_skipped,
            result.chunks_created,
        )
        if result.errors:
            for err in result.errors:
                logger.warning("Indexing error: %s", err)

    def _get_chunk_count(self) -> int:
        """Count chunks in the database."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0


def format_confidence(score: float) -> str:
    """Format confidence score for display."""
    percentage = score * 100
    if percentage >= 50:
        return f"[green]{percentage:.0f}%[/green]"
    elif percentage >= 25:
        return f"[yellow]{percentage:.0f}%[/yellow]"
    else:
        return f"[red]{percentage:.0f}%[/red]"
