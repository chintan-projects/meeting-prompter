"""
RAG Pipeline — standalone hybrid retrieval library.

Public API:
    RAGPipeline  — facade with setup/index/retrieve/close lifecycle
    RAGConfig    — all tunables in one immutable dataclass

Zero coupling to MCP. Embedding function injected via Embedder protocol.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from lib.rag.chunker.protocol import Chunker
from lib.rag.chunker.token_chunker import TokenChunker
from lib.rag.config import RAGConfig
from lib.rag.errors import RAGError
from lib.rag.index.protocol import Embedder
from lib.rag.parser.protocol import DocumentParser
from lib.rag.parser.text_parser import TextParser
from lib.rag.rank.heuristic import HeuristicRanker
from lib.rag.rank.protocol import Ranker
from lib.rag.retrieval.engine import retrieve as _retrieve
from lib.rag.storage.schema import init_schema, migrate_from_v1
from lib.rag.types import IndexResult, ParsedDocument, RetrievalResult

# Re-export public types
__all__ = ["RAGPipeline", "RAGConfig", "IndexResult", "RetrievalResult"]


class RAGPipeline:
    """Main entry point for the RAG library.

    Lifecycle: setup() -> index() / retrieve() -> close()
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        config: RAGConfig | None = None,
        parser: DocumentParser | None = None,
        chunker: Chunker | None = None,
        ranker: Ranker | None = None,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._config = config or RAGConfig()
        self._parser: DocumentParser = parser or TextParser()
        self._chunker: Chunker = chunker or TokenChunker()
        self._ranker: Ranker = ranker or HeuristicRanker(conn)

    @property
    def config(self) -> RAGConfig:
        """Current pipeline configuration."""
        return self._config

    def setup(self) -> None:
        """Initialise schema and run migrations. Call once on startup."""
        try:
            migrate_from_v1(self._conn)
        except Exception:
            pass  # No v1 schema to migrate from
        init_schema(self._conn)

    def index(self, paths: list[Path], recursive: bool = True) -> IndexResult:
        """Parse, chunk, embed, and index documents.

        Args:
            paths: Files or directories to index.
            recursive: Recurse into subdirectories.

        Returns:
            IndexResult with counts of indexed/skipped/updated documents.
        """
        result = IndexResult()
        file_types = set(self._config.file_types)

        for path in paths:
            if path.is_file():
                self._index_file(path, result)
            elif path.is_dir():
                pattern = "**/*" if recursive else "*"
                for file_path in sorted(path.glob(pattern)):
                    if file_path.is_file() and file_path.suffix.lower() in file_types:
                        self._index_file(file_path, result)

        self._conn.commit()
        return result

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_path: str | None = None,
    ) -> list[RetrievalResult]:
        """Full hybrid retrieval: FTS5 + vector + fusion + rank + citations."""
        return _retrieve(
            self._conn,
            query,
            self._embedder,
            self._config,
            top_k=top_k,
            filter_path=filter_path,
            ranker=self._ranker,
        )

    def index_document(self, doc: ParsedDocument) -> IndexResult:
        """Index a pre-parsed document (e.g. from Notion API or other sources).

        Bypasses file I/O — accepts a ``ParsedDocument`` directly.  Uses the
        same hash-based dedup, chunking, and embedding as ``index()``.
        """
        result = IndexResult()
        self._store_document(doc, result)
        self._conn.commit()
        return result

    def close(self) -> None:
        """Cleanup. Matches setup() for resource symmetry."""
        # Currently a no-op; reserved for future resource cleanup
        pass

    # ─── Internal ─────────────────────────────────────────────────────────

    def _index_file(self, file_path: Path, result: IndexResult) -> None:
        """Index a single file: parse, check hash, chunk, embed, store."""
        try:
            doc = self._parser.parse(file_path)
        except RAGError as exc:
            result.errors.append(f"{file_path}: {exc}")
            return
        except Exception as exc:
            result.errors.append(f"{file_path}: {exc}")
            return

        self._store_document(doc, result)

    def _store_document(self, doc: ParsedDocument, result: IndexResult) -> None:
        """Hash-check, chunk, embed, and store a parsed document."""
        file_hash = hashlib.sha256(doc.full_text.encode("utf-8")).hexdigest()
        abs_path = doc.path

        existing = self._conn.execute(
            "SELECT id, file_hash FROM documents WHERE path = ?",
            (abs_path,),
        ).fetchone()

        if existing and existing[1] == file_hash:
            result.documents_skipped += 1
            return

        if existing:
            doc_id: int = existing[0]
            self._conn.execute(
                "UPDATE documents SET content=?, file_hash=?, mime_type=?, "
                "token_count=?, indexed_at=datetime('now') WHERE id=?",
                (doc.full_text, file_hash, doc.mime_type, doc.token_count, doc_id),
            )
            self._conn.execute("DELETE FROM sections WHERE document_id=?", (doc_id,))
            self._conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            result.documents_updated += 1
        else:
            cursor = self._conn.execute(
                "INSERT INTO documents (path, filename, content, file_hash, "
                "mime_type, token_count) VALUES (?,?,?,?,?,?)",
                (
                    abs_path,
                    doc.filename,
                    doc.full_text,
                    file_hash,
                    doc.mime_type,
                    doc.token_count,
                ),
            )
            doc_id = cursor.lastrowid  # type: ignore[assignment]
            result.documents_indexed += 1

        # Insert sections
        section_ids: list[int | None] = []
        for idx, section in enumerate(doc.sections):
            sec_cursor = self._conn.execute(
                "INSERT INTO sections (document_id, heading, heading_level, "
                "heading_path, content, start_page, end_page, section_index) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    section.heading,
                    section.heading_level,
                    section.heading_path,
                    section.content,
                    section.start_page,
                    section.end_page,
                    idx,
                ),
            )
            section_ids.append(sec_cursor.lastrowid)
            result.sections_created += 1

        # Chunk sections
        chunks = self._chunker.chunk(doc.sections, self._config)

        # Embed all chunks in batch
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_batch(texts)

        # Serialise embeddings
        import struct

        dim = self._embedder.dimension
        pack_fmt = f"<{dim}f"

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb_bytes = struct.pack(pack_fmt, *emb)

            section_id = None
            if chunk.section_index < len(section_ids):
                section_id = section_ids[chunk.section_index]

            self._conn.execute(
                "INSERT INTO chunks (document_id, section_id, content, "
                "chunk_index, token_count, embedding) VALUES (?,?,?,?,?,?)",
                (
                    doc_id,
                    section_id,
                    chunk.content,
                    idx,
                    chunk.token_count,
                    emb_bytes,
                ),
            )
            result.chunks_created += 1
