"""
RAG pipeline SQLite schema.

Defines the enhanced schema for hybrid retrieval: documents, sections,
chunks (with FTS5), and chunk edges. Includes migration from the v1
flat schema for backward compatibility.
"""

from __future__ import annotations

import sqlite3

# ─── Schema DDL ──────────────────────────────────────────────────────────────

_SCHEMA_V2 = """
-- Documents (enhanced: add mime_type, token_count)
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,
    filename    TEXT    NOT NULL,
    mime_type   TEXT    NOT NULL DEFAULT 'text/plain',
    content     TEXT    NOT NULL,
    file_hash   TEXT    NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Sections (hierarchical document structure)
CREATE TABLE IF NOT EXISTS sections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading       TEXT    NOT NULL DEFAULT '',
    heading_level INTEGER NOT NULL DEFAULT 0,
    heading_path  TEXT    NOT NULL DEFAULT '',
    content       TEXT    NOT NULL,
    start_page    INTEGER,
    end_page      INTEGER,
    section_index INTEGER NOT NULL DEFAULT 0,
    UNIQUE(document_id, section_index)
);

-- Chunks (enhanced: section_id, parent_id, token_count, manual fields)
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id      INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    parent_id       INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    content         TEXT    NOT NULL,
    chunk_index     INTEGER NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    manual_keywords TEXT    DEFAULT '',
    manual_notes    TEXT    DEFAULT '',
    is_edited       INTEGER NOT NULL DEFAULT 0,
    UNIQUE(document_id, chunk_index)
);

-- Chunk edges (lightweight graph for adjacency/hierarchy)
CREATE TABLE IF NOT EXISTS chunk_edges (
    source_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    relationship TEXT    NOT NULL,
    weight       REAL    NOT NULL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, relationship)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunk_edges_source ON chunk_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_chunk_edges_target ON chunk_edges(target_id);
"""

_FTS5_SETUP = """
-- FTS5 virtual table for lexical search (standalone, synced via triggers)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    title,
    heading_path,
    keywords,
    tokenize='porter unicode61'
);
"""

_FTS5_TRIGGERS = """
-- Keep FTS5 in sync with chunks table (standalone mode — use regular DELETE)
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, title, heading_path, keywords)
    VALUES (
        NEW.id,
        NEW.content,
        (SELECT d.filename FROM documents d WHERE d.id = NEW.document_id),
        COALESCE(
            (SELECT s.heading_path FROM sections s WHERE s.id = NEW.section_id),
            ''
        ),
        COALESCE(NEW.manual_keywords, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete BEFORE DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = OLD.id;
    INSERT INTO chunks_fts(rowid, content, title, heading_path, keywords)
    VALUES (
        NEW.id, NEW.content,
        (SELECT d.filename FROM documents d WHERE d.id = NEW.document_id),
        COALESCE(
            (SELECT s.heading_path FROM sections s WHERE s.id = NEW.section_id),
            ''
        ),
        COALESCE(NEW.manual_keywords, '')
    );
END;
"""


# ─── Public API ──────────────────────────────────────────────────────────────


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, FTS5, triggers, and indexes if they don't exist."""
    conn.executescript(_SCHEMA_V2)
    # FTS5 must be created with separate execute calls (not in executescript
    # when mixing with regular DDL in some SQLite builds)
    _safe_create_fts5(conn)
    conn.executescript(_FTS5_TRIGGERS)
    conn.commit()


def migrate_from_v1(conn: sqlite3.Connection) -> None:
    """Migrate from v1 flat schema (documents + chunks) to v2.

    Adds missing columns and tables without losing data. Safe to call
    multiple times (all operations are idempotent).
    """
    # Check if migration is needed by looking for the sections table
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    if "sections" in tables:
        return  # Already migrated

    # Add new columns to documents (ignore if already exist)
    for col, default in [("mime_type", "'text/plain'"), ("token_count", "0")]:
        try:
            conn.execute(
                f"ALTER TABLE documents ADD COLUMN {col} "
                f"{'TEXT NOT NULL' if col == 'mime_type' else 'INTEGER NOT NULL'} "
                f"DEFAULT {default}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Add new columns to chunks
    for col, col_type, default in [
        ("section_id", "INTEGER", "NULL"),
        ("parent_id", "INTEGER", "NULL"),
        ("token_count", "INTEGER NOT NULL", "0"),
        ("manual_keywords", "TEXT", "''"),
        ("manual_notes", "TEXT", "''"),
        ("is_edited", "INTEGER NOT NULL", "0"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE chunks ADD COLUMN {col} {col_type} DEFAULT {default}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Create new tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            heading       TEXT    NOT NULL DEFAULT '',
            heading_level INTEGER NOT NULL DEFAULT 0,
            heading_path  TEXT    NOT NULL DEFAULT '',
            content       TEXT    NOT NULL,
            start_page    INTEGER,
            end_page      INTEGER,
            section_index INTEGER NOT NULL DEFAULT 0,
            UNIQUE(document_id, section_index)
        );

        CREATE TABLE IF NOT EXISTS chunk_edges (
            source_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            target_id    INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            relationship TEXT    NOT NULL,
            weight       REAL    NOT NULL DEFAULT 1.0,
            PRIMARY KEY (source_id, target_id, relationship)
        );

        CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_edges_source ON chunk_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_edges_target ON chunk_edges(target_id);
        """
    )

    # Create FTS5 and triggers
    _safe_create_fts5(conn)
    conn.executescript(_FTS5_TRIGGERS)

    # Backfill FTS5 index for existing chunks
    rows = conn.execute(
        """
        SELECT c.id, c.content, d.filename,
               COALESCE(c.manual_keywords, '') AS keywords
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        """
    ).fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content, title, heading_path, keywords) "
            "VALUES (?, ?, ?, '', ?)",
            (row[0], row[1], row[2], row[3]),
        )

    conn.commit()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _safe_create_fts5(conn: sqlite3.Connection) -> None:
    """Create the FTS5 virtual table, ignoring if it already exists."""
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                title,
                heading_path,
                keywords,
                tokenize='porter unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        pass  # FTS5 already exists or not available
