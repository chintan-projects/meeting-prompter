"""Tests for lib.notion.parser — Notion document parser for RAG pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lib.notion.parser import NotionDocumentParser
from lib.rag.types import ParsedDocument


def _make_client(
    title: str = "Test Page",
    blocks: list | None = None,
) -> MagicMock:
    """Create a mock NotionClient with configurable title and blocks."""
    client = MagicMock()
    client.get_page_title.return_value = title
    client.get_block_children.return_value = blocks or []
    return client


class _FakePath:
    """String wrapper that preserves notion:// scheme (Path collapses //)."""

    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


class TestParseValidPath:
    """parse() with notion:// URIs."""

    def test_valid_notion_path_returns_parsed_document(self) -> None:
        client = _make_client(
            title="Meeting Notes",
            blocks=[
                {
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "plain_text": "Agenda"}],
                    },
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "plain_text": "Discuss Q2 roadmap"}],
                    },
                },
            ],
        )
        parser = NotionDocumentParser(client)
        doc = parser.parse(_FakePath("notion://abc123"))  # type: ignore[arg-type]

        assert isinstance(doc, ParsedDocument)
        assert doc.path == "notion://abc123"
        assert doc.filename == "Meeting Notes"
        assert doc.mime_type == "text/notion"
        assert doc.token_count > 0
        assert len(doc.sections) > 0
        client.get_page_title.assert_called_once_with("abc123")
        client.get_block_children.assert_called_once_with("abc123")

    def test_page_id_extracted_from_path(self) -> None:
        client = _make_client()
        parser = NotionDocumentParser(client)
        parser.parse(_FakePath("notion://deadbeef-1234"))  # type: ignore[arg-type]
        client.get_page_title.assert_called_with("deadbeef-1234")


class TestParseInvalidPath:
    """parse() with non-notion paths."""

    def test_non_notion_path_raises_value_error(self) -> None:
        client = _make_client()
        parser = NotionDocumentParser(client)
        with pytest.raises(ValueError, match="notion://"):
            parser.parse(Path("/tmp/notes.md"))

    def test_http_path_raises_value_error(self) -> None:
        client = _make_client()
        parser = NotionDocumentParser(client)
        with pytest.raises(ValueError, match="notion://"):
            parser.parse(Path("https://notion.so/page"))


class TestParsePage:
    """parse_page() direct page ID calls."""

    def test_produces_correct_path(self) -> None:
        client = _make_client(title="Sprint Retro")
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("page-id-42")
        assert doc.path == "notion://page-id-42"

    def test_produces_correct_filename(self) -> None:
        client = _make_client(title="Sprint Retro")
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("page-id-42")
        assert doc.filename == "Sprint Retro"

    def test_produces_correct_mime_type(self) -> None:
        client = _make_client()
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("any-id")
        assert doc.mime_type == "text/notion"

    def test_sections_extracted_from_headings(self) -> None:
        client = _make_client(
            title="",
            blocks=[
                {
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "plain_text": "Overview"}],
                    },
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "plain_text": "Some content here"}],
                    },
                },
                {
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "plain_text": "Details"}],
                    },
                },
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "plain_text": "More details here"}],
                    },
                },
            ],
        )
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("sec-test")
        # Should have sections derived from the headings
        headings = [s.heading for s in doc.sections if s.heading]
        assert "Overview" in headings
        assert "Details" in headings


class TestTitlePrepending:
    """Title prepended as h1 when content lacks a leading heading."""

    def test_title_prepended_when_no_heading(self) -> None:
        client = _make_client(
            title="My Document",
            blocks=[
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "plain_text": "Just a paragraph"}],
                    },
                },
            ],
        )
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("title-test")
        assert doc.full_text.startswith("# My Document")

    def test_title_not_prepended_when_heading_exists(self) -> None:
        client = _make_client(
            title="My Document",
            blocks=[
                {
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "plain_text": "Existing Heading"}],
                    },
                },
            ],
        )
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("heading-test")
        # Should not double-prepend
        assert doc.full_text.startswith("# Existing Heading")
        assert doc.full_text.count("# My Document") == 0

    def test_empty_title_not_prepended(self) -> None:
        client = _make_client(
            title="",
            blocks=[
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "plain_text": "Content only"}],
                    },
                },
            ],
        )
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("no-title")
        assert not doc.full_text.startswith("# \n")


class TestEmptyPage:
    """Empty Notion pages produce valid documents."""

    def test_empty_page_produces_valid_document(self) -> None:
        client = _make_client(title="Empty", blocks=[])
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("empty-page")

        assert isinstance(doc, ParsedDocument)
        assert doc.path == "notion://empty-page"
        assert doc.filename == "Empty"
        assert len(doc.sections) >= 1  # at least a root section from title
        assert doc.token_count >= 1

    def test_empty_page_no_title(self) -> None:
        client = _make_client(title="", blocks=[])
        parser = NotionDocumentParser(client)
        doc = parser.parse_page("really-empty")
        assert isinstance(doc, ParsedDocument)
        assert doc.filename == "really-empty"  # falls back to page_id
