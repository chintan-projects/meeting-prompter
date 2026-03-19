"""Notion document parser — satisfies the ``DocumentParser`` protocol.

Fetches a Notion page's block tree via ``NotionClient``, converts to
markdown, then extracts heading-based sections using the shared
``extract_sections`` helper from ``TextParser``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lib.notion.block_converter import blocks_to_markdown
from lib.notion.client import NotionClient
from lib.rag.parser.text_parser import estimate_tokens, extract_sections
from lib.rag.types import ParsedDocument

logger = logging.getLogger(__name__)


class NotionDocumentParser:
    """Parses Notion pages into ``ParsedDocument`` for the RAG pipeline.

    The ``path`` field uses a synthetic ``notion://<page_id>`` URI so the
    storage layer can distinguish Notion documents from local files.

    Args:
        client: Authenticated ``NotionClient`` instance.
    """

    def __init__(self, client: NotionClient) -> None:
        self._client = client

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a Notion page identified by a synthetic path.

        Expects ``path`` to be ``notion://<page_id>`` (the stem is the ID).
        Raises ``ValueError`` for non-Notion paths.
        """
        path_str = str(path)
        if not path_str.startswith("notion://"):
            raise ValueError(f"NotionDocumentParser expects notion:// paths, got: {path_str}")

        page_id = path_str.replace("notion://", "")
        return self.parse_page(page_id)

    def parse_page(self, page_id: str) -> ParsedDocument:
        """Fetch and parse a Notion page by its ID."""
        title = self._client.get_page_title(page_id)
        blocks = self._client.get_block_children(page_id)
        markdown = blocks_to_markdown(blocks)

        if title and not markdown.startswith("#"):
            markdown = f"# {title}\n\n{markdown}"

        sections = extract_sections(markdown)
        token_count = estimate_tokens(markdown)

        return ParsedDocument(
            path=f"notion://{page_id}",
            filename=title or page_id,
            mime_type="text/notion",
            sections=sections,
            full_text=markdown,
            token_count=token_count,
        )
