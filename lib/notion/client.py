"""Notion API client wrapper.

Thin layer over the official ``notion-client`` Python SDK.  Handles
authentication (via env var), pagination, recursive block fetching,
and exponential backoff on 429 rate-limit responses.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Defaults — overridden by NotionConfig if provided
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_INITIAL_BACKOFF_S = 1.0
_DEFAULT_TIMEOUT_S = 30
_DEFAULT_MAX_BLOCK_DEPTH = 10


class NotionClientError(Exception):
    """Raised when a Notion API call fails after retries."""


class NotionClient:
    """Lightweight wrapper around the Notion SDK.

    Args:
        api_token: Notion integration token.  Falls back to the
            ``NOTION_API_TOKEN`` environment variable when *None*.
        max_retries: Maximum retry attempts on 429 rate-limit responses.
        initial_backoff: Initial backoff in seconds (doubles per retry).
        timeout_seconds: HTTP request timeout for Notion API calls.
        max_block_depth: Maximum recursion depth for block children.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_backoff: float = _DEFAULT_INITIAL_BACKOFF_S,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
        max_block_depth: int = _DEFAULT_MAX_BLOCK_DEPTH,
    ) -> None:
        token = api_token or os.environ.get("NOTION_API_TOKEN", "")
        if not token:
            raise NotionClientError(
                "Notion API token not provided and NOTION_API_TOKEN env var is empty"
            )

        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_block_depth = max_block_depth

        try:
            from notion_client import Client  # type: ignore[import-untyped]
        except ImportError as exc:
            raise NotionClientError(
                "notion-client package not installed — run: pip install notion-client"
            ) from exc

        self._client = Client(auth=token, timeout_ms=timeout_seconds * 1000)

    # ── Page operations ───────────────────────────────────────────────

    def get_page(self, page_id: str) -> Dict[str, object]:
        """Fetch a page's properties (not content blocks)."""
        return self._retry(lambda: self._client.pages.retrieve(page_id=page_id))

    def get_page_title(self, page_id: str) -> str:
        """Extract the plain-text title from a page's properties."""
        page = self.get_page(page_id)
        props = page.get("properties", {})
        if not isinstance(props, dict):
            return ""
        for prop in props.values():
            if not isinstance(prop, dict):
                continue
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                if isinstance(title_parts, list):
                    return "".join(
                        t.get("plain_text", "") for t in title_parts if isinstance(t, dict)
                    )
        return ""

    def get_last_edited(self, page_id: str) -> str:
        """Return the ``last_edited_time`` ISO string for a page."""
        page = self.get_page(page_id)
        result = page.get("last_edited_time", "")
        return str(result) if result else ""

    # ── Block operations ──────────────────────────────────────────────

    def get_block_children(
        self,
        block_id: str,
        _depth: int = 0,
    ) -> List[Dict[str, object]]:
        """Recursively fetch all child blocks of *block_id*.

        Args:
            block_id: The parent block/page ID.
            _depth: Internal recursion counter (do not set manually).
        """
        blocks: List[Dict[str, object]] = []
        cursor: Optional[str] = None

        while True:
            kwargs: Dict[str, object] = {"block_id": block_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = self._retry(lambda kw=kwargs: self._client.blocks.children.list(**kw))
            for block in resp.get("results", []):
                if not isinstance(block, dict):
                    continue
                blocks.append(block)
                if block.get("has_children") and _depth < self._max_block_depth:
                    block["children"] = self.get_block_children(str(block["id"]), _depth=_depth + 1)
                elif block.get("has_children"):
                    logger.warning(
                        "Skipping nested blocks at depth %d for block %s",
                        _depth,
                        block.get("id"),
                    )
            if not resp.get("has_more"):
                break
            cursor = str(resp.get("next_cursor", "")) or None

        return blocks

    # ── Database operations ───────────────────────────────────────────

    def get_database_pages(
        self,
        database_id: str,
        max_pages: int = 100,
    ) -> List[Dict[str, object]]:
        """Paginate through all pages in a database (up to *max_pages*)."""
        pages: List[Dict[str, object]] = []
        cursor: Optional[str] = None

        while len(pages) < max_pages:
            kwargs: Dict[str, object] = {"database_id": database_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = self._retry(lambda kw=kwargs: self._client.databases.query(**kw))
            pages.extend(resp.get("results", []))
            if not resp.get("has_more"):
                break
            cursor = str(resp.get("next_cursor", "")) or None

        return pages[:max_pages]

    # ── Page creation ─────────────────────────────────────────────────

    def create_page(
        self,
        parent_id: str,
        title: str,
        children: List[Dict[str, object]],
    ) -> str:
        """Create a child page under *parent_id* and return its URL."""
        page = self._retry(
            lambda: self._client.pages.create(
                parent={"page_id": parent_id},
                properties={
                    "title": [{"text": {"content": title}}],
                },
                children=children[:100],  # Notion limit: 100 blocks per create
            )
        )
        page_url: str = str(page.get("url", ""))

        # Append remaining children in batches (Notion limit: 100 per call)
        remaining = children[100:]
        page_id = page["id"]
        while remaining:
            batch, remaining = remaining[:100], remaining[100:]
            self._retry(
                lambda b=batch: self._client.blocks.children.append(
                    block_id=page_id,
                    children=b,
                )
            )

        return page_url

    # ── Retry helper ──────────────────────────────────────────────────

    def _retry(self, fn: Callable[[], T]) -> T:
        """Execute *fn* with exponential backoff on 429 / transient errors."""
        backoff = self._initial_backoff
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                # Check for rate-limit (HTTPStatusError with 429)
                status = getattr(exc, "status", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                if status == 429 and attempt < self._max_retries:
                    logger.warning(
                        "Notion rate-limited (429), retrying in %.1fs (attempt %d/%d)",
                        backoff,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise NotionClientError(f"Notion API error: {exc}") from exc

        raise NotionClientError(f"Notion API error after retries: {last_exc}") from last_exc
