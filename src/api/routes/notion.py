"""Notion integration routes — status, export, and RAG sync."""

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notion", tags=["notion"])


class NotionStatusResponse(BaseModel):
    enabled: bool
    has_token: bool
    export_parent_set: bool
    rag_sources_count: int


class NotionExportRequest(BaseModel):
    title: str = "Untitled Meeting"
    notes_md: str = ""
    transcript_md: str = ""
    include_transcript: bool = True
    participants: list[str] = []
    date: str = ""
    duration_seconds: float = 0.0


class NotionExportResponse(BaseModel):
    url: str
    status: str = "exported"


@router.get("/status", response_model=NotionStatusResponse)
def notion_status() -> NotionStatusResponse:
    """Check if Notion integration is configured and ready."""
    session = get_session()
    config = session.config.notion

    token = os.environ.get(config.api_token_env, "")
    source_count = len(config.rag_source_page_ids) + len(config.rag_source_database_ids)

    return NotionStatusResponse(
        enabled=config.enabled,
        has_token=bool(token),
        export_parent_set=bool(config.export_parent_page_id),
        rag_sources_count=source_count,
    )


@router.post("/export", response_model=NotionExportResponse)
def export_to_notion(req: NotionExportRequest) -> NotionExportResponse:
    """Export meeting transcript and notes to a Notion page.

    Declared as ``def`` (not ``async def``) so FastAPI runs it in a
    threadpool — the Notion SDK uses synchronous HTTP under the hood.
    """
    session = get_session()
    config = session.config.notion

    if not config.enabled:
        raise HTTPException(status_code=400, detail="Notion integration not enabled")

    token = os.environ.get(config.api_token_env, "")
    if not token:
        raise HTTPException(
            status_code=400,
            detail=f"Notion API token not set (env: {config.api_token_env})",
        )

    if not config.export_parent_page_id:
        raise HTTPException(status_code=400, detail="No export parent page configured")

    from lib.notion.client import NotionClient, NotionClientError
    from lib.notion.exporter import export_meeting

    try:
        client = NotionClient(
            api_token=token,
            max_retries=config.max_retries,
            initial_backoff=config.initial_backoff_s,
            timeout_seconds=config.timeout_s,
            max_block_depth=config.max_block_depth,
        )
        url = export_meeting(
            client=client,
            parent_page_id=config.export_parent_page_id,
            title=req.title,
            date=req.date,
            participants=req.participants,
            notes_md=req.notes_md,
            transcript_md=req.transcript_md if req.include_transcript else "",
            duration_seconds=req.duration_seconds,
        )
        return NotionExportResponse(url=url)
    except NotionClientError as exc:
        logger.error("Notion export failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Notion API error: {exc}") from exc


@router.post("/sync-rag")
def sync_notion_rag() -> dict:
    """Manually sync Notion sources into the RAG index.

    Declared as ``def`` for threadpool execution (sync Notion API calls).
    """
    session = get_session()
    config = session.config.notion

    if not config.enabled:
        raise HTTPException(status_code=400, detail="Notion integration not enabled")

    rag = session.get_rag_engine()
    if rag is None:
        raise HTTPException(status_code=409, detail="No active session with models loaded")

    result = rag.index_notion_sources(config)
    return {
        "status": "synced",
        "documents_indexed": result.documents_indexed,
        "documents_skipped": result.documents_skipped,
        "chunks_created": result.chunks_created,
        "errors": result.errors[:10],
    }
