"""Notes routes — editing, structured notes generation, export, and save."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from lib.rag_generator import RAGAnswerGenerator
from src.api.notes_generator import generate_structured_notes
from src.api.routes.session import get_session
from src.api.session import Session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notes", tags=["notes"])


def _get_extractor(session: Session) -> Optional[object]:
    """Return the Extract-model generator (F-507) if the orchestrator wired one.

    Returns None today — the LFM2.5-350M-Extract runtime is plumbed but not yet
    wired as the default (needs a live verification run). When the orchestrator
    exposes ``extract_generator``, structured notes route through it automatically.
    """
    orch = getattr(session, "_orchestrator", None)
    extractor = getattr(orch, "extract_generator", None) if orch else None
    return extractor


def _get_generator(session: Session) -> Optional[RAGAnswerGenerator]:
    """Safely extract the RAGAnswerGenerator from the session, or None.

    Returns None if the orchestrator hasn't loaded, the generator model
    is missing, or any component in the chain is unavailable. The caller
    falls back to template-based notes.
    """
    try:
        orch = getattr(session, "_orchestrator", None)
        if orch is None:
            return None
        gen = getattr(orch, "generator", None)
        if gen is None:
            return None
        rag_gen = getattr(gen, "_generator", None)
        if not isinstance(rag_gen, RAGAnswerGenerator):
            return None
        if rag_gen.llm is None:
            # Model not loaded — generate_text() will load lazily, but if
            # it was None after a crash, better to use template fallback
            return None
        return rag_gen
    except Exception:
        logger.warning("Could not access LLM generator — using template fallback")
        return None


class EditRequest(BaseModel):
    segment_id: str
    text: str


class ExportResponse(BaseModel):
    markdown: str
    segment_count: int


class StructuredNotesResponse(BaseModel):
    notes: str
    segment_count: int


@router.put("/edit")
async def edit_segment(req: EditRequest) -> dict:
    """Edit a transcript segment."""
    session = get_session()
    success = session.transcript.edit(req.segment_id, req.text)
    if not success:
        raise HTTPException(status_code=404, detail="Segment not found")
    return {"status": "updated", "segment_id": req.segment_id}


@router.get("/export")
async def export_notes(
    format: str = Query("markdown"),
) -> Dict[str, Union[str, int, float, List[dict]]]:
    """Export merged transcript as markdown or structured JSON.

    Query params:
        format: "markdown" (default) or "json"
    """
    session = get_session()

    if format == "json":
        title = session.meeting_context.title if session.meeting_context else ""
        return {
            "version": "1.0",
            "title": title,
            "duration_seconds": session.elapsed_seconds,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "segment_count": session.transcript.segment_count,
            "segments": session.transcript.export_json(),
        }

    return {
        "markdown": session.transcript.export_markdown(),
        "segment_count": session.transcript.segment_count,
    }


@router.post("/generate", response_model=StructuredNotesResponse)
async def generate_notes() -> StructuredNotesResponse:
    """Generate structured meeting notes from the transcript.

    Uses LFM2.5-Instruct to produce speaker-attributed notes when
    speaker data is available, or generic structured notes otherwise.
    Falls back to a template if no model is available or model crashed.
    """
    session = get_session()
    transcript_md = session.transcript.export_markdown()
    segments = session.transcript.export_json()
    # F-607: lexical speaker-consistency correction (named hand-off/gratitude cues),
    # non-destructive — applied to the notes' view of the segments only.
    if session.meeting_context and session.meeting_context.participants:
        from lib.attribution import correct_segments

        segments = correct_segments(segments, session.meeting_context.participants)

    # Try to use the session's generator if available (may be None after crash)
    generator = _get_generator(session)

    notes = generate_structured_notes(
        transcript_md,
        generator,
        segments=segments,
        meeting_context=session.meeting_context,
        trigger_history=session.trigger_history,
        extractor=_get_extractor(session),
    )
    return StructuredNotesResponse(
        notes=notes,
        segment_count=session.transcript.segment_count,
    )


@router.get("/segments")
async def get_segments() -> List[dict]:
    """Get all transcript segments with edits applied."""
    session = get_session()
    return session.transcript.get_merged()


@router.get("/raw")
async def get_raw_segments() -> List[dict]:
    """Get original unedited transcript segments."""
    session = get_session()
    return session.transcript.get_raw()


class SaveRequest(BaseModel):
    notes: str = ""
    include_transcript: bool = True


class SaveResponse(BaseModel):
    path: str
    filename: str


@router.post("/save", response_model=SaveResponse)
async def save_notes(req: SaveRequest) -> SaveResponse:
    """Save meeting notes and transcript to a markdown file in output/.

    Returns the file path so the frontend can show a confirmation.
    """
    session = get_session()
    transcript_md = session.transcript.export_markdown()
    meeting_title = session.meeting_context.title if session.meeting_context else "Meeting"

    # Build output
    ts = time.strftime("%Y-%m-%d_%H%M")
    safe_title = meeting_title.replace(" ", "-").replace("/", "-")[:40]
    filename = f"{ts}_{safe_title}.md"

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    parts = [f"# {meeting_title}", f"*{time.strftime('%B %d, %Y %H:%M')}*", ""]

    if req.notes.strip():
        parts.append(req.notes.strip())
        parts.append("")

    if req.include_transcript and transcript_md.strip():
        parts.append("---")
        parts.append("")
        parts.append("## Transcript")
        parts.append("")
        parts.append(transcript_md)

    content = "\n".join(parts)
    filepath.write_text(content, encoding="utf-8")
    logger.info("Saved meeting notes to %s (%d chars)", filepath, len(content))

    return SaveResponse(path=str(filepath), filename=filename)


@router.get("/download")
async def download_notes() -> PlainTextResponse:
    """Download the full meeting notes + transcript as markdown.

    Returns the content directly as a text/markdown response with
    Content-Disposition header for browser download.
    """
    session = get_session()
    transcript_md = session.transcript.export_markdown()
    segments = session.transcript.export_json()
    # F-607: lexical speaker-consistency correction (named hand-off/gratitude cues),
    # non-destructive — applied to the notes' view of the segments only.
    if session.meeting_context and session.meeting_context.participants:
        from lib.attribution import correct_segments

        segments = correct_segments(segments, session.meeting_context.participants)
    meeting_title = session.meeting_context.title if session.meeting_context else "Meeting"

    # Try to generate structured notes if possible (may be None after crash)
    generator = _get_generator(session)
    notes = generate_structured_notes(
        transcript_md,
        generator,
        segments=segments,
        meeting_context=session.meeting_context,
        trigger_history=session.trigger_history,
        extractor=_get_extractor(session),
    )

    parts = [
        f"# {meeting_title}",
        f"*{time.strftime('%B %d, %Y %H:%M')}*",
        "",
        notes,
    ]
    if transcript_md.strip():
        parts.extend(["", "---", "", "## Transcript", "", transcript_md])

    content = "\n".join(parts)
    ts = time.strftime("%Y-%m-%d_%H%M")
    filename = f"{ts}_meeting_notes.md"

    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
