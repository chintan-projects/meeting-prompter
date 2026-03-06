"""Notes routes — editing, structured notes generation, export, and save."""
import logging
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List

from src.api.notes_generator import generate_structured_notes
from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notes", tags=["notes"])


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


@router.get("/export", response_model=ExportResponse)
async def export_notes() -> ExportResponse:
    """Export merged transcript as markdown."""
    session = get_session()
    return ExportResponse(
        markdown=session.transcript.export_markdown(),
        segment_count=session.transcript.segment_count,
    )


@router.post("/generate", response_model=StructuredNotesResponse)
async def generate_notes() -> StructuredNotesResponse:
    """Generate structured meeting notes from the transcript.

    Uses LFM2.5-Instruct to produce Summary, Key Decisions,
    Action Items, and Follow-ups. Falls back to a template
    if no model is available.
    """
    session = get_session()
    transcript_md = session.transcript.export_markdown()

    # Try to use the session's generator if available
    generator = None
    if session._orchestrator is not None:
        gen = session._orchestrator.generator
        if gen._generator is not None:
            generator = gen._generator

    notes = generate_structured_notes(transcript_md, generator)
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
    meeting_title = session.meeting_context.title if session.meeting_context else "Meeting"

    # Try to generate structured notes if possible
    generator = None
    if session._orchestrator is not None:
        gen = session._orchestrator.generator
        if gen._generator is not None:
            generator = gen._generator

    notes = generate_structured_notes(transcript_md, generator)

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
