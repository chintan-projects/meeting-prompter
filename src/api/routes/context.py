"""Meeting context routes — load/view pre-meeting configuration."""
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from src.api.routes.session import get_session

router = APIRouter(prefix="/context", tags=["context"])


class ContextResponse(BaseModel):
    title: str
    agenda_items: List[str]
    watch_words: List[str]
    participants: List[str]
    key_topics: List[str]
    notes: str


class ContextInput(BaseModel):
    title: str = ""
    agenda_items: List[str] = []
    watch_words: List[str] = []
    participants: List[str] = []
    key_topics: List[str] = []
    notes: str = ""


@router.post("/load")
async def load_context(file: UploadFile = File(...)) -> ContextResponse:
    """Upload a meeting_context.yaml file."""
    if not file.filename or not file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="File must be YAML")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    session = get_session()
    ctx = session.load_context(tmp_path)
    tmp_path.unlink(missing_ok=True)

    if not ctx:
        raise HTTPException(status_code=400, detail="Failed to parse meeting context")

    return ContextResponse(
        title=ctx.title,
        agenda_items=ctx.agenda_items,
        watch_words=ctx.watch_words,
        participants=ctx.participants,
        key_topics=ctx.key_topics,
        notes=ctx.notes,
    )


@router.post("/set")
async def set_context(ctx_input: ContextInput) -> ContextResponse:
    """Set meeting context directly via JSON (no file upload)."""
    from lib.conversation.meeting_context import MeetingContext

    session = get_session()
    session.meeting_context = MeetingContext(
        title=ctx_input.title,
        agenda_items=ctx_input.agenda_items,
        watch_words=ctx_input.watch_words,
        participants=ctx_input.participants,
        key_topics=ctx_input.key_topics,
        notes=ctx_input.notes,
    )
    return ContextResponse(
        title=ctx_input.title,
        agenda_items=ctx_input.agenda_items,
        watch_words=ctx_input.watch_words,
        participants=ctx_input.participants,
        key_topics=ctx_input.key_topics,
        notes=ctx_input.notes,
    )


@router.get("/", response_model=Optional[ContextResponse])
async def get_context() -> Optional[ContextResponse]:
    """Get the currently loaded meeting context."""
    session = get_session()
    ctx = session.meeting_context
    if not ctx:
        return None
    return ContextResponse(
        title=ctx.title,
        agenda_items=ctx.agenda_items,
        watch_words=ctx.watch_words,
        participants=ctx.participants,
        key_topics=ctx.key_topics,
        notes=ctx.notes,
    )
