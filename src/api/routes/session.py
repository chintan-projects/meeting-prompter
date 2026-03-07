"""Session management routes — start/stop/pause/resume/status/reindex."""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.session import Session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["session"])

# Module-level session singleton (one active session at a time)
_session: Optional[Session] = None


def get_session() -> Session:
    """Get the current session, creating one if needed."""
    global _session
    if _session is None:
        _session = Session()
    return _session


class StartRequest(BaseModel):
    audio_device: str = "BlackHole 2ch"
    mic_device: str = "MacBook Pro Microphone"
    # Meeting context (optional — Quick Start omits these)
    title: str = ""
    agenda_items: List[str] = []
    watch_words: List[str] = []
    participants: List[str] = []


class AudioHealth(BaseModel):
    total_chunks: int = 0
    speech_chunks: int = 0
    last_rms: float = 0.0
    last_peak: float = 0.0
    all_silent: bool = False


class StatusResponse(BaseModel):
    running: bool
    paused: bool = False
    loading: bool
    elapsed_seconds: float
    segment_count: int
    meeting_title: str
    audio_health: AudioHealth = AudioHealth()


@router.get("/devices")
async def list_devices() -> dict:
    """List available audio input devices."""
    import sounddevice as sd

    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
    return {"devices": devices}


@router.post("/start")
async def start_session(req: StartRequest) -> dict:
    """Start a new meeting session.

    Creates a fresh session each time so transcript/RAG state is clean.
    The previous session (if any) is discarded. Meeting context (title,
    agenda, watch words, participants) is set on the new session before
    the pipeline starts.
    """
    global _session
    session = get_session()
    if session.is_running:
        raise HTTPException(status_code=409, detail="Session already running")

    # Create a fresh session for a new recording
    _session = Session()

    # Set meeting context if provided (fixes race where context was set on old session)
    if req.title or req.agenda_items or req.watch_words or req.participants:
        from lib.conversation.meeting_context import MeetingContext

        _session.meeting_context = MeetingContext(
            title=req.title,
            agenda_items=req.agenda_items,
            watch_words=req.watch_words,
            participants=req.participants,
        )

    _session.start(audio_device=req.audio_device, mic_device=req.mic_device)
    return {
        "status": "started",
        "audio_device": req.audio_device,
        "mic_device": req.mic_device,
    }


@router.post("/stop")
async def stop_session() -> dict:
    """Stop the current meeting session.

    Keeps the session object alive so transcript data remains
    available for export/notes generation. A new session is only
    created when the user starts a new recording.
    """
    session = get_session()
    if not session.is_running:
        raise HTTPException(status_code=409, detail="No session running")
    elapsed = session.elapsed_seconds
    session.stop()
    return {"status": "stopped", "elapsed_seconds": elapsed}


@router.post("/pause")
async def pause_session() -> dict:
    """Pause the current session. Audio capture stops, models stay loaded."""
    session = get_session()
    if not session.is_running:
        raise HTTPException(status_code=409, detail="No session running")
    if session.is_paused:
        raise HTTPException(status_code=409, detail="Session already paused")
    session.pause()
    return {"status": "paused", "elapsed_seconds": session.elapsed_seconds}


@router.post("/resume")
async def resume_session() -> dict:
    """Resume a paused session."""
    session = get_session()
    if not session.is_running:
        raise HTTPException(status_code=409, detail="No session running")
    if not session.is_paused:
        raise HTTPException(status_code=409, detail="Session not paused")
    session.resume()
    return {"status": "resumed", "elapsed_seconds": session.elapsed_seconds}


@router.get("/status", response_model=StatusResponse)
async def session_status() -> StatusResponse:
    """Get current session status."""
    session = get_session()
    status = session.get_status()
    return StatusResponse(**status)


@router.post("/reindex")
async def reindex_documents() -> dict:
    """Force-rebuild the RAG index from context/ directory.

    Call this after adding or removing files from context/ to ensure
    the RAG engine picks up changes without restarting the app.
    """
    session = get_session()
    if session.is_running and session._orchestrator:
        rag = session._orchestrator.rag
        logger.info("Rebuilding RAG index...")
        rag.rebuild_index()
        logger.info("RAG index rebuilt: %d chunks", rag.chunk_count)
        return {"status": "reindexed", "chunk_count": rag.chunk_count}
    return {"status": "error", "detail": "No active session with models loaded"}
