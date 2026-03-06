"""Session management routes — start/stop/status."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.api.session import Session

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


class AudioHealth(BaseModel):
    total_chunks: int = 0
    speech_chunks: int = 0
    last_rms: float = 0.0
    last_peak: float = 0.0
    all_silent: bool = False


class StatusResponse(BaseModel):
    running: bool
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
    """Start a new meeting session."""
    session = get_session()
    if session.is_running:
        raise HTTPException(status_code=409, detail="Session already running")
    session.start(audio_device=req.audio_device)
    return {"status": "started", "audio_device": req.audio_device}


@router.post("/stop")
async def stop_session() -> dict:
    """Stop the current meeting session."""
    global _session
    session = get_session()
    if not session.is_running:
        raise HTTPException(status_code=409, detail="No session running")
    elapsed = session.elapsed_seconds
    session.stop()
    # Reset singleton so next start gets a fresh session
    _session = None
    return {"status": "stopped", "elapsed_seconds": elapsed}


@router.get("/status", response_model=StatusResponse)
async def session_status() -> StatusResponse:
    """Get current session status."""
    session = get_session()
    status = session.get_status()
    return StatusResponse(**status)
