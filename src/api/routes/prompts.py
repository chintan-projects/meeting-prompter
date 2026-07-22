"""Prompts WebSocket + the user-gated answer paths (D-02).

The live path is retrieval-first (F-705/D-08): /ws/prompts streams borrowable
units (method="retrieval", with heading + source_text for expand-to-source) —
but only while the listen window is armed, since the default is quiet.

Three user-initiated surfaces, none of which pass through the listen gate
(asking *is* the permission):

- ``POST /prompts/listen``  — arm/disarm/toggle the temporal window (Cmd+L).
- ``POST /prompts/answer``  — answer a selected transcript span (spatial).
- ``POST /prompts/generate``— the demoted LLM path (the ✨ button on a card).
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prompts"])


class GenerateRequest(BaseModel):
    trigger_text: str
    trigger_type: str = "question"


class ListenRequest(BaseModel):
    """``armed`` omitted/null toggles; true/false sets explicitly (idempotent)."""

    armed: Optional[bool] = None


class AnswerRequest(BaseModel):
    text: str
    trigger_type: str = "question"


def _orchestrator() -> Any:
    """The live orchestrator, or 409 if no meeting is running."""
    session = get_session()
    orch = getattr(session, "_orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=409, detail="no active session — start a meeting first")
    return orch


def _gate() -> Any:
    """The live listen gate. Always present on a real orchestrator."""
    gate = getattr(_orchestrator(), "listen_gate", None)
    if gate is None:
        raise HTTPException(status_code=409, detail="listen gating unavailable")
    return gate


def _card(result: Any, trigger_text: str, trigger_type: str) -> Dict[str, Any]:
    """Shape a GenerationResult like a `prompt` WS message so the UI renders it identically."""
    return {
        "type": "prompt",
        "trigger_type": trigger_type,
        "trigger_text": trigger_text,
        "answer": result.answer,
        "confidence": result.confidence,
        "method": result.method,
        "latency_ms": result.latency_ms,
        "source": result.source,
        "heading": getattr(result, "heading", ""),
        "source_text": getattr(result, "source_text", ""),
        "persistence": "persistent",  # user asked for it — never auto-dismiss
        "dismiss_ms": 0,
    }


@router.get("/prompts/listen")
def listen_state() -> Dict[str, Any]:
    """Current listen-window state (armed, since, expiry, always-on types)."""
    return dict(_gate().state())


@router.post("/prompts/listen")
async def set_listen(req: ListenRequest) -> Dict[str, Any]:
    """Arm, disarm, or toggle the listen window (D-02, temporal).

    Broadcasts the resulting state on /ws/prompts so every connected client
    agrees on it — the backend is the source of truth, not the keypress.
    """
    session = get_session()
    gate = _gate()
    if req.armed is None:
        armed = gate.toggle()
    elif req.armed:
        armed = gate.arm()
    else:
        armed = gate.disarm()
    state = dict(gate.state())
    await session._prompt_queue.put({"type": "listen_state", **state})
    logger.info("listen window %s via API", "armed" if armed else "disarmed")
    return state


@router.post("/prompts/answer")
def answer_selection(req: AnswerRequest) -> Dict[str, Any]:
    """Answer an explicitly selected transcript span (D-02, spatial).

    Retrieval-first and gate-exempt: whatever the user highlighted is the query,
    whether or not it would have tripped a trigger.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    result = _orchestrator().retrieve_for_text(text, req.trigger_type)
    if result is None or not result.answer or result.answer.startswith("["):
        return {"answer": "", "note": "nothing borrowable in your corpus for that"}
    return _card(result, text, req.trigger_type)


@router.post("/prompts/generate")
def generate_on_demand(req: GenerateRequest) -> Dict[str, Any]:
    """User-gated LLM answer for a trigger (sync — runs in the threadpool)."""
    text = req.trigger_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="trigger_text must be non-empty")
    result = _orchestrator().generate_for_text(text, req.trigger_type)
    if result is None or not result.answer or result.answer.startswith("["):
        return {"answer": "", "note": "no grounded answer available"}
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "method": result.method,
        "latency_ms": result.latency_ms,
        "source": result.source,
    }


@router.websocket("/ws/prompts")
async def prompts_ws(ws: WebSocket) -> None:
    """Stream trigger results (prompts) in real-time.

    Sends JSON messages:
        {
            "type": "prompt",
            "trigger_type": "question",
            "trigger_text": "What is the deployment timeline?",
            "answer": "Edge SDK ships Q2...",
            "confidence": 0.85,
            "method": "hybrid",
            "latency_ms": 450.2,
            "source": "docs/roadmap.md"
        }
    """
    await ws.accept()
    session = get_session()
    logger.info("Prompts WebSocket connected")

    closed = asyncio.Event()

    async def send_loop() -> None:
        try:
            while not closed.is_set():
                msg = await session._prompt_queue.get()
                if closed.is_set():
                    break
                await ws.send_json(msg)
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
            pass

    async def recv_loop() -> None:
        """Wait for client disconnect."""
        try:
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            closed.set()

    sender = asyncio.create_task(send_loop())
    receiver = asyncio.create_task(recv_loop())

    try:
        done, pending = await asyncio.wait(
            [sender, receiver],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        closed.set()
        sender.cancel()
        receiver.cancel()
        logger.info("Prompts WebSocket disconnected")
