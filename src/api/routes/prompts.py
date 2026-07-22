"""Prompts WebSocket + on-demand generation.

The live path is retrieval-first (F-705/D-08): /ws/prompts streams borrowable
units (method="retrieval", with heading + source_text for expand-to-source).
POST /prompts/generate is the demoted, user-gated LLM path (D-02).
"""

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prompts"])


class GenerateRequest(BaseModel):
    trigger_text: str
    trigger_type: str = "question"


@router.post("/prompts/generate")
def generate_on_demand(req: GenerateRequest) -> Dict[str, Any]:
    """User-gated LLM answer for a trigger (sync — runs in the threadpool)."""
    text = req.trigger_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="trigger_text must be non-empty")
    session = get_session()
    orchestrator = getattr(session, "_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=409, detail="no active session — start a meeting first")
    result = orchestrator.generate_for_text(text, req.trigger_type)
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
