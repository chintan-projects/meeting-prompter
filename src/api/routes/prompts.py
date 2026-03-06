"""Prompts WebSocket — real-time trigger result streaming."""
import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prompts"])


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

    try:
        while True:
            msg = await session._prompt_queue.get()
            await ws.send_json(msg)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        logger.info("Prompts WebSocket disconnected")
