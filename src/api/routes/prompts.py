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
            [sender, receiver], return_when=asyncio.FIRST_COMPLETED,
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
