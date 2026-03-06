"""Transcript WebSocket — real-time turn-based transcript streaming."""
import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.routes.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcript"])


@router.websocket("/ws/transcript")
async def transcript_ws(ws: WebSocket) -> None:
    """Stream transcript turns in real-time.

    Sends two message types:
        transcript_update — partial turn (active, still accumulating):
            {"type": "transcript_update", "id": "turn-1", "text": "...",
             "timestamp": 1234.5, "end_timestamp": 1236.2, "is_final": false}

        transcript_final — completed turn (finalized on pause):
            {"type": "transcript_final", "id": "turn-1", "text": "...",
             "timestamp": 1234.5, "end_timestamp": 1238.1, "is_final": true}

    The client should upsert by turn ID: update existing turns on
    transcript_update, and mark as final on transcript_final.

    Receives edit messages:
        {"type": "edit", "id": "turn-1", "text": "corrected text"}
    """
    await ws.accept()
    session = get_session()
    logger.info("Transcript WebSocket connected")

    # Send existing turns so late-connecting clients catch up
    for seg_data in session.transcript.get_merged():
        await ws.send_json({
            "type": "transcript_final" if seg_data.get("is_final") else "transcript_update",
            **seg_data,
        })

    # Task to send transcript turns from queue
    async def send_loop() -> None:
        try:
            while True:
                msg = await session._transcript_queue.get()
                await ws.send_json(msg)
        except asyncio.CancelledError:
            pass

    # Task to receive edits from client
    async def recv_loop() -> None:
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "edit":
                    seg_id = msg.get("id", "")
                    new_text = msg.get("text", "")
                    session.transcript.edit(seg_id, new_text)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    sender = asyncio.create_task(send_loop())
    receiver = asyncio.create_task(recv_loop())

    try:
        await asyncio.gather(sender, receiver)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        receiver.cancel()
        logger.info("Transcript WebSocket disconnected")
