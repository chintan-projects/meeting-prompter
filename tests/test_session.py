"""Tests for src.api.session — thread-safe queue bridge and session lifecycle."""
import asyncio
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.api.session import Session


class TestThreadSafePut:
    """Tests for _thread_safe_put — bridging background thread to asyncio."""

    @pytest.mark.asyncio
    async def test_put_with_running_loop(self) -> None:
        """Items put via call_soon_threadsafe should arrive in the queue."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        queue: asyncio.Queue[dict] = asyncio.Queue()
        msg = {"type": "test", "data": "hello"}
        session._thread_safe_put(queue, msg)

        # call_soon_threadsafe schedules the put, give the loop a tick
        await asyncio.sleep(0.01)
        assert not queue.empty()
        item = queue.get_nowait()
        assert item == msg

    @pytest.mark.asyncio
    async def test_put_without_loop_falls_back(self) -> None:
        """Without a loop reference, fall back to direct put_nowait."""
        session = Session()
        session._loop = None

        queue: asyncio.Queue[dict] = asyncio.Queue()
        session._thread_safe_put(queue, {"type": "fallback"})
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_multiple_puts_all_arrive(self) -> None:
        """Multiple items should all arrive in order."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        queue: asyncio.Queue[dict] = asyncio.Queue()
        for i in range(10):
            session._thread_safe_put(queue, {"seq": i})

        await asyncio.sleep(0.05)
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert len(items) == 10
        assert [it["seq"] for it in items] == list(range(10))


class TestTurnCallbacks:
    """Tests for turn-based callbacks pushing to the transcript queue."""

    @pytest.mark.asyncio
    async def test_on_turn_update_message_format(self) -> None:
        """Turn update should push transcript_update to the queue."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="hello world",
            start_timestamp=123.456, end_timestamp=123.456,
        )
        session._on_turn_update(turn)

        await asyncio.sleep(0.01)
        msg = session._transcript_queue.get_nowait()
        assert msg["type"] == "transcript_update"
        assert msg["id"] == "turn-1"
        assert msg["text"] == "hello world"
        assert msg["timestamp"] == 123.456
        assert msg["is_final"] is False

    @pytest.mark.asyncio
    async def test_on_turn_final_message_format(self) -> None:
        """Turn finalization should push transcript_final to the queue."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="complete sentence here",
            start_timestamp=100.0, end_timestamp=105.0, is_final=True,
        )
        session._on_turn_final(turn)

        await asyncio.sleep(0.01)
        msg = session._transcript_queue.get_nowait()
        assert msg["type"] == "transcript_final"
        assert msg["id"] == "turn-1"
        assert msg["is_final"] is True
        assert msg["end_timestamp"] == 105.0

    @pytest.mark.asyncio
    async def test_turn_update_upserts_into_store(self) -> None:
        """Turn update should upsert into the transcript store."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="hello",
            start_timestamp=100.0, end_timestamp=100.0,
        )
        session._on_turn_update(turn)
        assert session.transcript.segment_count == 1

        # Update the same turn
        turn.text = "hello world"
        turn.end_timestamp = 101.0
        session._on_turn_update(turn)
        assert session.transcript.segment_count == 1  # Still 1, upserted
        merged = session.transcript.get_merged()
        assert merged[0]["text"] == "hello world"


class TestSessionLifecycle:
    """Tests for start/stop/status without real audio hardware."""

    def test_initial_state(self) -> None:
        session = Session()
        assert session.is_running is False
        assert session.elapsed_seconds == 0.0
        status = session.get_status()
        assert status["running"] is False
        assert status["loading"] is False
        assert status["segment_count"] == 0

    def test_start_sets_running_and_loading(self) -> None:
        """start() should set running/loading flags immediately."""
        session = Session()

        # Patch MeetingOrchestrator to avoid loading real models
        with patch("src.api.session.Session._run_pipeline"):
            session.start(audio_device="test")

        assert session._running is True
        assert session._loading is True

    def test_stop_clears_running(self) -> None:
        session = Session()
        session._running = True
        session._start_time = time.time()
        session.stop()
        assert session._running is False

    def test_elapsed_seconds(self) -> None:
        session = Session()
        session._start_time = time.time() - 10.0
        elapsed = session.elapsed_seconds
        assert 9.5 < elapsed < 11.0

    def test_double_start_rejected(self) -> None:
        """Starting twice should be a no-op."""
        session = Session()
        session._running = True  # Simulate already running
        # This should return early without creating a second thread
        session.start(audio_device="test")
        assert session._thread is None  # No new thread created

    def test_status_includes_audio_health(self) -> None:
        """Status should include audio_health when orchestrator has audio."""
        session = Session()
        mock_orch = MagicMock()
        mock_orch.audio.get_audio_health.return_value = {
            "total_chunks": 10,
            "speech_chunks": 3,
            "last_rms": 0.05,
            "last_peak": 0.1,
            "all_silent": False,
        }
        session._orchestrator = mock_orch

        status = session.get_status()
        assert status["audio_health"]["total_chunks"] == 10
        assert status["audio_health"]["speech_chunks"] == 3
        assert status["audio_health"]["all_silent"] is False
