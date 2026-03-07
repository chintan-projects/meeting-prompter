"""Tests for src.api.session — callback-based pipeline and session lifecycle."""
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
    async def test_on_turn_final_emits_polished_when_refiner_active(self) -> None:
        """With a text refiner, _on_turn_final emits transcript_polished."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        # Mock the text refiner
        mock_refiner = MagicMock()
        mock_refiner.refine.return_value = "Complete sentence here, polished."
        session._text_refiner = mock_refiner

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="complete sentence here",
            start_timestamp=100.0, end_timestamp=105.0, is_final=True,
        )
        session._on_turn_final(turn)

        await asyncio.sleep(0.02)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        # Should have transcript_final + transcript_polished
        types = [m["type"] for m in messages]
        assert "transcript_final" in types
        assert "transcript_polished" in types

        polished_msg = next(m for m in messages if m["type"] == "transcript_polished")
        assert polished_msg["text"] == "Complete sentence here, polished."
        assert polished_msg["id"] == "turn-1"
        assert polished_msg["is_final"] is True

    @pytest.mark.asyncio
    async def test_on_turn_final_no_polished_when_text_unchanged(self) -> None:
        """If refiner returns same text, no transcript_polished is emitted."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        mock_refiner = MagicMock()
        mock_refiner.refine.return_value = "same text here"
        session._text_refiner = mock_refiner

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="same text here",
            start_timestamp=100.0, end_timestamp=105.0, is_final=True,
        )
        session._on_turn_final(turn)

        await asyncio.sleep(0.02)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        types = [m["type"] for m in messages]
        assert "transcript_final" in types
        assert "transcript_polished" not in types

    @pytest.mark.asyncio
    async def test_on_turn_final_no_polished_without_refiner(self) -> None:
        """Without a text refiner, only transcript_final is emitted."""
        session = Session()
        session._loop = asyncio.get_running_loop()
        session._text_refiner = None

        from src.api.transcript_buffer import Turn

        turn = Turn(
            id="turn-1", text="raw text here",
            start_timestamp=100.0, end_timestamp=105.0, is_final=True,
        )
        session._on_turn_final(turn)

        await asyncio.sleep(0.01)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        assert len(messages) == 1
        assert messages[0]["type"] == "transcript_final"

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


class TestSourceBasedAttribution:
    """Tests for dual-stream source-based speaker attribution."""

    @pytest.mark.asyncio
    async def test_mic_source_sets_speaker_you(self) -> None:
        """Turn with source='mic' should get speaker='You'."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        turn = Turn(
            id="turn-1", text="hello from mic",
            start_timestamp=100.0, end_timestamp=102.0,
            is_final=True, source="mic",
        )
        session._on_turn_final(turn)

        assert turn.speaker == "You"
        await asyncio.sleep(0.01)
        msg = session._transcript_queue.get_nowait()
        assert msg["speaker"] == "You"
        assert msg["source"] == "mic"

    @pytest.mark.asyncio
    async def test_system_source_sets_speaker_others(self) -> None:
        """Turn with source='system' should get speaker='Others'."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        turn = Turn(
            id="turn-1", text="hello from system audio",
            start_timestamp=100.0, end_timestamp=102.0,
            is_final=True, source="system",
        )
        session._on_turn_final(turn)

        assert turn.speaker == "Others"
        await asyncio.sleep(0.01)
        msg = session._transcript_queue.get_nowait()
        assert msg["speaker"] == "Others"
        assert msg["source"] == "system"

    @pytest.mark.asyncio
    async def test_empty_source_keeps_speaker_empty(self) -> None:
        """Turn with no source stays with empty speaker (legacy compat)."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        turn = Turn(
            id="turn-1", text="no source set",
            start_timestamp=100.0, end_timestamp=102.0,
            is_final=True, source="",
        )
        session._on_turn_final(turn)

        assert turn.speaker == ""

    @pytest.mark.asyncio
    async def test_on_transcription_tags_system_source(self) -> None:
        """_on_transcription should tag chunks with source='system'."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session._on_transcription("system speech here", 100.0)
        turn = session._transcript_buffer.active_turn
        assert turn is not None
        assert turn.source == "system"

    @pytest.mark.asyncio
    async def test_on_mic_transcription_tags_mic_source(self) -> None:
        """_on_mic_transcription should tag chunks with source='mic'."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session._on_mic_transcription("mic speech here", 100.0)
        turn = session._transcript_buffer.active_turn
        assert turn is not None
        assert turn.source == "mic"

    @pytest.mark.asyncio
    async def test_source_change_creates_new_turn(self) -> None:
        """Switching source (mic→system) should finalize and create new turn."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session._on_transcription("system speech", 100.0)
        session._on_mic_transcription("mic speech", 103.5)

        # Should have finalized the system turn and started a new mic turn
        assert session._transcript_buffer.active_turn is not None
        assert session._transcript_buffer.active_turn.source == "mic"
        assert len(session._transcript_buffer._finalized_turns) == 1
        assert session._transcript_buffer._finalized_turns[0].source == "system"


class TestOrchestratorCallbacks:
    """Tests for orchestrator callback wiring."""

    @pytest.mark.asyncio
    async def test_on_transcription_feeds_buffer(self) -> None:
        """_on_transcription should add text to the transcript buffer."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session._on_transcription("Hello world there", 100.0)

        assert session._transcript_buffer.active_turn is not None
        assert session._transcript_buffer.active_turn.text == "Hello world there"

    @pytest.mark.asyncio
    async def test_on_silence_detected_notifies_buffer(self) -> None:
        """_on_silence_detected should trigger turn finalization in buffer."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session._on_transcription("Some speech here", 100.0)
        assert session._transcript_buffer.active_turn is not None

        session._on_silence_detected(103.0)  # Gap > turn_pause (2.0)
        assert session._transcript_buffer.active_turn is None

    @pytest.mark.asyncio
    async def test_on_trigger_result_pushes_to_prompt_queue(self) -> None:
        """_on_trigger_result should push a prompt message."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType

        session = Session()
        session._loop = asyncio.get_running_loop()

        # Mock orchestrator buffer for Q&A pair
        mock_orch = MagicMock()
        session._orchestrator = mock_orch

        trigger = Trigger(
            type=TriggerType.QUESTION,
            text="What is the timeline?",
            confidence=0.8,
        )
        result = GenerationResult(
            answer="Q2 beta release",
            trigger_type=TriggerType.QUESTION,
            confidence=0.75,
            method="hybrid",
            latency_ms=480,
            source="docs/roadmap.md",
        )

        session._on_trigger_result(trigger, result)

        await asyncio.sleep(0.02)
        msg = session._prompt_queue.get_nowait()
        assert msg["type"] == "prompt"
        assert msg["trigger_type"] == "question"
        assert msg["answer"] == "Q2 beta release"
        assert msg["confidence"] == 0.75


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
