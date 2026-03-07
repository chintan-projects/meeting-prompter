"""Tests for src.api.session — callback-based pipeline and session lifecycle."""
import asyncio
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
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


class TestPauseResume:
    """Tests for session pause/resume and timer tracking."""

    def test_initial_not_paused(self) -> None:
        """Fresh session should not be paused."""
        session = Session()
        assert session.is_paused is False

    def test_pause_sets_flag(self) -> None:
        """pause() should set is_paused when running."""
        session = Session()
        session._running = True
        session._start_time = time.time()
        session.pause()
        assert session.is_paused is True

    def test_pause_noop_when_not_running(self) -> None:
        """pause() should be a no-op when session is not running."""
        session = Session()
        session.pause()
        assert session.is_paused is False

    def test_pause_noop_when_already_paused(self) -> None:
        """pause() should be a no-op if already paused."""
        session = Session()
        session._running = True
        session._start_time = time.time()
        session.pause()
        first_pause_start = session._pause_start
        session.pause()  # Should not reset _pause_start
        assert session._pause_start == first_pause_start

    def test_resume_clears_flag(self) -> None:
        """resume() should clear the paused flag."""
        session = Session()
        session._running = True
        session._start_time = time.time()
        session.pause()
        assert session.is_paused is True
        session.resume()
        assert session.is_paused is False

    def test_resume_noop_when_not_paused(self) -> None:
        """resume() should be a no-op if not paused."""
        session = Session()
        session._running = True
        session._start_time = time.time()
        session.resume()
        assert session._total_pause_time == 0.0

    def test_elapsed_excludes_pause_time(self) -> None:
        """elapsed_seconds should exclude time spent paused."""
        session = Session()
        session._start_time = time.time() - 20.0
        session._total_pause_time = 5.0
        elapsed = session.elapsed_seconds
        assert 14.5 < elapsed < 16.0

    def test_elapsed_accounts_for_active_pause(self) -> None:
        """elapsed_seconds during active pause should exclude current pause duration."""
        session = Session()
        session._start_time = time.time() - 10.0
        session._running = True
        session._paused = True
        session._pause_start = time.time() - 3.0  # Paused 3 seconds ago
        session._total_pause_time = 0.0

        elapsed = session.elapsed_seconds
        # Total 10s, minus ~3s pause → ~7s
        assert 6.5 < elapsed < 8.0

    def test_resume_accumulates_pause_duration(self) -> None:
        """resume() should add the pause duration to _total_pause_time."""
        session = Session()
        session._running = True
        session._start_time = time.time() - 10.0
        session._paused = True
        session._pause_start = time.time() - 2.0  # Paused 2 seconds ago

        session.resume()
        assert session._total_pause_time >= 1.5  # At least ~2 seconds
        assert session._pause_start == 0.0  # Reset

    def test_pause_flushes_transcript_buffer(self) -> None:
        """pause() should flush the active turn in transcript buffer."""
        session = Session()
        session._running = True
        session._start_time = time.time()

        # Add a chunk to create an active turn
        session._on_transcription("some speech here", time.time())
        assert session._transcript_buffer.active_turn is not None

        session.pause()
        assert session._transcript_buffer.active_turn is None

    def test_pause_pauses_audio_captures(self) -> None:
        """pause() should call pause on both orchestrator audio and mic capture."""
        session = Session()
        session._running = True
        session._start_time = time.time()

        mock_orch = MagicMock()
        session._orchestrator = mock_orch
        mock_mic = MagicMock()
        session._mic_capture = mock_mic

        session.pause()
        mock_orch.audio.pause.assert_called_once()
        mock_mic.pause.assert_called_once()

    def test_resume_resumes_audio_captures(self) -> None:
        """resume() should call resume on both orchestrator audio and mic capture."""
        session = Session()
        session._running = True
        session._paused = True
        session._pause_start = time.time()
        session._start_time = time.time()

        mock_orch = MagicMock()
        session._orchestrator = mock_orch
        mock_mic = MagicMock()
        session._mic_capture = mock_mic

        session.resume()
        mock_orch.audio.resume.assert_called_once()
        mock_mic.resume.assert_called_once()

    def test_status_includes_paused(self) -> None:
        """get_status() should include the paused flag."""
        session = Session()
        status = session.get_status()
        assert "paused" in status
        assert status["paused"] is False

        session._running = True
        session._paused = True
        status = session.get_status()
        assert status["paused"] is True


class TestContextOnStart:
    """Tests for meeting context being set during session start."""

    def test_meeting_context_set_before_pipeline(self) -> None:
        """Meeting context should be available before pipeline runs."""
        from lib.conversation.meeting_context import MeetingContext

        session = Session()
        session.meeting_context = MeetingContext(
            title="Sprint Planning",
            agenda_items=["Review roadmap", "Sprint goals"],
            watch_words=["budget", "timeline"],
            participants=["Alice (PM)", "Bob (Eng)"],
        )

        assert session.meeting_context.title == "Sprint Planning"
        assert len(session.meeting_context.agenda_items) == 2
        assert "budget" in session.meeting_context.watch_words

    def test_status_shows_meeting_title(self) -> None:
        """get_status() should include the meeting title from context."""
        from lib.conversation.meeting_context import MeetingContext

        session = Session()
        session.meeting_context = MeetingContext(
            title="Design Review",
            agenda_items=[],
            watch_words=[],
            participants=[],
        )

        status = session.get_status()
        assert status["meeting_title"] == "Design Review"

    def test_no_context_shows_empty_title(self) -> None:
        """Without context, meeting_title should be empty."""
        session = Session()
        status = session.get_status()
        assert status["meeting_title"] == ""


class TestTriggerHistory:
    """Tests for trigger result accumulation (F-106)."""

    def test_initial_trigger_history_empty(self) -> None:
        """Fresh session should have empty trigger history."""
        session = Session()
        assert session.trigger_history == []

    @pytest.mark.asyncio
    async def test_trigger_result_accumulates(self) -> None:
        """_on_trigger_result should append to trigger_history."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType

        session = Session()
        session._loop = asyncio.get_running_loop()

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

        assert len(session.trigger_history) == 1
        entry = session.trigger_history[0]
        assert entry["trigger_type"] == "question"
        assert entry["trigger_text"] == "What is the timeline?"
        assert entry["answer"] == "Q2 beta release"
        assert entry["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_multiple_triggers_accumulate(self) -> None:
        """Multiple trigger results should all be stored."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType

        session = Session()
        session._loop = asyncio.get_running_loop()

        for i in range(3):
            trigger = Trigger(
                type=TriggerType.QUESTION,
                text=f"Question {i}?",
                confidence=0.7,
            )
            result = GenerationResult(
                answer=f"Answer {i}",
                trigger_type=TriggerType.QUESTION,
                confidence=0.6,
                method="hybrid",
                latency_ms=100,
                source="test",
            )
            session._on_trigger_result(trigger, result)

        assert len(session.trigger_history) == 3

    def test_trigger_history_returns_copy(self) -> None:
        """trigger_history property should return a copy, not the internal list."""
        session = Session()
        history = session.trigger_history
        history.append({"fake": True})
        assert len(session.trigger_history) == 0  # Internal list unchanged


class TestDiarizationIntegration:
    """Tests for Tier 2 neural speaker diarization in session pipeline."""

    @pytest.mark.asyncio
    async def test_relabel_speaker_updates_store_and_queue(self) -> None:
        """_relabel_speaker should upsert new speaker and emit transcript_relabeled."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        # Mock diarizer that returns "Speaker B"
        mock_diarizer = MagicMock()
        mock_diarizer.process_turn.return_value = "Speaker B"
        session._diarizer = mock_diarizer

        # Mock orchestrator with audio segment
        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = np.zeros(32000, dtype=np.float32)
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-1", text="hello from system",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system", speaker="Others",
        )
        session._relabel_speaker(turn)

        await asyncio.sleep(0.02)
        msg = session._transcript_queue.get_nowait()
        assert msg["type"] == "transcript_relabeled"
        assert msg["speaker"] == "Speaker B"
        assert msg["id"] == "turn-1"
        assert msg["source"] == "system"

        # Check transcript store was updated
        merged = session.transcript.get_merged()
        assert merged[0]["speaker"] == "Speaker B"

    @pytest.mark.asyncio
    async def test_relabel_skipped_when_speaker_unchanged(self) -> None:
        """If diarizer returns same speaker, no relabeled message emitted."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        mock_diarizer = MagicMock()
        mock_diarizer.process_turn.return_value = "Others"  # Same as initial
        session._diarizer = mock_diarizer

        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = np.zeros(32000, dtype=np.float32)
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-1", text="hello",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system", speaker="Others",
        )
        session._relabel_speaker(turn)

        await asyncio.sleep(0.02)
        assert session._transcript_queue.empty()

    @pytest.mark.asyncio
    async def test_relabel_skipped_when_no_audio(self) -> None:
        """If no audio segment available, relabeling should be skipped."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        mock_diarizer = MagicMock()
        session._diarizer = mock_diarizer

        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = None
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-1", text="hello",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system", speaker="Others",
        )
        session._relabel_speaker(turn)

        mock_diarizer.process_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_diarization_only_on_system_turns(self) -> None:
        """_on_turn_final should only diarize system turns, not mic turns."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        mock_diarizer = MagicMock()
        session._diarizer = mock_diarizer

        # Mic turn — should NOT trigger diarization
        mic_turn = Turn(
            id="turn-1", text="my speech",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="mic",
        )
        session._on_turn_final(mic_turn)

        await asyncio.sleep(0.01)
        mock_diarizer.process_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_diarization_disabled_no_relabel(self) -> None:
        """With diarizer=None, system turns keep 'Others' label."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()
        session._diarizer = None

        turn = Turn(
            id="turn-1", text="system speech",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system",
        )
        session._on_turn_final(turn)

        assert turn.speaker == "Others"
        await asyncio.sleep(0.01)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        types = [m["type"] for m in messages]
        assert "transcript_relabeled" not in types
        assert "transcript_final" in types

    @pytest.mark.asyncio
    async def test_relabel_failure_graceful(self) -> None:
        """Diarization failure should log warning, not crash."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        mock_diarizer = MagicMock()
        mock_diarizer.process_turn.side_effect = RuntimeError("model error")
        session._diarizer = mock_diarizer

        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = np.zeros(32000, dtype=np.float32)
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-1", text="hello",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system", speaker="Others",
        )
        # Should not raise
        session._relabel_speaker(turn)

        # No relabeled message should be emitted
        await asyncio.sleep(0.01)
        assert session._transcript_queue.empty()

    @pytest.mark.asyncio
    async def test_on_turn_final_with_refiner_and_diarizer(self) -> None:
        """Full pipeline: finalize → polish → diarize, all three messages emitted."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        # Mock refiner
        mock_refiner = MagicMock()
        mock_refiner.refine.return_value = "Polished system speech."
        session._text_refiner = mock_refiner

        # Mock diarizer
        mock_diarizer = MagicMock()
        mock_diarizer.process_turn.return_value = "Speaker C"
        session._diarizer = mock_diarizer

        # Mock orchestrator with audio
        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = np.zeros(32000, dtype=np.float32)
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-1", text="system speech here",
            start_timestamp=100.0, end_timestamp=104.0,
            is_final=True, source="system",
        )
        session._on_turn_final(turn)

        await asyncio.sleep(0.03)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        types = [m["type"] for m in messages]
        assert "transcript_final" in types
        assert "transcript_polished" in types
        assert "transcript_relabeled" in types

        relabeled = next(m for m in messages if m["type"] == "transcript_relabeled")
        assert relabeled["speaker"] == "Speaker C"


class TestRenameSpeaker:
    """Tests for click-to-rename speaker labels (Tier 3 lite)."""

    @pytest.mark.asyncio
    async def test_rename_updates_store_and_emits_messages(self) -> None:
        """rename_speaker should update store and emit transcript_relabeled."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        # Pre-populate transcript
        session.transcript.upsert("turn-1", "hello", 100.0, 104.0, True, "Speaker A", "system")
        session.transcript.upsert("turn-2", "world", 105.0, 108.0, True, "Speaker A", "system")
        session.transcript.upsert("turn-3", "bye", 110.0, 112.0, True, "Speaker B", "system")

        session.rename_speaker("Speaker A", "Alice")

        await asyncio.sleep(0.02)
        messages = []
        while not session._transcript_queue.empty():
            messages.append(session._transcript_queue.get_nowait())

        assert len(messages) == 2
        assert all(m["type"] == "transcript_relabeled" for m in messages)
        assert all(m["speaker"] == "Alice" for m in messages)
        assert {m["id"] for m in messages} == {"turn-1", "turn-2"}

        # Store should reflect rename
        raw = session.transcript.get_raw()
        assert raw[0]["speaker"] == "Alice"
        assert raw[1]["speaker"] == "Alice"
        assert raw[2]["speaker"] == "Speaker B"

    @pytest.mark.asyncio
    async def test_future_diarizer_uses_name_mapping(self) -> None:
        """After rename, diarizer results should resolve to custom name."""
        from src.api.transcript_buffer import Turn

        session = Session()
        session._loop = asyncio.get_running_loop()

        # Rename Speaker A → Alice
        session.transcript.upsert("turn-1", "hi", 100.0, 104.0, True, "Speaker A", "system")
        session.rename_speaker("Speaker A", "Alice")

        # Drain rename messages
        await asyncio.sleep(0.02)
        while not session._transcript_queue.empty():
            session._transcript_queue.get_nowait()

        # Mock diarizer returns "Speaker A" for a new turn
        mock_diarizer = MagicMock()
        mock_diarizer.process_turn.return_value = "Speaker A"
        session._diarizer = mock_diarizer

        mock_orch = MagicMock()
        mock_orch.audio.get_audio_segment.return_value = np.zeros(32000, dtype=np.float32)
        session._orchestrator = mock_orch

        turn = Turn(
            id="turn-2", text="new system speech",
            start_timestamp=200.0, end_timestamp=204.0,
            is_final=True, source="system", speaker="Others",
        )
        session._relabel_speaker(turn)

        await asyncio.sleep(0.02)
        msg = session._transcript_queue.get_nowait()
        assert msg["speaker"] == "Alice"

    @pytest.mark.asyncio
    async def test_rename_others_works(self) -> None:
        """Can rename the default 'Others' label too."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session.transcript.upsert("turn-1", "hi", 100.0, 104.0, True, "Others", "system")
        session.rename_speaker("Others", "External")

        await asyncio.sleep(0.02)
        msg = session._transcript_queue.get_nowait()
        assert msg["speaker"] == "External"
        assert session.transcript.get_raw()[0]["speaker"] == "External"

    @pytest.mark.asyncio
    async def test_rename_no_match_is_noop(self) -> None:
        """Renaming a non-existent speaker should not emit messages."""
        session = Session()
        session._loop = asyncio.get_running_loop()

        session.transcript.upsert("turn-1", "hi", 100.0, 104.0, True, "Speaker A", "system")
        session.rename_speaker("Speaker Z", "Zach")

        await asyncio.sleep(0.02)
        assert session._transcript_queue.empty()


class TestConfigWiring:
    """Tests for config-driven TranscriptBuffer parameters (F-105)."""

    def test_default_buffer_params(self) -> None:
        """Default config should wire turn_pause=2.0, max_turn_duration=30.0."""
        session = Session()
        assert session._transcript_buffer._turn_pause == 2.0
        assert session._transcript_buffer._max_turn_duration == 30.0

    def test_custom_buffer_params_from_config(self) -> None:
        """Custom config values should be wired to TranscriptBuffer."""
        from lib.config import AppConfig, load_config

        config = load_config()
        config.buffer.turn_pause = 3.5
        config.buffer.max_turn_duration = 45.0

        session = Session(config=config)
        assert session._transcript_buffer._turn_pause == 3.5
        assert session._transcript_buffer._max_turn_duration == 45.0
