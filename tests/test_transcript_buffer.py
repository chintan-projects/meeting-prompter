"""Tests for src.api.transcript_buffer — silence-based turn accumulation.

Key design: Turn boundaries are detected by silence events from the audio
capture layer, NOT by time gaps between add_chunk() calls. The ~3.5s gap
between consecutive ASR chunks is a pipeline artifact, not a speech pause.
"""
from typing import List

from src.api.transcript_buffer import TranscriptBuffer, Turn


class TestTurnCreation:
    """Test basic turn creation and accumulation."""

    def test_first_chunk_creates_turn(self) -> None:
        buf = TranscriptBuffer()
        turn = buf.add_chunk("Hello world", timestamp=100.0)
        assert turn is not None
        assert turn.id == "turn-1"
        assert turn.text == "Hello world"
        assert turn.start_timestamp == 100.0
        assert turn.end_timestamp == 100.0
        assert turn.is_final is False
        assert turn.chunk_count == 1

    def test_consecutive_chunks_merge_regardless_of_gap(self) -> None:
        """Chunks merge into same turn even with large time gaps (no silence event)."""
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Hello", timestamp=100.0)
        # 3.5s gap — typical ASR pipeline cadence. Without silence event, they merge.
        turn = buf.add_chunk("world", timestamp=103.5)
        assert turn is not None
        assert turn.id == "turn-1"
        assert turn.text == "Hello world"
        assert turn.chunk_count == 2
        assert turn.start_timestamp == 100.0
        assert turn.end_timestamp == 103.5

    def test_many_chunks_merge_in_continuous_speech(self) -> None:
        """Simulates real pipeline: 8 chunks at 3.5s intervals, no silence."""
        buf = TranscriptBuffer(turn_pause=2.0)
        for i in range(8):
            turn = buf.add_chunk(f"chunk {i}", timestamp=100.0 + i * 3.5)
        assert turn is not None
        assert turn.id == "turn-1"
        assert turn.chunk_count == 8
        assert len(buf._finalized_turns) == 0

    def test_empty_text_ignored(self) -> None:
        buf = TranscriptBuffer()
        assert buf.add_chunk("", timestamp=100.0) is None
        assert buf.add_chunk("  ", timestamp=101.0) is None
        assert buf.turn_count == 0

    def test_whitespace_stripped(self) -> None:
        buf = TranscriptBuffer()
        turn = buf.add_chunk("  hello  ", timestamp=100.0)
        assert turn is not None
        assert turn.text == "hello"


class TestSilenceBasedBoundaries:
    """Test turn finalization via silence events from audio capture."""

    def test_silence_between_chunks_creates_new_turn(self) -> None:
        """Silence event between speech chunks should finalize and start new turn."""
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("First sentence here", timestamp=100.0)

        # Audio capture detects silence (gap > turn_pause -> immediate finalize)
        buf.on_silence(timestamp=103.0)
        assert buf.active_turn is None
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].is_final is True

        # Next speech chunk starts a new turn
        turn = buf.add_chunk("Second sentence here", timestamp=106.0)
        assert turn is not None
        assert turn.id == "turn-2"
        assert turn.text == "Second sentence here"

    def test_silence_flag_causes_finalization_on_next_chunk(self) -> None:
        """Even if on_silence gap < turn_pause, the flag forces finalization."""
        buf = TranscriptBuffer(turn_pause=5.0)
        buf.add_chunk("Some speech here", timestamp=100.0)

        # Silence detected but gap (1.5s) < turn_pause (5.0s) -> no immediate finalize
        result = buf.on_silence(timestamp=101.5)
        assert result is None
        assert buf.active_turn is not None

        # But the flag is set, so next add_chunk finalizes
        turn = buf.add_chunk("New speech here", timestamp=105.0)
        assert turn is not None
        assert turn.id == "turn-2"
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].text == "Some speech here"

    def test_no_silence_means_no_finalization(self) -> None:
        """Without silence events, chunks always merge (pipeline cadence artifact)."""
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Hello there", timestamp=100.0)
        # 7s gap but no silence event — still merges
        turn = buf.add_chunk("general Kenobi", timestamp=107.0)
        assert turn is not None
        assert turn.id == "turn-1"
        assert turn.text == "Hello there general Kenobi"
        assert len(buf._finalized_turns) == 0

    def test_max_duration_forces_finalization(self) -> None:
        """Max duration still works as a safety limit regardless of silence."""
        buf = TranscriptBuffer(turn_pause=2.0, max_turn_duration=10.0)
        buf.add_chunk("First chunk here", timestamp=100.0)
        buf.add_chunk("second chunk here", timestamp=103.5)
        buf.add_chunk("third chunk here", timestamp=107.0)
        # Duration 100->110.5 = 10.5s > max_turn_duration 10s -> finalize
        turn = buf.add_chunk("fourth chunk here", timestamp=110.5)
        assert turn is not None
        assert turn.id == "turn-2"
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].chunk_count == 3

    def test_silence_at_exact_threshold_finalizes(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("First turn text", timestamp=100.0)
        # Exactly 2.0s gap -> should finalize
        result = buf.on_silence(timestamp=102.0)
        assert result is not None
        assert result.id == "turn-1"
        assert result.is_final is True

    def test_multiple_silence_events_are_idempotent(self) -> None:
        """Multiple silence events after finalization don't cause issues."""
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Some speech here", timestamp=100.0)
        buf.on_silence(timestamp=103.0)  # Finalizes
        buf.on_silence(timestamp=106.0)  # No-op (no active turn)
        buf.on_silence(timestamp=109.0)  # No-op
        assert len(buf._finalized_turns) == 1

    def test_silence_flag_reset_on_speech(self) -> None:
        """After add_chunk, silence flag is reset."""
        buf = TranscriptBuffer(turn_pause=5.0)
        buf.add_chunk("First speech", timestamp=100.0)
        buf.on_silence(timestamp=101.0)  # Gap < pause, sets flag only
        buf.add_chunk("Second speech", timestamp=104.0)  # Sees flag, finalizes, resets

        # Now add another chunk — should merge (flag was reset)
        turn = buf.add_chunk("Third speech", timestamp=107.0)
        assert turn.id == "turn-2"
        assert turn.text == "Second speech Third speech"


class TestOnSilence:
    """Test silence-triggered finalization."""

    def test_silence_finalizes_active_turn(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Some speech here", timestamp=100.0)
        result = buf.on_silence(timestamp=102.5)
        assert result is not None
        assert result.id == "turn-1"
        assert result.is_final is True
        assert buf.active_turn is None

    def test_silence_below_threshold_keeps_turn(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Some speech here", timestamp=100.0)
        result = buf.on_silence(timestamp=101.5)
        assert result is None
        assert buf.active_turn is not None

    def test_silence_with_no_active_turn(self) -> None:
        buf = TranscriptBuffer()
        result = buf.on_silence(timestamp=100.0)
        assert result is None


class TestFlush:
    """Test force-flush behavior."""

    def test_flush_finalizes_active_turn(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("Active speech here", timestamp=100.0)
        result = buf.flush()
        assert result is not None
        assert result.is_final is True
        assert buf.active_turn is None

    def test_flush_with_no_active_turn(self) -> None:
        buf = TranscriptBuffer()
        assert buf.flush() is None

    def test_flush_keeps_short_turns(self) -> None:
        """Short turns are now emitted — no min_words discard."""
        buf = TranscriptBuffer()
        buf.add_chunk("hi", timestamp=100.0)
        result = buf.flush()
        assert result is not None
        assert result.text == "hi"
        assert result.is_final is True


class TestMinWordsDisabled:
    """Verify that min_turn_words=0 (default) emits all turns."""

    def test_short_turn_emitted_on_silence(self) -> None:
        """Short turns are now emitted, not discarded."""
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("ok", timestamp=100.0)
        buf.on_silence(timestamp=103.0)

        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].text == "ok"
        assert buf._finalized_turns[0].is_final is True

    def test_single_word_turn_emitted(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("Yeah", timestamp=100.0)
        buf.on_silence(timestamp=103.0)
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].text == "Yeah"

    def test_long_enough_turn_also_kept(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("hello world", timestamp=100.0)
        buf.on_silence(timestamp=103.0)
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].text == "hello world"

    def test_explicit_min_words_no_longer_gates(self) -> None:
        """min_turn_words param exists for compat but _finalize_active no longer checks it."""
        buf = TranscriptBuffer(turn_pause=2.0, min_turn_words=3)
        buf.add_chunk("ok", timestamp=100.0)
        buf.on_silence(timestamp=103.0)
        # Short turns are emitted regardless
        assert len(buf._finalized_turns) == 1


class TestCallbacks:
    """Test on_update and on_final callbacks."""

    def test_on_update_fires_on_each_chunk(self) -> None:
        updates: List[Turn] = []
        buf = TranscriptBuffer(on_update=lambda t: updates.append(Turn(**{
            "id": t.id, "text": t.text, "start_timestamp": t.start_timestamp,
            "end_timestamp": t.end_timestamp, "is_final": t.is_final,
            "chunk_count": t.chunk_count,
        })))
        buf.add_chunk("first chunk here", timestamp=100.0)
        buf.add_chunk("second chunk here", timestamp=103.5)
        assert len(updates) == 2
        assert updates[0].text == "first chunk here"
        assert updates[1].text == "first chunk here second chunk here"

    def test_on_final_fires_on_silence(self) -> None:
        finals: List[Turn] = []
        buf = TranscriptBuffer(
            turn_pause=2.0,
            on_final=lambda t: finals.append(Turn(**{
                "id": t.id, "text": t.text, "start_timestamp": t.start_timestamp,
                "end_timestamp": t.end_timestamp, "is_final": t.is_final,
                "chunk_count": t.chunk_count,
            })),
        )
        buf.add_chunk("First turn text", timestamp=100.0)
        buf.on_silence(timestamp=103.0)  # Silence triggers finalization
        assert len(finals) == 1
        assert finals[0].id == "turn-1"
        assert finals[0].is_final is True

    def test_on_final_fires_for_short_turns(self) -> None:
        """Short turns now fire on_final callback (no discard)."""
        finals: List[Turn] = []
        buf = TranscriptBuffer(
            turn_pause=2.0,
            on_final=lambda t: finals.append(t),
        )
        buf.add_chunk("ok", timestamp=100.0)
        buf.on_silence(timestamp=103.0)
        assert len(finals) == 1
        assert finals[0].text == "ok"


class TestAllTurns:
    """Test the all_turns property."""

    def test_includes_finalized_and_active(self) -> None:
        buf = TranscriptBuffer(turn_pause=2.0)
        buf.add_chunk("First complete turn", timestamp=100.0)
        buf.on_silence(timestamp=103.0)  # Finalize turn-1
        buf.add_chunk("Second active turn", timestamp=106.0)

        turns = buf.all_turns
        assert len(turns) == 2
        assert turns[0].is_final is True
        assert turns[1].is_final is False

    def test_empty_buffer(self) -> None:
        buf = TranscriptBuffer()
        assert buf.all_turns == []
        assert buf.turn_count == 0


class TestReset:
    """Test buffer reset."""

    def test_reset_clears_all_state(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("some text here", timestamp=100.0)
        buf.flush()
        buf.add_chunk("more text here", timestamp=200.0)
        buf.reset()

        assert buf.active_turn is None
        assert buf.all_turns == []
        assert buf.turn_count == 0

    def test_reset_allows_new_turns(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("old turn text", timestamp=100.0)
        buf.reset()
        turn = buf.add_chunk("new turn text", timestamp=200.0)
        assert turn is not None
        assert turn.id == "turn-1"  # counter resets

    def test_reset_clears_silence_flag(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("some text here", timestamp=100.0)
        buf.on_silence(timestamp=101.0)  # Sets flag
        buf.reset()
        # After reset, no flag — chunk creates turn-1
        turn = buf.add_chunk("new text here", timestamp=200.0)
        assert turn.id == "turn-1"
        # Second chunk should merge (no silence, no flag)
        turn = buf.add_chunk("more text", timestamp=203.5)
        assert turn.id == "turn-1"
        assert turn.chunk_count == 2


class TestRealisticPipeline:
    """Integration tests simulating real ASR pipeline timing."""

    def test_three_turn_conversation_with_silence(self) -> None:
        """Realistic: speech -> silence -> speech -> silence -> speech."""
        finals: List[Turn] = []
        buf = TranscriptBuffer(
            turn_pause=2.0,
            max_turn_duration=30.0,
            on_final=lambda t: finals.append(t),
        )

        # Turn 1: two chunks of continuous speech (3.5s apart, no silence)
        buf.add_chunk("What about the deployment", timestamp=100.0)
        buf.add_chunk("timeline for Edge SDK", timestamp=103.5)

        # Silence detected -> turn 1 finalized
        buf.on_silence(timestamp=107.0)

        # Turn 2: response (two chunks)
        buf.add_chunk("We are targeting Q2", timestamp=110.0)
        buf.add_chunk("for the beta release", timestamp=113.5)

        # Silence detected -> turn 2 finalized
        buf.on_silence(timestamp=117.0)

        # Turn 3: follow-up (one chunk)
        buf.add_chunk("And what about compliance", timestamp=120.0)

        # Flush remaining
        buf.flush()

        assert len(finals) == 3
        assert "deployment" in finals[0].text
        assert "timeline" in finals[0].text
        assert finals[0].chunk_count == 2

        assert "targeting Q2" in finals[1].text
        assert finals[1].chunk_count == 2

        assert "compliance" in finals[2].text
        assert finals[2].chunk_count == 1

    def test_continuous_speech_stays_as_one_turn(self) -> None:
        """6 chunks at pipeline cadence (3.5s each) with no silence -> one turn."""
        buf = TranscriptBuffer(turn_pause=2.0)
        for i in range(6):
            buf.add_chunk(f"word{i} extra", timestamp=100.0 + i * 3.5)

        assert buf.turn_count == 1
        assert buf.active_turn is not None
        assert buf.active_turn.chunk_count == 6
        assert len(buf._finalized_turns) == 0

    def test_to_dict_serialization(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("Hello world there", timestamp=100.0)
        turn = buf.active_turn
        assert turn is not None

        d = turn.to_dict()
        assert d["id"] == "turn-1"
        assert d["text"] == "Hello world there"
        assert d["timestamp"] == 100.0
        assert d["is_final"] is False
        assert d["speaker"] == ""
        assert d["source"] == ""


class TestSourceField:
    """Tests for source field (dual-stream support)."""

    def test_source_passed_to_turn(self) -> None:
        buf = TranscriptBuffer()
        turn = buf.add_chunk("hello", timestamp=100.0, source="mic")
        assert turn is not None
        assert turn.source == "mic"

    def test_source_in_to_dict(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("hello", timestamp=100.0, source="system")
        d = buf.active_turn.to_dict()
        assert d["source"] == "system"

    def test_same_source_chunks_merge(self) -> None:
        buf = TranscriptBuffer()
        buf.add_chunk("hello", timestamp=100.0, source="mic")
        turn = buf.add_chunk("world", timestamp=103.5, source="mic")
        assert turn.id == "turn-1"
        assert turn.text == "hello world"
        assert turn.source == "mic"

    def test_source_change_finalizes_turn(self) -> None:
        """Changing source should finalize the active turn and start a new one."""
        buf = TranscriptBuffer()
        buf.add_chunk("system speech here", timestamp=100.0, source="system")
        turn = buf.add_chunk("mic speech here", timestamp=103.5, source="mic")
        assert turn.id == "turn-2"
        assert turn.source == "mic"
        assert len(buf._finalized_turns) == 1
        assert buf._finalized_turns[0].source == "system"

    def test_source_change_back_and_forth(self) -> None:
        """Multiple source changes should create distinct turns."""
        buf = TranscriptBuffer()
        buf.add_chunk("system turn one", timestamp=100.0, source="system")
        buf.add_chunk("mic turn one", timestamp=103.5, source="mic")
        buf.add_chunk("system turn two", timestamp=107.0, source="system")
        assert len(buf._finalized_turns) == 2
        assert buf._finalized_turns[0].source == "system"
        assert buf._finalized_turns[1].source == "mic"
        assert buf.active_turn.source == "system"

    def test_empty_source_doesnt_trigger_change(self) -> None:
        """If source is empty, it should not trigger source-change finalization."""
        buf = TranscriptBuffer()
        buf.add_chunk("first", timestamp=100.0, source="mic")
        turn = buf.add_chunk("second", timestamp=103.5, source="")
        # Empty source should not cause finalization
        assert turn.id == "turn-1"
        assert len(buf._finalized_turns) == 0
