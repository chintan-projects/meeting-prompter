"""Tests for src.api.transcript_store — append-only transcript with edit overlay."""
from src.api.transcript_store import TranscriptStore, TranscriptSegment


class TestTranscriptSegment:
    """Tests for the TranscriptSegment dataclass."""

    def test_to_dict(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=100.0, speaker="Alice")
        d = seg.to_dict()
        assert d["id"] == "seg-1"
        assert d["text"] == "hello"
        assert d["timestamp"] == 100.0
        assert d["speaker"] == "Alice"

    def test_default_speaker(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=0.0)
        assert seg.speaker == ""

    def test_default_is_final(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=0.0)
        assert seg.is_final is False

    def test_end_timestamp_defaults_to_timestamp(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=100.0)
        d = seg.to_dict()
        assert d["end_timestamp"] == 100.0

    def test_to_dict_includes_all_fields(self) -> None:
        seg = TranscriptSegment(
            id="turn-1", text="hello", timestamp=100.0,
            end_timestamp=105.0, is_final=True, speaker="Bob",
        )
        d = seg.to_dict()
        assert d == {
            "id": "turn-1",
            "text": "hello",
            "timestamp": 100.0,
            "end_timestamp": 105.0,
            "is_final": True,
            "speaker": "Bob",
        }


class TestTranscriptStore:
    """Tests for append, edit, merge, and export."""

    def test_append_returns_sequential_ids(self) -> None:
        store = TranscriptStore()
        id1 = store.append("first", timestamp=1.0)
        id2 = store.append("second", timestamp=2.0)
        assert id1 == "seg-1"
        assert id2 == "seg-2"

    def test_segment_count(self) -> None:
        store = TranscriptStore()
        assert store.segment_count == 0
        store.append("one", timestamp=1.0)
        store.append("two", timestamp=2.0)
        assert store.segment_count == 2

    def test_get_raw_returns_original(self) -> None:
        store = TranscriptStore()
        store.append("hello world", timestamp=10.0)
        raw = store.get_raw()
        assert len(raw) == 1
        assert raw[0]["text"] == "hello world"

    def test_edit_applies_overlay(self) -> None:
        store = TranscriptStore()
        seg_id = store.append("original text", timestamp=1.0)
        assert store.edit(seg_id, "corrected text") is True

        merged = store.get_merged()
        assert merged[0]["text"] == "corrected text"
        assert merged[0].get("edited") is True

        # Raw should be unchanged
        raw = store.get_raw()
        assert raw[0]["text"] == "original text"

    def test_edit_nonexistent_segment(self) -> None:
        store = TranscriptStore()
        assert store.edit("seg-999", "nope") is False

    def test_get_merged_without_edits(self) -> None:
        store = TranscriptStore()
        store.append("no edits", timestamp=1.0)
        merged = store.get_merged()
        assert merged[0]["text"] == "no edits"
        assert "edited" not in merged[0]

    def test_export_markdown(self) -> None:
        store = TranscriptStore()
        store.append("hello there", speaker="Alice", timestamp=0.0)
        store.append("general kenobi", timestamp=0.0)

        md = store.export_markdown()
        assert "**Alice**" in md
        assert "hello there" in md
        assert "general kenobi" in md

    def test_export_markdown_with_edits(self) -> None:
        store = TranscriptStore()
        seg_id = store.append("typo here", timestamp=0.0)
        store.edit(seg_id, "fixed here")

        md = store.export_markdown()
        assert "fixed here" in md
        assert "typo here" not in md


class TestUpsert:
    """Tests for the turn-based upsert method."""

    def test_upsert_creates_new_segment(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "hello world", timestamp=100.0)
        assert store.segment_count == 1
        merged = store.get_merged()
        assert merged[0]["id"] == "turn-1"
        assert merged[0]["text"] == "hello world"
        assert merged[0]["is_final"] is False

    def test_upsert_updates_existing_segment(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "hello", timestamp=100.0)
        store.upsert("turn-1", "hello world", timestamp=100.0, end_timestamp=101.0)
        assert store.segment_count == 1
        merged = store.get_merged()
        assert merged[0]["text"] == "hello world"
        assert merged[0]["end_timestamp"] == 101.0

    def test_upsert_finalizes_segment(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "partial text", timestamp=100.0)
        store.upsert("turn-1", "complete text here", timestamp=100.0, is_final=True)
        merged = store.get_merged()
        assert merged[0]["is_final"] is True
        assert merged[0]["text"] == "complete text here"

    def test_upsert_multiple_turns(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "first turn", timestamp=100.0, is_final=True)
        store.upsert("turn-2", "second turn", timestamp=103.0, is_final=False)
        assert store.segment_count == 2
        merged = store.get_merged()
        assert merged[0]["id"] == "turn-1"
        assert merged[1]["id"] == "turn-2"

    def test_upsert_with_speaker(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "hello", timestamp=100.0, speaker="Alice")
        merged = store.get_merged()
        assert merged[0]["speaker"] == "Alice"

    def test_edit_works_with_upserted_segments(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "original text", timestamp=100.0, is_final=True)
        assert store.edit("turn-1", "edited text") is True
        merged = store.get_merged()
        assert merged[0]["text"] == "edited text"
        assert merged[0].get("edited") is True

    def test_upsert_does_not_overwrite_speaker_with_empty(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "hello", timestamp=100.0, speaker="Alice")
        store.upsert("turn-1", "hello world", timestamp=100.0, speaker="")
        merged = store.get_merged()
        assert merged[0]["speaker"] == "Alice"

    def test_mixed_append_and_upsert(self) -> None:
        store = TranscriptStore()
        store.append("legacy chunk", timestamp=1.0)
        store.upsert("turn-1", "modern turn", timestamp=2.0)
        assert store.segment_count == 2
        merged = store.get_merged()
        assert merged[0]["id"] == "seg-1"
        assert merged[1]["id"] == "turn-1"

    def test_export_markdown_with_turns(self) -> None:
        store = TranscriptStore()
        store.upsert("turn-1", "What about the deployment timeline", timestamp=0.0)
        store.upsert("turn-2", "We are targeting Q2 for beta", timestamp=0.0, speaker="Bob")
        md = store.export_markdown()
        assert "deployment timeline" in md
        assert "**Bob**" in md
        assert "targeting Q2" in md
