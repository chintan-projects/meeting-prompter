"""Tests for src.api.transcript_store — append-only transcript with edit overlay."""
import pytest

from src.api.transcript_store import TranscriptStore, TranscriptSegment


class TestTranscriptSegment:
    """Tests for the TranscriptSegment dataclass."""

    def test_to_dict(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=100.0, speaker="Alice")
        d = seg.to_dict()
        assert d == {"id": "seg-1", "text": "hello", "timestamp": 100.0, "speaker": "Alice"}

    def test_default_speaker(self) -> None:
        seg = TranscriptSegment(id="seg-1", text="hello", timestamp=0.0)
        assert seg.speaker == ""


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
