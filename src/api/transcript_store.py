"""Transcript store — append-only raw transcript with edit overlay.

The raw transcript is immutable (append-only log from ASR).
User edits are stored as a separate overlay keyed by segment ID.
The merged view combines both for display and export.

Supports two modes:
- append(): legacy per-chunk storage (seg-N IDs)
- upsert(): turn-based storage (turn-N IDs) — creates or updates segments
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech (maps to a turn or chunk)."""

    id: str
    text: str
    timestamp: float
    speaker: str = ""
    source: str = ""
    end_timestamp: float = 0.0
    is_final: bool = False

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "source": self.source,
            "end_timestamp": self.end_timestamp or self.timestamp,
            "is_final": self.is_final,
        }


class TranscriptStore:
    """Append-only transcript with user edit overlay.

    Raw segments are immutable. Edits are stored separately so we can
    always reconstruct the original transcript.
    """

    def __init__(self) -> None:
        self._segments: List[TranscriptSegment] = []
        self._edits: Dict[str, str] = {}  # segment_id -> edited text
        self._counter: int = 0
        self._index: Dict[str, int] = {}  # segment_id -> list index (for upsert)

    def append(self, text: str, speaker: str = "", timestamp: Optional[float] = None) -> str:
        """Append a new transcript segment. Returns segment ID."""
        self._counter += 1
        seg_id = f"seg-{self._counter}"
        segment = TranscriptSegment(
            id=seg_id,
            text=text,
            timestamp=timestamp or time.time(),
            speaker=speaker,
            is_final=True,
        )
        self._index[seg_id] = len(self._segments)
        self._segments.append(segment)
        return seg_id

    def upsert(
        self,
        seg_id: str,
        text: str,
        timestamp: float,
        end_timestamp: float = 0.0,
        is_final: bool = False,
        speaker: str = "",
        source: str = "",
    ) -> str:
        """Create or update a transcript segment (turn-based).

        If seg_id already exists, updates its text and metadata.
        If not, creates a new segment with the given ID.
        Returns the segment ID.
        """
        if seg_id in self._index:
            idx = self._index[seg_id]
            seg = self._segments[idx]
            seg.text = text
            seg.end_timestamp = end_timestamp or timestamp
            seg.is_final = is_final
            if speaker:
                seg.speaker = speaker
            if source:
                seg.source = source
        else:
            segment = TranscriptSegment(
                id=seg_id,
                text=text,
                timestamp=timestamp,
                end_timestamp=end_timestamp or timestamp,
                is_final=is_final,
                speaker=speaker,
                source=source,
            )
            self._index[seg_id] = len(self._segments)
            self._segments.append(segment)
        return seg_id

    def rename_speaker(self, old_name: str, new_name: str) -> List[str]:
        """Rename all segments with old_name speaker to new_name.

        Returns list of affected segment IDs.
        """
        affected: List[str] = []
        for seg in self._segments:
            if seg.speaker == old_name:
                seg.speaker = new_name
                affected.append(seg.id)
        return affected

    def edit(self, segment_id: str, new_text: str) -> bool:
        """Apply an edit overlay to a segment. Returns True if segment exists."""
        if segment_id in self._index:
            self._edits[segment_id] = new_text
            return True
        # Fallback: linear search for legacy compatibility
        for seg in self._segments:
            if seg.id == segment_id:
                self._edits[segment_id] = new_text
                return True
        return False

    def get_merged(self) -> List[dict]:
        """Get all segments with edits applied."""
        result = []
        for seg in self._segments:
            entry = seg.to_dict()
            if seg.id in self._edits:
                entry["text"] = self._edits[seg.id]
                entry["edited"] = True
            result.append(entry)
        return result

    def get_raw(self) -> List[dict]:
        """Get original unedited segments."""
        return [seg.to_dict() for seg in self._segments]

    def export_json(self) -> List[dict]:
        """Export merged transcript as structured JSON with edit status.

        Returns a list of segment dicts suitable for machine consumption.
        Each segment includes an explicit 'edited' field. Meeting-level
        metadata (title, duration) is added by the caller.
        """
        result = []
        for seg in self._segments:
            entry = seg.to_dict()
            if seg.id in self._edits:
                entry["text"] = self._edits[seg.id]
                entry["edited"] = True
            else:
                entry["edited"] = False
            result.append(entry)
        return result

    def export_markdown(self) -> str:
        """Export merged transcript as markdown.

        Each turn becomes a timestamped paragraph, producing readable
        meeting notes rather than fragmented per-chunk output.
        """
        lines: List[str] = []
        for seg in self._segments:
            text = self._edits.get(seg.id, seg.text)
            ts = time.strftime("%H:%M:%S", time.localtime(seg.timestamp))
            prefix = f"**{seg.speaker}** " if seg.speaker else ""
            lines.append(f"[{ts}] {prefix}{text}")
        return "\n\n".join(lines)

    @property
    def segment_count(self) -> int:
        """Number of segments (turns) in the transcript."""
        return len(self._segments)
