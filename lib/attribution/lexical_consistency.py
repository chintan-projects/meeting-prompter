"""Lexical speaker-consistency pass (F-607).

A cheap, model-free post-hoc pass over (attribution hypothesis + transcript) that
catches named turn-taking cues the acoustic/channel layers miss:

  * hand-off cues ("over to you, Priya" / "Raj, what do you think?") name the
    speaker of the NEXT remote turn,
  * gratitude cues ("thanks, Priya") name the speaker of the PREVIOUS remote turn.

It is a CORRECTION LAYER, not the attribution mechanism: it only proposes
relabels (with a reason + confidence) for generically-labeled remote turns
("Others" / "Speaker X"), scoped to names on the meeting roster. The caller
decides whether to apply them. Conservative by design — it never overrides a
name that is already set to a roster member.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence


class TurnLike(Protocol):
    id: str
    text: str
    speaker: str
    source: str


@dataclass
class SpeakerCorrection:
    """A proposed relabel for one turn."""

    turn_id: str
    suggested_speaker: str
    reason: str
    confidence: float


# Cues where the named person speaks NEXT (hand-off).
_HANDOFF_PATTERNS = [
    r"over to you,?\s+{name}",
    r"go ahead,?\s+{name}",
    r"{name},?\s+what do you think",
    r"{name},?\s+your thoughts",
    r"{name},?\s+do you\b",
    r"{name},?\s+can you\b",
    r"what about you,?\s+{name}",
]

# Cues where the named person spoke PREVIOUSLY (gratitude / acknowledgement).
_GRATITUDE_PATTERNS = [
    r"thanks,?\s+{name}",
    r"thank you,?\s+{name}",
    r"good point,?\s+{name}",
]


def _first_name(roster_entry: str) -> str:
    """'Priya (Eng)' -> 'priya'. Lowercased bare first token."""
    head = roster_entry.split("(")[0].strip()
    return head.split()[0].lower() if head.split() else ""


def _is_generic(speaker: str) -> bool:
    """A relabel-eligible label: the Others bucket or an anonymous Speaker N."""
    s = speaker.strip()
    return s in ("", "Others", "Others (room)") or s.startswith("Speaker ")


class LexicalConsistencyPass:
    """Proposes name corrections from lexical turn-taking cues."""

    def __init__(self, roster: Optional[List[str]] = None) -> None:
        # first-name (lower) → display name from the roster entry
        self._names: Dict[str, str] = {}
        for entry in roster or []:
            fn = _first_name(entry)
            if fn:
                self._names[fn] = entry.split("(")[0].strip()

    def _cued_name(self, text: str, patterns: List[str]) -> Optional[str]:
        low = text.lower()
        for fn, display in self._names.items():
            for pat in patterns:
                if re.search(pat.format(name=re.escape(fn)), low):
                    return display
        return None

    def analyze(self, turns: Sequence[TurnLike]) -> List[SpeakerCorrection]:
        """Return proposed relabels for generically-labeled remote turns."""
        corrections: List[SpeakerCorrection] = []
        if not self._names:
            return corrections

        for i, turn in enumerate(turns):
            text = getattr(turn, "text", "") or ""

            # Hand-off: the named person speaks in the NEXT remote turn.
            handoff = self._cued_name(text, _HANDOFF_PATTERNS)
            if handoff and i + 1 < len(turns):
                nxt = turns[i + 1]
                if nxt.source == "system" and _is_generic(nxt.speaker):
                    corrections.append(
                        SpeakerCorrection(
                            turn_id=nxt.id,
                            suggested_speaker=handoff,
                            reason=f"hand-off cue to {handoff} in prior turn",
                            confidence=0.6,
                        )
                    )

            # Gratitude: the named person spoke in the PREVIOUS remote turn.
            gratitude = self._cued_name(text, _GRATITUDE_PATTERNS)
            if gratitude and i - 1 >= 0:
                prev = turns[i - 1]
                if prev.source == "system" and _is_generic(prev.speaker):
                    corrections.append(
                        SpeakerCorrection(
                            turn_id=prev.id,
                            suggested_speaker=gratitude,
                            reason=f"gratitude cue to {gratitude} in following turn",
                            confidence=0.6,
                        )
                    )

        return corrections


@dataclass
class _SegView:
    """Adapt an export_json() segment dict to the TurnLike surface."""

    id: str
    text: str
    speaker: str
    source: str


def correct_segments(
    segments: List[Dict[str, Any]], roster: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """Apply lexical corrections to a copy of transcript segments (non-destructive).

    Corrected turns are flagged ``low_confidence`` so downstream (notes) reflects
    that the name came from a lexical cue, not ground truth. Returns the original
    list unchanged when there is no roster or nothing to correct.
    """
    if not roster or not segments:
        return segments
    turns = [
        _SegView(
            id=str(s.get("id", "")),
            text=s.get("text", "") or "",
            speaker=s.get("speaker", "") or "",
            source=s.get("source", "") or "",
        )
        for s in segments
    ]
    corrections = {c.turn_id: c for c in LexicalConsistencyPass(roster).analyze(turns)}
    if not corrections:
        return segments
    out: List[Dict[str, Any]] = []
    for s in segments:
        c = corrections.get(str(s.get("id", "")))
        if c is not None:
            s = {**s, "speaker": c.suggested_speaker, "low_confidence": True}
        out.append(s)
    return out
