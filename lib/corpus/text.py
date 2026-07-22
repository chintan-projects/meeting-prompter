"""Text shaping utilities for the corpus pipeline.

`clean_markdown` reduces a raw markdown chunk to readable, *borrowable* prose —
what a speaker could say out loud. It is the display-layer cleaner; the distiller
deliberately feeds its model backends the RAW section instead (tables and code
intact), because answer content often lives in tables that this cleaner strips.
"""

from __future__ import annotations

import re


def clean_markdown(text: str) -> str:
    """Reduce a raw chunk to readable, borrowable prose.

    Drops what you would never *say* out loud — fenced code, table rows, heading
    hashes, blockquote/list markers, inline emphasis/link syntax — and collapses
    whitespace. Tables/code are removed on purpose: a chunk that's mostly table is
    not an answer-shaped unit, and the near-empty result is itself a corpus signal
    (surfaced via BORROWABLE_MIN_WORDS at the call sites).
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced code blocks
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|") or re.fullmatch(r"[-|:\s]+", line):
            continue  # table row / separator
        line = re.sub(r"^#{1,6}\s*", "", line)  # heading hashes
        line = re.sub(r"^>\s*", "", line)  # blockquote
        line = re.sub(r"^[-*+]\s+", "", line)  # bullet
        line = re.sub(r"^\d+\.\s+", "", line)  # ordered list
        kept.append(line)
    out = " ".join(kept)
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)  # bold
    out = re.sub(r"\*(.+?)\*", r"\1", out)  # italic
    out = re.sub(r"`([^`]+)`", r"\1", out)  # inline code
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)  # links → text
    return re.sub(r"\s+", " ", out).strip()
