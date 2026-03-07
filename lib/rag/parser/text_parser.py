"""
Text/markdown document parser.

Extracts heading-based sections from markdown and plain text files.
Builds heading paths like "Installation > Prerequisites" for citations.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from lib.rag.errors import ParseError
from lib.rag.types import ParsedDocument, ParsedSection

# Regex for markdown headings: ^#{1,6} followed by text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    """Approximate token count. ~1.3 tokens per whitespace-delimited word."""
    return max(1, int(len(text.split()) * 1.3))


class TextParser:
    """Parser for plain text and markdown files."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a text/markdown file into sections."""
        if not path.is_file():
            raise ParseError(f"File not found: {path}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ParseError(f"Cannot read file: {path}: {exc}") from exc

        mime_type = mimetypes.guess_type(str(path))[0] or "text/plain"
        sections = _extract_sections(text)

        return ParsedDocument(
            path=str(path.resolve()),
            filename=path.name,
            mime_type=mime_type,
            sections=sections,
            full_text=text,
            token_count=_estimate_tokens(text),
        )


def _extract_sections(text: str) -> list[ParsedSection]:
    """Split text into sections based on markdown headings."""
    headings: list[tuple[int, int, str]] = []  # (start_pos, level, title)

    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((match.start(), level, title))

    if not headings:
        # No headings found — treat entire document as one section
        return [
            ParsedSection(
                heading="",
                heading_level=0,
                heading_path="",
                content=text,
            )
        ]

    sections: list[ParsedSection] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title) for building paths

    # Content before the first heading
    preamble = text[: headings[0][0]].strip()
    if preamble:
        sections.append(
            ParsedSection(
                heading="",
                heading_level=0,
                heading_path="",
                content=preamble,
            )
        )

    for i, (start_pos, level, title) in enumerate(headings):
        # Determine content end
        if i + 1 < len(headings):
            content_end = headings[i + 1][0]
        else:
            content_end = len(text)

        # Content is everything from the heading line to the next heading
        content = text[start_pos:content_end].strip()

        # Build heading path by maintaining a stack
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        heading_path = " > ".join(t for _, t in heading_stack)

        sections.append(
            ParsedSection(
                heading=title,
                heading_level=level,
                heading_path=heading_path,
                content=content,
            )
        )

    return sections
