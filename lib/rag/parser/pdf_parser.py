"""PDF document parser using pypdf.

Extracts text from PDF pages and builds section structure.
Each page becomes a ParsedSection with page number metadata
for citation support. pypdf is already in requirements.txt.
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from lib.rag.errors import ParseError
from lib.rag.types import ParsedDocument, ParsedSection


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


class PdfParser:
    """Parser for PDF files. Implements DocumentParser protocol.

    Each PDF page becomes a section with start_page/end_page
    populated for downstream citation generation.
    """

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a PDF file into structured sections by page."""
        if not path.is_file():
            raise ParseError(f"File not found: {path}")

        try:
            reader = PdfReader(path)
        except Exception as exc:
            raise ParseError(f"Cannot read PDF: {path}: {exc}") from exc

        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        sections: list[ParsedSection] = []
        for i, page_text in enumerate(pages_text):
            cleaned = re.sub(r"\s+", " ", page_text).strip()
            if not cleaned:
                continue
            sections.append(
                ParsedSection(
                    heading=f"Page {i + 1}",
                    heading_level=1,
                    heading_path=f"Page {i + 1}",
                    content=cleaned,
                    start_page=i + 1,
                    end_page=i + 1,
                )
            )

        if not sections:
            sections.append(
                ParsedSection(
                    heading="",
                    heading_level=0,
                    heading_path="",
                    content=full_text or "",
                    start_page=None,
                    end_page=None,
                )
            )

        return ParsedDocument(
            path=str(path.resolve()),
            filename=path.name,
            mime_type="application/pdf",
            sections=sections,
            full_text=full_text,
            token_count=_estimate_tokens(full_text),
        )
