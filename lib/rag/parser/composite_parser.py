"""Composite document parser that delegates by file extension.

Routes .pdf to PdfParser, everything else to TextParser.
Satisfies the DocumentParser protocol.
"""
from __future__ import annotations

from pathlib import Path

from lib.rag.errors import ParseError
from lib.rag.parser.pdf_parser import PdfParser
from lib.rag.parser.text_parser import TextParser
from lib.rag.types import ParsedDocument


class CompositeParser:
    """Routes parsing to the appropriate parser by file extension."""

    def __init__(self) -> None:
        self._text_parser = TextParser()
        self._pdf_parser = PdfParser()

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a file, delegating by extension."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf_parser.parse(path)
        if suffix in {".md", ".txt", ".markdown"}:
            return self._text_parser.parse(path)
        raise ParseError(f"Unsupported file type: {suffix}")
