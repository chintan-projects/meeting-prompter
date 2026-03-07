"""
Document parser protocol.

Defines the interface for parsing files into structured sections.
Implementations: TextParser (markdown/plaintext).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from lib.rag.types import ParsedDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for document parsers. Implement to add new file formats."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a file into a structured document with sections."""
        ...
