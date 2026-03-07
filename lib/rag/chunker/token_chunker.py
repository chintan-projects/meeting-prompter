"""
Token-based chunker.

Splits sections into chunks of at most max_chunk_tokens tokens.
Respects paragraph boundaries first, then sentence boundaries.
Configurable overlap between consecutive chunks.
"""

from __future__ import annotations

import re

from lib.rag.config import RAGConfig
from lib.rag.types import ChunkOutput, ParsedSection

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


class TokenChunker:
    """Chunk sections into token-bounded pieces."""

    def chunk(
        self, sections: list[ParsedSection], config: RAGConfig
    ) -> list[ChunkOutput]:
        """Split all sections into chunks."""
        max_tokens = config.max_chunk_tokens
        overlap_tokens = config.chunk_overlap_tokens
        results: list[ChunkOutput] = []

        for section_idx, section in enumerate(sections):
            section_chunks = _chunk_section(
                section.content, max_tokens, overlap_tokens
            )
            for chunk_text in section_chunks:
                results.append(
                    ChunkOutput(
                        content=chunk_text,
                        token_count=_estimate_tokens(chunk_text),
                        section_index=section_idx,
                    )
                )

        return results


def _chunk_section(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split a section's text into token-bounded chunks."""
    if not text.strip():
        return []

    tokens_est = _estimate_tokens(text)
    if tokens_est <= max_tokens:
        return [text.strip()]

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if para_tokens > max_tokens:
            # Flush current buffer
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = _get_overlap_parts(current_parts, overlap_tokens)
                current_tokens = sum(_estimate_tokens(p) for p in current_parts)

            # Split oversized paragraph by sentences
            for sentence_chunk in _split_by_sentences(para, max_tokens):
                chunks.append(sentence_chunk)
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = _get_overlap_parts(current_parts, overlap_tokens)
            current_tokens = sum(_estimate_tokens(p) for p in current_parts)

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks if chunks else [text.strip()]


def _split_by_sentences(text: str, max_tokens: int) -> list[str]:
    """Split text by sentence boundaries when paragraphs are too large."""
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _get_overlap_parts(parts: list[str], overlap_tokens: int) -> list[str]:
    """Return trailing parts that fit within the overlap token budget."""
    if overlap_tokens <= 0:
        return []

    result: list[str] = []
    tokens = 0
    for part in reversed(parts):
        part_tokens = _estimate_tokens(part)
        if tokens + part_tokens > overlap_tokens:
            break
        result.insert(0, part)
        tokens += part_tokens

    return result
