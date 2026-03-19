"""Bidirectional conversion between Notion blocks and markdown.

``blocks_to_markdown``  — Notion block tree → markdown string (for RAG ingestion).
``markdown_to_blocks``  — markdown string → Notion block list (for export).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ── Notion blocks → Markdown ──────────────────────────────────────────────────


def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    """Convert a Notion block tree to a markdown string."""
    lines: List[str] = []
    _convert_blocks(blocks, lines, indent=0)
    return "\n".join(lines).strip()


def _convert_blocks(
    blocks: List[Dict[str, Any]],
    lines: List[str],
    indent: int,
) -> None:
    """Recursively walk blocks, appending markdown lines."""
    prefix = "    " * indent
    for block in blocks:
        block_type = block.get("type", "")
        data = block.get(block_type, {})

        if block_type in ("heading_1", "heading_2", "heading_3"):
            level = int(block_type[-1])
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{'#' * level} {text}")
            lines.append("")

        elif block_type == "paragraph":
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{prefix}{text}")
            lines.append("")

        elif block_type == "bulleted_list_item":
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{prefix}- {text}")
            children = block.get("children", [])
            if children:
                _convert_blocks(children, lines, indent + 1)

        elif block_type == "numbered_list_item":
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{prefix}1. {text}")
            children = block.get("children", [])
            if children:
                _convert_blocks(children, lines, indent + 1)

        elif block_type == "to_do":
            text = _rich_text_to_md(data.get("rich_text", []))
            checked = data.get("checked", False)
            marker = "[x]" if checked else "[ ]"
            lines.append(f"{prefix}- {marker} {text}")

        elif block_type == "code":
            text = _rich_text_to_md(data.get("rich_text", []))
            lang = data.get("language", "")
            lines.append(f"{prefix}```{lang}")
            lines.append(f"{prefix}{text}")
            lines.append(f"{prefix}```")
            lines.append("")

        elif block_type == "quote":
            text = _rich_text_to_md(data.get("rich_text", []))
            for line in text.split("\n"):
                lines.append(f"{prefix}> {line}")
            lines.append("")

        elif block_type == "callout":
            icon = block.get("icon", {})
            emoji = icon.get("emoji", "") if icon else ""
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{prefix}> {emoji} {text}")
            lines.append("")

        elif block_type == "toggle":
            text = _rich_text_to_md(data.get("rich_text", []))
            lines.append(f"{prefix}<details>")
            lines.append(f"{prefix}<summary>{text}</summary>")
            lines.append("")
            children = block.get("children", [])
            if children:
                _convert_blocks(children, lines, indent)
            lines.append(f"{prefix}</details>")
            lines.append("")

        elif block_type == "divider":
            lines.append(f"{prefix}---")
            lines.append("")

        elif block_type == "table":
            _convert_table(block, lines, prefix)

        elif block_type in ("child_page", "child_database"):
            title = data.get("title", "")
            lines.append(f"{prefix}[{title}]")
            lines.append("")

        # Skip unsupported types silently (image, video, embed, etc.)


def _rich_text_to_md(rich_text: List[Dict[str, Any]]) -> str:
    """Convert Notion rich text array to inline markdown."""
    parts: List[str] = []
    for segment in rich_text:
        text = segment.get("plain_text", "")
        if not text:
            continue

        annotations = segment.get("annotations", {})
        href = segment.get("href")

        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        if href:
            text = f"[{text}]({href})"

        parts.append(text)
    return "".join(parts)


def _convert_table(
    block: Dict[str, Any],
    lines: List[str],
    prefix: str,
) -> None:
    """Convert a Notion table block to markdown table."""
    rows = block.get("children", [])
    if not rows:
        return

    for i, row in enumerate(rows):
        cells = row.get("table_row", {}).get("cells", [])
        cell_texts = [_rich_text_to_md(cell) for cell in cells]
        lines.append(f"{prefix}| {' | '.join(cell_texts)} |")
        if i == 0:
            lines.append(f"{prefix}| {' | '.join('---' for _ in cell_texts)} |")

    lines.append("")


# ── Markdown → Notion blocks ─────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.+)$")
_TODO_RE = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```(\w*)$")


def markdown_to_blocks(md: str) -> List[Dict[str, Any]]:
    """Convert a markdown string to Notion API block objects."""
    blocks: List[Dict[str, Any]] = []
    lines = md.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code fence
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            lang = fence_match.group(1) or "plain text"
            code_lines: List[str] = []
            i += 1
            while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append(code_block("\n".join(code_lines), lang))
            continue

        # Heading
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            blocks.append(heading_block(text, level))
            i += 1
            continue

        # To-do item
        todo_match = _TODO_RE.match(line)
        if todo_match:
            checked = todo_match.group(1).lower() == "x"
            text = todo_match.group(2)
            blocks.append(todo_block(text, checked))
            i += 1
            continue

        # Bullet list
        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            blocks.append(bulleted_block(bullet_match.group(1)))
            i += 1
            continue

        # Numbered list
        num_match = _NUMBERED_RE.match(line)
        if num_match:
            blocks.append(numbered_block(num_match.group(1)))
            i += 1
            continue

        # Empty line — skip
        if not line.strip():
            i += 1
            continue

        # Default: paragraph
        blocks.append(paragraph_block(line))
        i += 1

    return blocks


def rich_text(text: str) -> List[Dict[str, Any]]:
    """Create a simple Notion rich_text array from plain text."""
    return [{"type": "text", "text": {"content": text}}]


def paragraph_block(text: str) -> Dict[str, Any]:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(text)}}


def heading_block(text: str, level: int) -> Dict[str, Any]:
    level = max(1, min(3, level))  # Notion supports h1-h3 only
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": rich_text(text)}}


def bulleted_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rich_text(text)},
    }


def numbered_block(text: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": rich_text(text)},
    }


def todo_block(text: str, checked: bool) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {"rich_text": rich_text(text), "checked": checked},
    }


def code_block(text: str, language: str) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": rich_text(text), "language": language},
    }


def toggle_block(summary: str, children: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {"rich_text": rich_text(summary), "children": children},
    }


def callout_block(text: str, emoji: str = "📋") -> Dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": rich_text(text),
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }
