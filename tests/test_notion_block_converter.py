"""Tests for lib.notion.block_converter — bidirectional Notion block / markdown conversion."""

from typing import Any, Dict, List

from lib.notion.block_converter import blocks_to_markdown, markdown_to_blocks

# ── Helpers ──────────────────────────────────────────────────────────────────


def _rt(text: str) -> List[Dict[str, Any]]:
    """Shorthand for a plain rich_text array."""
    return [{"type": "text", "plain_text": text}]


def _annotated_rt(
    text: str,
    bold: bool = False,
    italic: bool = False,
    code: bool = False,
    strikethrough: bool = False,
    href: str | None = None,
) -> List[Dict[str, Any]]:
    """Rich text segment with annotations."""
    segment: Dict[str, Any] = {
        "type": "text",
        "plain_text": text,
        "annotations": {
            "bold": bold,
            "italic": italic,
            "code": code,
            "strikethrough": strikethrough,
        },
    }
    if href:
        segment["href"] = href
    return [segment]


# ── blocks_to_markdown ──────────────────────────────────────────────────────


class TestHeadingsToMarkdown:
    """Heading blocks → markdown headings."""

    def test_heading_1(self) -> None:
        blocks = [{"type": "heading_1", "heading_1": {"rich_text": _rt("Title")}}]
        assert blocks_to_markdown(blocks) == "# Title"

    def test_heading_2(self) -> None:
        blocks = [{"type": "heading_2", "heading_2": {"rich_text": _rt("Section")}}]
        assert blocks_to_markdown(blocks) == "## Section"

    def test_heading_3(self) -> None:
        blocks = [{"type": "heading_3", "heading_3": {"rich_text": _rt("Subsection")}}]
        assert blocks_to_markdown(blocks) == "### Subsection"


class TestParagraphsToMarkdown:
    """Paragraph blocks → markdown text."""

    def test_plain_paragraph(self) -> None:
        blocks = [{"type": "paragraph", "paragraph": {"rich_text": _rt("Hello world")}}]
        assert blocks_to_markdown(blocks) == "Hello world"

    def test_bold_text(self) -> None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _annotated_rt("bold", bold=True)},
            }
        ]
        assert "**bold**" in blocks_to_markdown(blocks)

    def test_italic_text(self) -> None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _annotated_rt("em", italic=True)},
            }
        ]
        assert "*em*" in blocks_to_markdown(blocks)

    def test_code_inline(self) -> None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _annotated_rt("fn()", code=True)},
            }
        ]
        assert "`fn()`" in blocks_to_markdown(blocks)

    def test_strikethrough(self) -> None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _annotated_rt("removed", strikethrough=True)},
            }
        ]
        assert "~~removed~~" in blocks_to_markdown(blocks)

    def test_link(self) -> None:
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": _annotated_rt("click", href="https://example.com")},
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "[click](https://example.com)" in md

    def test_mixed_rich_text(self) -> None:
        """Multiple rich text segments concatenate correctly."""
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "plain_text": "Hello ", "annotations": {}},
                        {
                            "type": "text",
                            "plain_text": "world",
                            "annotations": {"bold": True},
                        },
                    ]
                },
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "Hello " in md
        assert "**world**" in md


class TestListsToMarkdown:
    """List blocks → markdown lists."""

    def test_bulleted_list(self) -> None:
        blocks = [
            {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt("Item A")}},
            {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt("Item B")}},
        ]
        md = blocks_to_markdown(blocks)
        assert "- Item A" in md
        assert "- Item B" in md

    def test_numbered_list(self) -> None:
        blocks = [
            {"type": "numbered_list_item", "numbered_list_item": {"rich_text": _rt("First")}},
            {"type": "numbered_list_item", "numbered_list_item": {"rich_text": _rt("Second")}},
        ]
        md = blocks_to_markdown(blocks)
        assert "1. First" in md
        assert "1. Second" in md

    def test_nested_bulleted_list(self) -> None:
        blocks = [
            {
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": _rt("Parent")},
                "children": [
                    {
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": _rt("Child")},
                    }
                ],
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "- Parent" in md
        assert "    - Child" in md


class TestCodeBlocksToMarkdown:
    """Code blocks → fenced markdown."""

    def test_code_with_language(self) -> None:
        blocks = [
            {
                "type": "code",
                "code": {"rich_text": _rt("print('hi')"), "language": "python"},
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "```python" in md
        assert "print('hi')" in md
        assert md.strip().endswith("```")

    def test_code_without_language(self) -> None:
        blocks = [{"type": "code", "code": {"rich_text": _rt("x = 1"), "language": ""}}]
        md = blocks_to_markdown(blocks)
        assert "```" in md
        assert "x = 1" in md


class TestQuoteAndCalloutToMarkdown:
    """Quote and callout blocks → markdown blockquotes."""

    def test_quote_block(self) -> None:
        blocks = [{"type": "quote", "quote": {"rich_text": _rt("A wise saying")}}]
        md = blocks_to_markdown(blocks)
        assert "> A wise saying" in md

    def test_callout_with_emoji(self) -> None:
        blocks = [
            {
                "type": "callout",
                "callout": {"rich_text": _rt("Important note")},
                "icon": {"emoji": "⚠️"},
            }
        ]
        md = blocks_to_markdown(blocks)
        assert ">" in md
        assert "Important note" in md


class TestToggleBlockToMarkdown:
    """Toggle blocks → HTML details/summary."""

    def test_toggle_with_children(self) -> None:
        blocks = [
            {
                "type": "toggle",
                "toggle": {"rich_text": _rt("Click to expand")},
                "children": [
                    {"type": "paragraph", "paragraph": {"rich_text": _rt("Hidden content")}},
                ],
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "<details>" in md
        assert "<summary>Click to expand</summary>" in md
        assert "Hidden content" in md
        assert "</details>" in md


class TestToDoToMarkdown:
    """To-do items → checkbox markdown."""

    def test_unchecked(self) -> None:
        blocks = [{"type": "to_do", "to_do": {"rich_text": _rt("Buy milk"), "checked": False}}]
        md = blocks_to_markdown(blocks)
        assert "- [ ] Buy milk" in md

    def test_checked(self) -> None:
        blocks = [{"type": "to_do", "to_do": {"rich_text": _rt("Buy eggs"), "checked": True}}]
        md = blocks_to_markdown(blocks)
        assert "- [x] Buy eggs" in md


class TestDividerToMarkdown:
    """Divider blocks → horizontal rule."""

    def test_divider(self) -> None:
        blocks = [{"type": "divider", "divider": {}}]
        md = blocks_to_markdown(blocks)
        assert "---" in md


class TestTableToMarkdown:
    """Table blocks → markdown table."""

    def test_simple_table(self) -> None:
        blocks = [
            {
                "type": "table",
                "table": {},
                "children": [
                    {
                        "type": "table_row",
                        "table_row": {"cells": [_rt("Name"), _rt("Age")]},
                    },
                    {
                        "type": "table_row",
                        "table_row": {"cells": [_rt("Alice"), _rt("30")]},
                    },
                ],
            }
        ]
        md = blocks_to_markdown(blocks)
        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md

    def test_empty_table(self) -> None:
        blocks = [{"type": "table", "table": {}, "children": []}]
        md = blocks_to_markdown(blocks)
        # No crash, no table output
        assert "|" not in md


class TestEmptyAndEdgeCases:
    """Edge cases for blocks_to_markdown."""

    def test_empty_block_list(self) -> None:
        assert blocks_to_markdown([]) == ""

    def test_empty_paragraph(self) -> None:
        blocks = [{"type": "paragraph", "paragraph": {"rich_text": []}}]
        assert blocks_to_markdown(blocks) == ""

    def test_unknown_block_type_skipped(self) -> None:
        blocks = [
            {"type": "image", "image": {"url": "https://example.com/photo.png"}},
            {"type": "paragraph", "paragraph": {"rich_text": _rt("After image")}},
        ]
        md = blocks_to_markdown(blocks)
        assert "After image" in md
        assert "photo.png" not in md

    def test_empty_rich_text_segment(self) -> None:
        """Rich text segment with empty plain_text is skipped."""
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "plain_text": ""},
                        {"type": "text", "plain_text": "visible"},
                    ]
                },
            }
        ]
        md = blocks_to_markdown(blocks)
        assert md.strip() == "visible"


# ── markdown_to_blocks ──────────────────────────────────────────────────────


class TestMarkdownToBlocksHeadings:
    """Markdown headings → Notion heading blocks."""

    def test_h1(self) -> None:
        blocks = markdown_to_blocks("# Title")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_1"

    def test_h2(self) -> None:
        blocks = markdown_to_blocks("## Section")
        assert blocks[0]["type"] == "heading_2"

    def test_h3(self) -> None:
        blocks = markdown_to_blocks("### Sub")
        assert blocks[0]["type"] == "heading_3"


class TestMarkdownToBlocksLists:
    """Markdown lists → Notion list item blocks."""

    def test_bulleted_list(self) -> None:
        md = "- Alpha\n- Beta"
        blocks = markdown_to_blocks(md)
        assert len(blocks) == 2
        assert all(b["type"] == "bulleted_list_item" for b in blocks)

    def test_numbered_list(self) -> None:
        md = "1. First\n2. Second"
        blocks = markdown_to_blocks(md)
        assert len(blocks) == 2
        assert all(b["type"] == "numbered_list_item" for b in blocks)


class TestMarkdownToBlocksCode:
    """Markdown code fences → Notion code blocks."""

    def test_code_block_with_lang(self) -> None:
        md = "```python\nprint('hi')\n```"
        blocks = markdown_to_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "python"
        assert blocks[0]["code"]["rich_text"][0]["text"]["content"] == "print('hi')"

    def test_code_block_no_lang(self) -> None:
        md = "```\nfoo\n```"
        blocks = markdown_to_blocks(md)
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "plain text"


class TestMarkdownToBlocksToDo:
    """Markdown checkboxes → Notion to_do blocks."""

    def test_unchecked(self) -> None:
        blocks = markdown_to_blocks("- [ ] Task A")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "to_do"
        assert blocks[0]["to_do"]["checked"] is False

    def test_checked(self) -> None:
        blocks = markdown_to_blocks("- [x] Task B")
        assert blocks[0]["to_do"]["checked"] is True


class TestMarkdownToBlocksParagraph:
    """Plain text → Notion paragraph blocks."""

    def test_plain_paragraph(self) -> None:
        blocks = markdown_to_blocks("Just some text here")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"

    def test_empty_lines_skipped(self) -> None:
        blocks = markdown_to_blocks("\n\n\n")
        assert len(blocks) == 0


# ── Round-trip ───────────────────────────────────────────────────────────────


class TestRoundTrip:
    """Markdown → blocks → markdown preserves core structure.

    Note: markdown_to_blocks produces Notion API write format (text.content),
    while blocks_to_markdown reads Notion API response format (plain_text).
    These tests verify structural round-trip by injecting plain_text into
    the blocks produced by markdown_to_blocks.
    """

    @staticmethod
    def _inject_plain_text(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add plain_text keys to rich_text segments (simulating API response)."""
        for block in blocks:
            btype = block.get("type", "")
            data = block.get(btype, {})
            for seg in data.get("rich_text", []):
                if "text" in seg and "plain_text" not in seg:
                    seg["plain_text"] = seg["text"]["content"]
            for child in block.get("children", []):
                TestRoundTrip._inject_plain_text([child])
        return blocks

    def test_heading_round_trip(self) -> None:
        original = "## My Section"
        blocks = self._inject_plain_text(markdown_to_blocks(original))
        md = blocks_to_markdown(blocks)
        assert "## My Section" in md

    def test_bullet_list_round_trip(self) -> None:
        original = "- Item A\n- Item B"
        blocks = self._inject_plain_text(markdown_to_blocks(original))
        md = blocks_to_markdown(blocks)
        assert "- Item A" in md
        assert "- Item B" in md

    def test_code_round_trip(self) -> None:
        original = "```python\nx = 1\n```"
        blocks = self._inject_plain_text(markdown_to_blocks(original))
        md = blocks_to_markdown(blocks)
        assert "```python" in md
        assert "x = 1" in md

    def test_mixed_content_round_trip(self) -> None:
        original = "# Title\n\nSome text\n\n- bullet\n\n1. numbered"
        blocks = self._inject_plain_text(markdown_to_blocks(original))
        md = blocks_to_markdown(blocks)
        assert "# Title" in md
        assert "Some text" in md
        assert "- bullet" in md
        assert "1. numbered" in md
