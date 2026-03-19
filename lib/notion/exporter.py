"""Export meeting transcripts and notes to Notion pages.

Creates a child page under a configured parent with:
- Metadata callout (title, date, participants, duration)
- Meeting notes sections
- Transcript in a toggle block (collapsed by default)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from lib.notion.block_converter import (
    callout_block,
    heading_block,
    markdown_to_blocks,
    paragraph_block,
    toggle_block,
)
from lib.notion.client import NotionClient

logger = logging.getLogger(__name__)


def export_meeting(
    client: NotionClient,
    parent_page_id: str,
    title: str,
    date: str,
    participants: Optional[List[str]] = None,
    notes_md: str = "",
    transcript_md: str = "",
    duration_seconds: float = 0.0,
) -> str:
    """Export a meeting to Notion as a child page.

    Args:
        client: Authenticated Notion client.
        parent_page_id: Notion page ID to create the meeting under.
        title: Meeting title (used as page title).
        date: Meeting date string (ISO or human-readable).
        participants: List of participant names.
        notes_md: Markdown meeting notes (from LLM generation).
        transcript_md: Markdown transcript (from TranscriptStore export).
        duration_seconds: Meeting duration for the metadata block.

    Returns:
        URL of the created Notion page.
    """
    children: List[Dict[str, Any]] = []

    # 1. Metadata callout
    meta_parts = [f"Date: {date}"]
    if participants:
        meta_parts.append(f"Participants: {', '.join(participants)}")
    if duration_seconds > 0:
        minutes = int(duration_seconds // 60)
        meta_parts.append(f"Duration: {minutes} min")
    children.append(callout_block(" | ".join(meta_parts), emoji="\U0001f4cb"))

    # 2. Notes section
    if notes_md.strip():
        children.append(heading_block("Meeting Notes", 2))
        notes_blocks = markdown_to_blocks(notes_md)
        children.extend(notes_blocks)

    # 3. Transcript in a toggle (collapsed by default)
    if transcript_md.strip():
        transcript_blocks = markdown_to_blocks(transcript_md)
        if transcript_blocks:
            toggle_children = transcript_blocks[:100]
            children.append(toggle_block("Full Transcript", toggle_children))

            if len(transcript_blocks) > 100:
                children.append(paragraph_block("(transcript continues below)"))
                children.extend(transcript_blocks[100:])

    if not children:
        children.append(paragraph_block("(No content recorded)"))

    page_title = f"{title} \u2014 {date}" if date else title
    url = client.create_page(parent_page_id, page_title, children)
    logger.info("Exported meeting to Notion: %s (%d blocks)", url, len(children))
    return url
