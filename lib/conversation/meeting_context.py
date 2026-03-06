"""Meeting context loader — pre-meeting configuration.

Loads agenda, watch words, participants, and key topics from a YAML file.
Used to configure the trigger engine before a meeting starts.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_TEMPLATE = """\
# Meeting Context Configuration
# Load with: python coach.py --context meeting_context.yaml

title: "Meeting Title"

# Agenda items (tracked during meeting)
agenda_items:
  - "Item 1"
  - "Item 2"

# Watch words trigger immediate alerts when detected
watch_words:
  - "competitor"
  - "pricing"
  - "timeline"
  - "budget"

# Participants (for context)
participants:
  - "Alice (PM)"
  - "Bob (Engineering)"

# Key topics to track
key_topics:
  - "deployment"
  - "performance"
  - "compliance"

# Free-form notes included in generation context
notes: ""
"""


@dataclass
class MeetingContext:
    """Pre-meeting configuration loaded from YAML."""

    title: str = ""
    agenda_items: List[str] = field(default_factory=list)
    watch_words: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    key_topics: List[str] = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        """Human-readable summary for display."""
        parts = []
        if self.title:
            parts.append(f"Meeting: {self.title}")
        if self.agenda_items:
            parts.append(f"Agenda: {len(self.agenda_items)} items")
        if self.watch_words:
            parts.append(f"Watch words: {', '.join(self.watch_words[:5])}")
        if self.participants:
            parts.append(f"Participants: {len(self.participants)}")
        return " | ".join(parts) if parts else "No meeting context loaded"

    def as_prompt_context(self) -> str:
        """Format meeting context for inclusion in generation prompts."""
        lines: list[str] = []
        if self.title:
            lines.append(f"Meeting: {self.title}")
        if self.participants:
            lines.append(f"Participants: {', '.join(self.participants)}")
        if self.agenda_items:
            lines.append("Agenda:")
            for item in self.agenda_items:
                lines.append(f"  - {item}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


def load_meeting_context(path: Path) -> Optional[MeetingContext]:
    """Load meeting context from a YAML file.

    Returns None if file doesn't exist or can't be parsed.
    """
    if not path.exists():
        logger.info("No meeting context file at %s", path)
        return None

    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("pyyaml not installed, cannot load meeting context")
        return None
    except Exception as e:
        logger.warning("Failed to load meeting context: %s", e)
        return None

    return MeetingContext(
        title=raw.get("title", ""),
        agenda_items=raw.get("agenda_items", []),
        watch_words=raw.get("watch_words", []),
        participants=raw.get("participants", []),
        key_topics=raw.get("key_topics", []),
        notes=raw.get("notes", ""),
    )


def create_meeting_template(path: Path) -> None:
    """Create a template meeting_context.yaml for the user to fill in."""
    if path.exists():
        logger.info("Template already exists at %s", path)
        return

    path.write_text(_TEMPLATE)
    logger.info("Created meeting context template at %s", path)
