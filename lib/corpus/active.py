"""Active corpus selection — which docs dir feeds the live session (F-704).

The "Prepare corpus" flow ends with *ready*: the distilled corpus becomes the
live retrieval source. Rather than editing config.yaml, activation writes a
small state file; the orchestrator resolves it at session construction, so the
switch takes effect on the next session start (sessions are rebuilt per start —
see the session-lifecycle design note in CLAUDE.md).

The activated corpus gets its own index DB so the configured corpus's index is
never polluted with distilled chunks (and deactivating is instant + clean).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from lib.paths import get_docs_dir

logger = logging.getLogger(__name__)

ACTIVE_FILE = Path("data/corpus_active.json")
ACTIVE_DB = Path("data/rag_active.db")


def set_active_dir(docs_dir: Optional[Path], state_file: Optional[Path] = None) -> None:
    """Activate ``docs_dir`` as the live corpus (None deactivates → configured dir)."""
    state_file = state_file or ACTIVE_FILE
    if docs_dir is None:
        state_file.unlink(missing_ok=True)
        logger.info("active corpus cleared — configured docs dir applies")
        return
    if not docs_dir.is_dir():
        raise ValueError(f"corpus dir not found: {docs_dir}")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"dir": str(docs_dir)}), encoding="utf-8")
    logger.info("active corpus set to %s (applies on next session start)", docs_dir)


def get_active_dir(state_file: Optional[Path] = None) -> Optional[Path]:
    """The activated corpus dir, or None if unset/invalid (invalid = ignored)."""
    state_file = state_file or ACTIVE_FILE
    if not state_file.exists():
        return None
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        path = Path(str(raw["dir"]))
    except (json.JSONDecodeError, KeyError, OSError):
        logger.warning("unreadable %s — falling back to configured docs dir", state_file)
        return None
    if not path.is_dir():
        logger.warning("active corpus dir %s missing — falling back to configured", path)
        return None
    return path


def resolve_corpus(configured_docs_dir: str, configured_db: str) -> Tuple[Path, Path]:
    """(docs_dir, db_path) for the live session: the activated corpus with its
    own DB when set, otherwise the configured pair."""
    active = get_active_dir()
    if active is not None:
        return active, ACTIVE_DB
    return get_docs_dir(configured_docs_dir), Path(configured_db)
