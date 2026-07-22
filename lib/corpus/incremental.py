"""Incremental corpus distillation — re-distill only what changed (F-706).

Distilling a document through a model is minutes of work; editing one doc must
not force a full corpus rebuild. `distill_dir` keeps a manifest next to the
distilled outputs recording each source's content hash plus the backend/mode
that produced it. On the next run, only sources whose hash changed (or that are
new) are re-distilled; outputs whose source disappeared are removed, so the
distilled directory — and any index built from it — stays consistent with the
source corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from lib.corpus.distiller import distill

#: Progress hook: (source_name, index_1based, total_sources) before each doc.
ProgressCallback = Callable[[str, int, int], None]

logger = logging.getLogger(__name__)

MANIFEST_NAME = ".distill_manifest.json"
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")  # CompositeParser's set


def _doc_hash(path: Path) -> str:
    """Content hash of a source document (renames alone don't force re-distill)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        logger.warning("unreadable distill manifest %s — full re-distill", path)
        return {}


def _out_name(src: Path) -> str:
    return f"{src.stem}.distilled.md"


def distill_dir(
    src_dir: Path,
    out_dir: Path,
    backend: str = "heuristic",
    mode: str = "consolidated",
    force: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Distill every supported document in ``src_dir`` into ``out_dir``, skipping
    sources unchanged since the manifest's last run.

    A backend or mode different from the manifest's invalidates everything —
    mixed-provenance corpora would make coverage numbers unattributable.

    Args:
        src_dir: source documents directory.
        out_dir: distilled outputs + manifest; created if missing.
        backend: distiller backend for changed docs ("heuristic"|"local"|"cloud").
        mode: distiller mode for changed docs.
        force: re-distill everything regardless of the manifest.
        progress_cb: called as (source_name, index_1based, total) before each doc
            (skipped docs included) — feeds the wizard's progress display.

    Returns:
        ``{distilled: [...], skipped: [...], removed: [...], units, out_dir}``.
    """
    if not src_dir.is_dir():
        raise ValueError(f"source dir not found: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    same_recipe = manifest.get("backend") == backend and manifest.get("mode") == mode
    docs: dict[str, Any] = manifest.get("docs", {}) if (same_recipe and not force) else {}

    sources = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    distilled: list[str] = []
    skipped: list[str] = []
    new_docs: dict[str, Any] = {}
    total_units = 0

    for i, src in enumerate(sources, start=1):
        if progress_cb is not None:
            progress_cb(src.name, i, len(sources))
        digest = _doc_hash(src)
        out_path = out_dir / _out_name(src)
        prior = docs.get(src.name)
        if prior and prior.get("sha256") == digest and out_path.exists():
            skipped.append(src.name)
            new_docs[src.name] = prior
            total_units += int(prior.get("units") or 0)
            continue
        stats = distill(src, out_path, backend=backend, mode=mode)
        distilled.append(src.name)
        new_docs[src.name] = {"sha256": digest, "out": _out_name(src), "units": stats["units"]}
        total_units += int(stats["units"])

    # Sources that disappeared: drop their outputs so the distilled corpus (and
    # any index rebuilt from it) can't serve answers from deleted documents.
    removed: list[str] = []
    current = {_out_name(s) for s in sources}
    for stale in sorted(out_dir.glob("*.distilled.md")):
        if stale.name not in current:
            stale.unlink()
            removed.append(stale.name)

    (out_dir / MANIFEST_NAME).write_text(
        json.dumps({"backend": backend, "mode": mode, "docs": new_docs}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "distill_dir: %d distilled, %d skipped, %d removed (%s)",
        len(distilled),
        len(skipped),
        len(removed),
        out_dir,
    )
    return {
        "distilled": distilled,
        "skipped": skipped,
        "removed": removed,
        "units": total_units,
        "out_dir": str(out_dir),
    }
