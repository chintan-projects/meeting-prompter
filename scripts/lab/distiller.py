"""Thin CLI wrapper over the productized distiller (lib.corpus.distiller, F-701).

The distiller core lives in lib/corpus/ now; this keeps the lab entry point:

    python -m scripts.lab.distiller <source.md> [--backend heuristic|cloud]
                                    [--mode atomic|consolidated] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lib.config import load_config
from lib.corpus.distiller import (  # noqa: F401 — re-exported for lab/test compat
    MAX_SECTION_CHARS,
    MAX_TOPIC_CHARS,
    MAX_UNIT_WORDS,
    MIN_SECTION_WORDS,
    _distill_cloud,
    _distill_heuristic,
    distill,
)
from lib.paths import get_docs_dir

logger = logging.getLogger("lab.distiller")


def _default_src() -> Path:
    """The single .md in the configured corpus dir, so `distiller` needs no arg."""
    docs = get_docs_dir(load_config().paths.docs_dir)
    mds = sorted(docs.glob("*.md")) if docs.is_dir() else []
    if len(mds) != 1:
        raise SystemExit(
            f"pass a source .md — the corpus dir {docs} has {len(mds)} markdown files, "
            "so the default is ambiguous."
        )
    return mds[0]


def _meta_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def _guard_overwrite(out: Path, backend: str, force: bool) -> None:
    """Refuse to silently replace a corpus built by a different backend.

    A distill run costs either minutes of local compute or real API spend, and
    the output path is shared. Overwriting a cloud corpus with a local one (or
    vice versa) destroys the artifact AND invalidates any measurement taken
    against it — silently, because both produce a file with the same name.
    """
    meta = _meta_path(out)
    if force or not out.exists() or not meta.exists():
        return
    try:
        prior = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    prior_backend = str(prior.get("backend", ""))
    if prior_backend and prior_backend != backend:
        raise SystemExit(
            f"refusing to overwrite: {out} was built by the '{prior_backend}' backend "
            f"({prior.get('units', '?')} units) and you are running '{backend}'.\n"
            f"Pass --force to replace it, or --out to write elsewhere."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Distill an explainer doc into borrowable answer-units."
    )
    ap.add_argument("src", nargs="?", help="source .md (default: the configured corpus doc)")
    ap.add_argument(
        "--backend",
        choices=["heuristic", "local", "cloud"],
        default="local",
        help="local = on-device model (shipped default, ADR-001); "
        "heuristic = no-model floor; cloud = offline validation only",
    )
    ap.add_argument(
        "--mode",
        choices=["atomic", "consolidated"],
        default="consolidated",
        help="atomic = 1-5 short facts/section; consolidated = one complete answer/section",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--force",
        action="store_true",
        help="replace a corpus built by a different backend (destroys that artifact)",
    )
    args = ap.parse_args()
    src = Path(args.src) if args.src else _default_src()
    out = Path(args.out) if args.out else Path("data/distilled") / f"{src.stem}.distilled.md"
    _guard_overwrite(out, args.backend, args.force)
    logger.info("distilling %s (backend=%s, mode=%s)", src.name, args.backend, args.mode)
    stats = distill(src, out, args.backend, args.mode)
    # Record provenance next to the corpus so the next run (and the reader of any
    # coverage number) knows which backend produced it.
    _meta_path(out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
