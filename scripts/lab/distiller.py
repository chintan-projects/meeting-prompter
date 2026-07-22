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
    args = ap.parse_args()
    src = Path(args.src) if args.src else _default_src()
    out = Path(args.out) if args.out else Path("data/distilled") / f"{src.stem}.distilled.md"
    logger.info("distilling %s (backend=%s, mode=%s)", src.name, args.backend, args.mode)
    stats = distill(src, out, args.backend, args.mode)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
