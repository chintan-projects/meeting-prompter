"""F-510 runner: persist the probe dataset + print the probe-vs-heuristic gate.

Encoder-only, no GPU/forge/egress. Writes the seed labeled set to the gitignored
``data/fixtures/trigger_probe_dataset.jsonl`` (hand-extendable), then fits the
frozen-encoder linear probe and reports macro-F1 + per-class against the
heuristic router on a frozen held-out split, with the wiring decision.

Run (project venv):  python scripts/eval_probe_head.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    from lib.intelligence.heads.linear_probe import evaluate_probe_gate
    from lib.intelligence.heads.probe_data import write_seed_dataset

    n = write_seed_dataset()
    print(f"persisted {n} seed examples → data/fixtures/trigger_probe_dataset.jsonl\n")

    report = evaluate_probe_gate()
    print(json.dumps(report, indent=2))

    decision = (
        "WIRE probe as default"
        if report["wire_as_default"]
        else "KEEP heuristic default; probe wired-but-off"
    )
    print(f"\ndecision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
