#!/usr/bin/env python3
"""F-503 ship gate — trained head vs F-510 probe vs heuristic on the frozen held-out.

Runs all three routers COLD on tests/eval/f503_trigger_router_eval.yaml (domains held
out of the training families) and prints per-class + macro-F1 for each. Ship rule
(same as F-510's evaluate_probe_gate, conservative): F-503 ships as the default Head
only if it beats BOTH the probe and the heuristic on macro-F1 AND does not regress the
`question` class (the one the heuristic owns).

F-503 numbers are read from the forge eval JSON (produced by the artifact eval script on
MPS/CPU); the probe + heuristic are computed here on the identical rows. Local only.

Usage:
  MODELS_DIR=/abs/_models python tests/eval/f503_gate.py \
      --heldout tests/eval/f503_trigger_router_eval.yaml \
      --f503-json forge/f503-trigger-router/heldout_f503.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

LABELS = ("question", "alert", "topic", "followup", "none")


def _fail(msg: str) -> "None":
    """Print a clear diagnostic and exit non-zero (this is a CLI gate, not a library)."""
    print(f"f503_gate: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _f1_report(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    per: Dict[str, Dict[str, float]] = {}
    for c in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(1 for t in y_true if t == c),
        }
    macro = sum(v["f1"] for v in per.values()) / len(per)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {"accuracy": acc, "macro_f1": macro, "per_class": per}


def _load_heldout(path: Path) -> Tuple[List[str], List[str]]:
    if not path.is_file():
        _fail(f"held-out file not found: {path}")
    try:
        if path.suffix == ".jsonl":
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        else:
            doc = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict) or "examples" not in doc:
                _fail(f"{path}: expected a mapping with an 'examples' key")
            rows = doc["examples"]
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        _fail(f"{path}: could not parse ({exc})")
    if not rows:
        _fail(f"{path}: no rows found")
    try:
        return [r["text"] for r in rows], [r["label"] for r in rows]
    except (TypeError, KeyError):
        _fail(f"{path}: every row needs 'text' and 'label' fields")
        raise  # unreachable; satisfies type checker


def _probe_predict(texts: Sequence[str]) -> List[str]:
    """F-510 frozen linear probe, fit on its full seed set, predicting held-out."""
    from sklearn.linear_model import LogisticRegression

    from lib.intelligence.encoder import EncoderBackbone
    from lib.intelligence.heads.probe_data import load_probe_examples

    enc = EncoderBackbone()
    examples = load_probe_examples()
    x_tr = enc.embed_batch([t for t, _ in examples])
    y_tr = [y for _, y in examples]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    x_te = enc.embed_batch(list(texts))
    return [str(p) for p in clf.predict(x_te)]


def _heuristic_predict(texts: Sequence[str]) -> List[str]:
    """Production heuristic router: only questions are decidable from isolated text
    (alert needs watch-words, topic a RAG match, followup a pause) — everything else
    falls to `none`. Same asymmetry the F-510 gate reports transparently."""
    from lib.config import TriggerConfig
    from lib.triggers.question_trigger import QuestionTrigger

    qt = QuestionTrigger(TriggerConfig())
    out: List[str] = []
    for t in texts:
        res = qt.evaluate(t, "")
        out.append("question" if res is not None else "none")
    return out


def _print_report(name: str, rep: Dict[str, object]) -> None:
    pc = rep["per_class"]  # type: ignore[index]
    print(f"\n{name}: acc={rep['accuracy']:.3f}  macro-F1={rep['macro_f1']:.3f}")
    for c in LABELS:
        m = pc[c]
        print(
            f"  {c:9s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--heldout", type=Path, default=Path("tests/eval/f503_trigger_router_eval.yaml")
    )
    ap.add_argument(
        "--f503-json", type=Path, default=Path("forge/f503-trigger-router/heldout_f503.json")
    )
    args = ap.parse_args()

    texts, gold = _load_heldout(args.heldout)

    if not args.f503_json.is_file():
        _fail(f"F-503 eval JSON not found: {args.f503_json} (run the artifact eval script first)")
    try:
        f503 = json.loads(args.f503_json.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"{args.f503_json}: invalid JSON ({exc})")
    missing = [k for k in ("accuracy", "macro_f1", "per_class") if k not in f503]
    if missing:
        _fail(f"{args.f503_json}: missing keys {missing}")
    if any(c not in f503["per_class"] for c in LABELS):
        _fail(f"{args.f503_json}: per_class must cover all labels {LABELS}")
    f503_rep = {
        "accuracy": f503["accuracy"],
        "macro_f1": f503["macro_f1"],
        "per_class": f503["per_class"],
    }

    probe_rep = _f1_report(gold, _probe_predict(texts))
    heur_rep = _f1_report(gold, _heuristic_predict(texts))

    print("=" * 66)
    print(f"F-503 SHIP GATE — {len(texts)} held-out-domain rows (unseen by training)")
    print("=" * 66)
    _print_report("F-503 (trained encoder head)", f503_rep)
    _print_report("F-510 (frozen linear probe)", probe_rep)
    _print_report("heuristic (question-only router)", heur_rep)

    fm, pm, hm = f503_rep["macro_f1"], probe_rep["macro_f1"], heur_rep["macro_f1"]
    fq = f503_rep["per_class"]["question"]["f1"]
    pq = probe_rep["per_class"]["question"]["f1"]
    hq = heur_rep["per_class"]["question"]["f1"]
    beats = fm > pm and fm > hm
    q_ok = fq >= pq and fq >= hq
    ship = bool(beats and q_ok)

    print("\n" + "-" * 66)
    print(f"macro-F1:  F-503={fm:.3f}  probe={pm:.3f}  heuristic={hm:.3f}")
    print(f"question-F1: F-503={fq:.3f}  probe={pq:.3f}  heuristic={hq:.3f}")
    print(f"beats both on macro-F1: {beats}   |   no question regression: {q_ok}")
    print(
        f"\nDECISION: {'SHIP as default Head' if ship else 'KEEP heuristic default; F-503 wired-but-off'}"
    )
    print("-" * 66)


if __name__ == "__main__":
    main()
