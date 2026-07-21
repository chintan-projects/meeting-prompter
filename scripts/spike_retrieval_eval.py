"""Stage-0 spike: retrieval A/B — all-MiniLM-L6 vs LFM2.5-Embedding-350M (F-502).

Runs the EXISTING eval harness (tests/eval) twice through the same hybrid
pipeline, swapping only the embedder, and prints Hit@1 / Hit@3 / MRR side by
side. Baseline to beat: Hit@1=94.4%, Hit@3=100%, MRR=0.972 (MiniLM).

Note: symmetric encoding (no query/passage instruction prefix) for parity with
how MiniLM is used. LFM2.5-Embedding may score higher with proper task prompts —
flagged as a follow-up if the raw number is close.

Run (project venv):  python scripts/spike_retrieval_eval.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LFM_EMBED = Path.home() / "Projects" / "_models" / "LFM2.5-Embedding-350M"
# MRR noise floor: treat a candidate within this of baseline as a tie, not a regression.
MRR_TIE_TOLERANCE = 0.005


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "rag_eval_harness", REPO / "tests" / "eval" / "test_rag_eval.py"
    )
    assert (
        spec and spec.loader
    ), f"could not load eval harness at {REPO}/tests/eval/test_rag_eval.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    harness = load_harness()
    from lib.rag.embedder import SentenceTransformerEmbedder

    class LFMEmbedder(SentenceTransformerEmbedder):
        """LFM2.5-Embedding via sentence-transformers (needs trust_remote_code)."""

        def __init__(self) -> None:
            super().__init__(model_name=str(LFM_EMBED))

        def _load_model(self) -> None:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)

        @property
        def dimension(self) -> int:
            return 1024

    def run(tag: str):
        print(f"\n{'=' * 56}\n {tag}\n{'=' * 56}")
        report = harness.run_eval(harness.DOCS_DIR, harness.DATASET_PATH)
        return report

    # Baseline (MiniLM) — the class run_eval instantiates by default.
    base = run("all-MiniLM-L6-v2  (baseline)")

    # Swap the embedder the harness constructs, then re-run.
    harness.SentenceTransformerEmbedder = LFMEmbedder  # type: ignore[attr-defined]
    lfm = run("LFM2.5-Embedding-350M  (candidate)")

    def row(name: str, r) -> str:
        return (
            f"{name:<26}  Hit@1={r.mean_hit_at_1:6.1%}   "
            f"Hit@3={r.mean_hit_at_3:6.1%}   MRR={r.mean_mrr:.3f}"
        )

    print(f"\n{'=' * 56}\n COMPARISON\n{'=' * 56}")
    print(row("all-MiniLM-L6-v2", base))
    print(row("LFM2.5-Embedding-350M", lfm))
    d1 = lfm.mean_hit_at_1 - base.mean_hit_at_1
    dm = lfm.mean_mrr - base.mean_mrr
    print(f"\nΔ Hit@1 = {d1:+.1%}   Δ MRR = {dm:+.3f}")
    verdict = (
        "LANDS (>= baseline)"
        if (d1 >= 0 and dm >= -MRR_TIE_TOLERANCE)
        else "REVIEW (below baseline)"
    )
    print(f"verdict: {verdict}  —  gate is 'do not regress'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
