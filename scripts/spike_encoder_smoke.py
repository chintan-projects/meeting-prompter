"""Stage-0 spike: LFM2.5-Encoder-350M smoke test (F-500).

Answers three questions before we build the intelligence layer on it:
  1. Does the from-scratch bidirectional encoder load (with real weights) and run?
  2. What is the steady-state per-turn forward latency (mean-pooled) on-device?
  3. Is mean-pool giving us a sane sequence vector to hang classifier heads off?

This is a throwaway spike — not production code. Run:
    python scripts/spike_encoder_smoke.py
    MODELS_DIR=/path/to/_models python scripts/spike_encoder_smoke.py

Loading note: the checkpoint is saved from the MLM wrapper
`Lfm2BidirForMaskedLM_theirs`, whose encoder backbone is nested under `.lfm2`.
Loading the bare AutoModel (`Lfm2BidirectionalModel_theirs`) silently random-
initializes every layer (prefix mismatch), so we load AutoModelForMaskedLM and
read hidden states from `.lfm2`.

MPS note: latency is measured on a FIXED padded shape and warmed up, because
Metal recompiles the graph per new sequence length — otherwise the numbers are
compilation overhead, not inference.

Requires: torch, transformers (custom modeling loaded via trust_remote_code).
No network — reads the local checkpoint only.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

MODEL_DIRNAME = "LFM2.5-Encoder-350M"
PAD_LEN = 128  # fixed shape so MPS doesn't recompile per length
WARMUP = 3
REPEATS = 5
PER_TURN_BUDGET_MS = 80.0  # a per-turn gate should stay comfortably under this

# Representative rolling-window transcript snippets (what the layer sees per turn).
SAMPLES: list[str] = [
    "so the main blocker on the AMD router work is the latency budget, we're at "
    "about 390 milliseconds and the target was under 300. do we have headroom on "
    "the quantization side or is that already maxed out?",
    "right, and I think the other thing worth raising is whether the eval set "
    "actually matches production distribution. the flat benchmark drove five "
    "iterations that all underperformed.",
    "yeah okay. let's move on.",
    "can you remind me what the fallback chain looks like if the primary model "
    "is unavailable during a live call?",
]


def resolve_model_path() -> Path:
    models_dir = os.environ.get("MODELS_DIR")
    candidates: list[Path] = []
    if models_dir:
        candidates.append(Path(models_dir).expanduser() / MODEL_DIRNAME)
    candidates.append(Path.home() / "Projects" / "_models" / MODEL_DIRNAME)
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(f"Could not find {MODEL_DIRNAME}. Tried:\n  {tried}")


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install with: pip install torch transformers")

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    model_path = resolve_model_path()
    device = pick_device()
    print(f"model:  {model_path}")
    print(f"device: {device}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    wrapper, info = AutoModelForMaskedLM.from_pretrained(
        str(model_path), trust_remote_code=True, output_loading_info=True
    )
    # The MLM wrapper nests the encoder backbone under `.lfm2`; fail loudly if not.
    backbone = getattr(wrapper, "lfm2", None)
    if backbone is None:
        children = list(dict(wrapper.named_children()).keys())
        raise SystemExit(
            f"Expected MLM wrapper to expose `.lfm2` backbone; got submodules: {children}"
        )
    backbone.eval().to(device)
    load_s = time.perf_counter() - t0
    hidden = int(getattr(backbone.config, "hidden_size", -1))

    missing = info.get("missing_keys", [])
    print(f"loaded in {load_s:.1f}s · hidden_size={hidden}")
    print(
        f"weight load: {len(missing)} missing keys "
        f"({'CLEAN' if not missing else 'PROBLEM — random init'})\n"
    )

    @torch.no_grad()
    def encode(text: str) -> tuple["torch.Tensor", int]:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=PAD_LEN,
        ).to(device)
        out = backbone(**enc)
        hidden_states = out.last_hidden_state  # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)  # [1, T, 1]
        # Mean-pool over real tokens (final encoder layer is conv — never last-token).
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        n_tokens = int(enc["attention_mask"].sum().item())
        return pooled.squeeze(0), n_tokens

    import torch as _torch

    # Warm up the fixed shape so MPS compiles once, before timing.
    for _ in range(WARMUP):
        encode(SAMPLES[0])
    if device == "mps":
        _torch.mps.synchronize()

    latencies_ms: list[float] = []
    vectors = []
    print(f"{'tokens':>7}  {'ms (best of %d)' % REPEATS:>15}  vector[:3]")
    print("-" * 52)
    for text in SAMPLES:
        best = float("inf")
        vec = None
        n_tokens = 0
        for _ in range(REPEATS):
            t = time.perf_counter()
            vec, n_tokens = encode(text)
            if device == "mps":
                _torch.mps.synchronize()
            best = min(best, (time.perf_counter() - t) * 1000.0)
        latencies_ms.append(best)
        vectors.append(vec)
        preview = ", ".join(f"{v:+.3f}" for v in vec[:3].tolist())
        print(f"{n_tokens:>7}  {best:>15.1f}  [{preview}]")

    cos = _torch.nn.functional.cosine_similarity(vectors[0], vectors[-1], dim=0).item()

    print("\n--- verdict ---")
    print(
        f"forward latency  mean={statistics.mean(latencies_ms):.1f}ms  "
        f"p50={statistics.median(latencies_ms):.1f}ms  "
        f"max={max(latencies_ms):.1f}ms  (fixed {PAD_LEN}-token shape)"
    )
    print(f"pooled dim       {tuple(vectors[0].shape)}  (expect ({hidden},))")
    print(f"cos(sample0, sample3) = {cos:+.3f}  (want < 1.0 → vectors discriminate)")
    budget_ok = statistics.mean(latencies_ms) < PER_TURN_BUDGET_MS
    print(
        f"per-turn budget  {'OK' if budget_ok else 'REVIEW'} "
        f"(<{PER_TURN_BUDGET_MS:.0f}ms target for a per-turn gate)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
