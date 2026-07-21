"""Stage-0 spike: frozen-encoder linear probe on the trigger task (no training).

Measures how much trigger-classification signal is ALREADY in the frozen
LFM2.5-Encoder-350M representation — before any LoRA/GPU. Method:
  1. mean-pool the FROZEN encoder → 1024-d vector per utterance (no grad),
  2. fit a cheap linear probe (logistic regression) on a small labeled set,
  3. report macro-F1 + per-class on a held-out split.
Also runs a zero-fit nearest-centroid baseline (no probe) for contrast, and the
majority-class floor. Encoder stays frozen throughout — this is NOT training the
model, it's reading its features + a linear head fit in seconds on CPU.

Caveat: the labeled set is small + synthetic → a DIRECTIONAL ceiling estimate,
not a production number. Real labeled meeting data would tighten it.

Run (project venv): python scripts/spike_encoder_linear_probe.py
"""

from __future__ import annotations

import os
from pathlib import Path

MODEL_DIRNAME = "LFM2.5-Encoder-350M"

# Small, deliberately-varied labeled set (some ambiguity on purpose).
DATA: list[tuple[str, str]] = [
    # question — genuine, answer-worthy
    ("what's our current latency on the AMD router?", "question"),
    ("how does the fallback chain work if the primary model is down?", "question"),
    ("can you remind me what Decagon's valuation was?", "question"),
    ("what models are we bundling in the core app?", "question"),
    ("which quantization level did we ship for the audio model?", "question"),
    ("how many enterprise deals closed last quarter?", "question"),
    ("what's the target download size for the app store build?", "question"),
    ("do we have benchmarks against ModernBERT?", "question"),
    ("what context length does the encoder support?", "question"),
    ("what's the confidence threshold for the RAG pipeline?", "question"),
    ("who owns the packaging workstream right now?", "question"),
    ("is BlackHole still a dependency for system audio?", "question"),
    ("what net retention number are we quoting customers?", "question"),
    ("when is the role-scoping session with Ramin?", "question"),
    # alert — heads-up / risk / watch-word
    ("just so you know, the competitor just dropped pricing thirty percent.", "alert"),
    ("heads up, the deadline for the BMW deliverable moved to Friday.", "alert"),
    ("that number contradicts what we told the customer last week.", "alert"),
    ("careful, that's confidential — it's under NDA with Denso.", "alert"),
    ("we're about to blow the latency budget on this path.", "alert"),
    ("legal flagged the data retention clause as a blocker.", "alert"),
    ("the customer explicitly said security is their top concern.", "alert"),
    ("watch out, that claim isn't validated on their data yet.", "alert"),
    ("our GPU costs are running way over budget this month.", "alert"),
    ("the Verizon contract renewal is at risk.", "alert"),
    ("reminder: don't commit the API keys to the repo.", "alert"),
    ("they mentioned they're also evaluating a competitor.", "alert"),
    ("the demo build is failing on the latest macOS.", "alert"),
    ("that feature was deprecated two releases ago.", "alert"),
    # topic — doc-matching fact being discussed
    ("right, the hybrid RAG uses FTS5 and vector fusion.", "topic"),
    ("the encoder is a bidirectional masked-language model.", "topic"),
    ("our model runs entirely on-device on Apple Silicon.", "topic"),
    ("the diarization is two-tier, source-based then neural.", "topic"),
    ("we use ChatML delimiters for the instruct model.", "topic"),
    ("the consent gate keeps notes local until export.", "topic"),
    ("section-aware chunking splits on markdown headers.", "topic"),
    ("the audio model is LFM2.5-Audio at 1.5 billion parameters.", "topic"),
    ("the dual-buffer design separates display from intelligence.", "topic"),
    ("Notion export is the only opt-in network egress.", "topic"),
    ("turn accumulation happens at the transcript buffer layer.", "topic"),
    ("embeddings are 1024-dimensional from the retriever.", "topic"),
    ("the re-ranker is heuristic, applied after fusion.", "topic"),
    ("we fall back to extraction bullets when generation is weak.", "topic"),
    # followup — nudge / opportunity
    ("we should probably circle back on the pricing model.", "followup"),
    ("let's make sure to ask them about their timeline.", "followup"),
    ("it might be worth mentioning the on-device privacy angle.", "followup"),
    ("we could follow up with a benchmark on their data.", "followup"),
    ("remind me to send them the architecture doc.", "followup"),
    ("maybe we bring up the conference-room use case next time.", "followup"),
    ("we ought to loop in legal before we promise that.", "followup"),
    ("worth flagging the latency win in the follow-up email.", "followup"),
    ("let's schedule a deeper technical dive with their team.", "followup"),
    ("we should get their eval set before the next call.", "followup"),
    ("it'd be good to demo the slides-into-RAG feature for them.", "followup"),
    ("we might want to propose a pilot on one workflow.", "followup"),
    ("we could offer to forge a specialist model for their task.", "followup"),
    ("worth suggesting they try the desktop app first.", "followup"),
    # none — filler / backchannel / smalltalk
    ("yeah, totally.", "none"),
    ("okay, sounds good to me.", "none"),
    ("right, right, exactly.", "none"),
    ("um, let me think about that for a sec.", "none"),
    ("can you hear me okay?", "none"),
    ("sorry, I was on mute.", "none"),
    ("let me share my screen real quick.", "none"),
    ("so anyway, where were we?", "none"),
    ("I think we lost you for a second there.", "none"),
    ("give me one moment, my dog is barking.", "none"),
    ("that's a great point, thanks.", "none"),
    ("no worries, take your time.", "none"),
    ("let's take a five minute break.", "none"),
    ("cool, cool.", "none"),
]


def resolve_model_path() -> Path:
    models_dir = os.environ.get("MODELS_DIR")
    for p in (
        (Path(models_dir).expanduser() / MODEL_DIRNAME) if models_dir else None,
        Path.home() / "Projects" / "_models" / MODEL_DIRNAME,
    ):
        if p and p.exists():
            return p
    raise SystemExit(f"Could not find {MODEL_DIRNAME}")


def main() -> int:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestCentroid
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    mp = resolve_model_path()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(str(mp), trust_remote_code=True).lfm2
    backbone.eval().to(device)

    @torch.no_grad()
    def embed(texts: list[str]) -> "np.ndarray":
        out = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64).to(device)
            hs = backbone(**enc).last_hidden_state  # [1,T,H]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            out.append(pooled.squeeze(0).cpu().numpy())
        return np.array(out)

    texts = [t for t, _ in DATA]
    labels = [y for _, y in DATA]
    print(f"model: {mp.name} · device: {device} · n={len(DATA)} · classes={sorted(set(labels))}")

    X = embed(texts)
    y = np.array(labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    print(f"train={len(ytr)} · test={len(yte)}\n")

    # Linear probe (frozen features + logistic regression fit in seconds)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    probe_f1 = f1_score(yte, pred, average="macro")

    # Zero-fit contrast: nearest centroid (no learned weights)
    nc = NearestCentroid()
    nc.fit(Xtr, ytr)
    nc_f1 = f1_score(yte, nc.predict(Xte), average="macro")

    floor = max(np.mean(yte == c) for c in set(y))  # majority-class accuracy

    print("--- linear probe (frozen encoder + logistic regression) ---")
    print(classification_report(yte, pred, digits=3, zero_division=0))
    print("--- summary ---")
    print(f"majority-class floor (acc):     {floor:.3f}")
    print(f"nearest-centroid  macro-F1:     {nc_f1:.3f}   (zero-fit contrast)")
    print(f"linear probe      macro-F1:     {probe_f1:.3f}   (frozen encoder, no model training)")
    print("\nRead: probe >> floor/centroid → the frozen encoder already separates the")
    print("classes linearly; LoRA fine-tuning would lift from here. Small synthetic set →")
    print("directional ceiling, not a production number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
