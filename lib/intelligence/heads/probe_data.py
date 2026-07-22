"""Labeled seed set for the frozen-encoder linear-probe head (F-510).

The 5-way trigger-routing task: each utterance is one of
{question, alert, topic, followup, none}. This seed set is the same deliberately-
varied synthetic set the Stage-0 spike measured (macro-F1 0.886 off-the-shelf).
It is committed so the probe is reproducible and testable; additional hand-added
examples may live in ``data/fixtures/trigger_probe_dataset.jsonl`` (gitignored,
NO teacher/egress) and are merged on load.

Synthetic + small → a DIRECTIONAL ceiling, not a production number. Real labeled
meeting data (Stage 3 / forge) is the measured upgrade over this.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

PROBE_LABELS: Tuple[str, ...] = ("question", "alert", "topic", "followup", "none")

# (text, label) — canonical seed set (origin: scripts/spike_encoder_linear_probe.py).
SEED_EXAMPLES: List[Tuple[str, str]] = [
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

# Gitignored overlay: hand-added examples (no teacher/egress). Merged if present.
OVERLAY_PATH = Path("data/fixtures/trigger_probe_dataset.jsonl")


def load_probe_examples(
    overlay_path: Path = OVERLAY_PATH,
) -> List[Tuple[str, str]]:
    """Return the seed set plus any valid examples from the gitignored overlay.

    Deduplicated by (text, label): the overlay is often a persisted copy of the
    seed with hand-adds appended, so a naive merge would double-count rows and
    leak identical utterances across a train/test split. Dedup keeps first-seen
    order and makes the loader idempotent w.r.t. re-persisting the seed.
    """
    seen: set[Tuple[str, str]] = set()
    examples: List[Tuple[str, str]] = []

    def _add(text: str, label: str) -> None:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            examples.append(key)

    for text, label in SEED_EXAMPLES:
        _add(text, label)

    if overlay_path.exists():
        for line in overlay_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                text, label = row["text"], row["label"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping bad probe overlay line: %s (%s)", line[:60], exc)
                continue
            if label in PROBE_LABELS and isinstance(text, str) and text.strip():
                _add(text, label)
    return examples


def write_seed_dataset(path: Path = OVERLAY_PATH) -> int:
    """Persist the seed set to the gitignored JSONL (the F-510 'persist' step).

    Returns the number of rows written. Idempotent — overwrites the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for text, label in SEED_EXAMPLES:
            f.write(json.dumps({"text": text, "label": label}) + "\n")
    return len(SEED_EXAMPLES)
