# F-503 trigger-router — results & reproduction

Encoder-backed trigger router (LFM2.5-Encoder-350M, bidirectional, mean-pool, CE),
LoRA-trained via forge on a **shape-first** synthetic corpus and gated on **real
meeting transcript turns**. The winning configuration is a hybrid: F-503 primary +
the heuristic question detector as a rescue.

## Design principle (why it generalizes)

Teach the **pragmatic shape** (question / alert / topic / followup / none as speech
acts), never the **content**. The corpus spans ~30 unrelated domains (hospital,
construction, shipping port, …) so no content feature predicts the label — only the
shape does. Downstream layers still own content (RAG for topic, watch-words for alert).
See `contract.md` and `genspec.yaml`.

## Iterations (the honest arc)

| ver | teacher | change | real-transcript macro-F1 |
|-----|---------|--------|--------------------------|
| v1  | gpt-4o-mini | first pass | — (templated collapse caught on inspection: followup 89% "we should") |
| v2  | gpt-4o | shape-first, diverse domains | **0.596** (alert 0.22 — missed marker-free alerts) |
| v3  | gpt-4o | alerts forced marker-free (valence in content), richer followups | **0.822** (alert 0.92) |
| v3 + hybrid | — | heuristic question-rescue | **0.846** (question 0.80→0.88) |

The synthetic in-distribution holdout was ~1.0 throughout — uninformative. The real
39-turn Mercedes set (hand-labeled by the contract, held local) was the decisive gate.

## Final gate (real held-out, 39 turns)

| router | macro-F1 | question | alert | topic | followup | none |
|--------|----------|----------|-------|-------|----------|------|
| heuristic | 0.264 | 0.89 | 0.00 | 0.00 | 0.00 | 0.43 |
| F-510 probe | 0.547 | 0.48 | 0.60 | 0.00 | 0.78 | 0.88 |
| F-503 v3 | 0.822 | 0.80 | 0.92 | 0.88 | 0.71 | 0.80 |
| **F-503 + hybrid** | **0.846** | 0.88 | 0.92 | 0.88 | 0.71 | 0.84 |

On the frozen out-of-domain synthetic held-out (`tests/eval/f503_trigger_router_eval.yaml`)
F-503 v3 scores macro-F1 0.897 and ships the strict gate outright.

## Reproduce

```bash
# 1. data loop (OpenRouter teacher) → GREEN, then approve
forge dataloop --spec genspec.yaml --out run/ --max-rounds 5
corpuscope approve run/train.jsonl
# 2. train on GPU (bidirectional encoder classifier)
forge train --spec genspec.yaml --run run/ --out artifact/ \
  --head sequence_classifier_encoder --size 350M --execute
# 3. gate: F-503 vs probe vs heuristic (synthetic held-out)
python3 artifact/eval_clf_*.py --checkpoint artifact/checkpoint \
  --eval heldout_eval.jsonl --base $MODELS_DIR/LFM2.5-Encoder-350M --out heldout_f503.json
MODELS_DIR=~/Projects/_models python3 ../../tests/eval/f503_gate.py \
  --heldout ../../tests/eval/f503_trigger_router_eval.yaml --f503-json heldout_f503.json
```

Deliverable: `LFM2.5-TriggerRouter-350M` (LoRA adapter) in the model registry, loaded
by `lib/intelligence/heads/trigger_router.py`. Wired config-gated
(`triggers.f503_router_enabled`, default OFF pending the WS-14 live call).
