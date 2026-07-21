# Overnight autonomous brief — Liquid re-architecture (code waves; head training deferred)

Mission (TONIGHT): execute the **code waves** of the Liquid re-architecture ADR
unattended — the full refactor (Wave A). **No head training tonight** — that is a
separate future forge session, gated on a prerequisite (see Wave B). Do as much of
Wave A as passes its gate; park what is blocked; never halt the whole run because
one item is blocked. Leave a branch + a morning report for review.

SPEC (source of truth):
- `docs/architecture/liquid-rearchitecture.md` + `FEATURES.yaml`
- Training MUST follow the `forge` skill (`~/.claude/skills/forge/SKILL.md`) and the
  skills it references (`liquid-models-architecture`, `liquid-finetuning-playbook`,
  `data-distribution-engineering`, `model-heads-and-decoding`).

## Operating rules (non-negotiable, safety-by-construction)

- Branch `liquid-rearch` off main. Work only there. NEVER push, NEVER touch main, NEVER force.
- Project venv `./venv` (transformers 4.56.2 — do NOT upgrade; encoder + embedding load via `trust_remote_code`).
- After EVERY code change: `pytest -q` must stay green. Can't fix in 2 tries → revert THAT change, log it, continue. Never leave the tree red.
- Commit after each feature/head passes its gate. Small commits, clear messages.
- delete-as-you-replace: a head that beats its heuristic REPLACES it (delete heuristic + dead tests, add new tests). Never two parallel impls.
- **Local only tonight — NO external egress.** No OpenRouter, no Notion, no GPU host: tonight is code, and all models are already on disk. (Those are used only by the future forge session below.)
- Per-item stop-and-park: blocked on a missing input → log in `STATUS-overnight.md`, move on. NEVER fake inputs.

## Prereqs — tonight (verify FIRST; record in STATUS)

Tonight is code only — no training — so it needs NO OpenRouter / Notion / GPU access.

1. Project venv `./venv` present; `pytest -q` green on a clean checkout before starting.
2. Local models on disk: `LFM2.5-Encoder-350M`, `LFM2.5-Embedding-350M` (both used by
   Wave A code paths, loaded via `trust_remote_code`).

(OpenRouter teacher key, Notion access, and GPU reachability are prereqs for the
FUTURE forge training session below — NOT for tonight.)

## WAVE A — code (full ADR, no external inputs). In order, each test-gated.

- **F-502** retrieval swap → `LFM2.5-Embedding-350M` behind `embedder.py` + config. GATE: `tests/eval` holds Hit@1 ≥ 94.4% / MRR ≥ 0.972.
- **F-509** harden the eval: chunk-level labels, 2–3 confusable fixture docs, query/passage prompts. Record MiniLM-vs-LFM margin.
- **F-501** `lib/intelligence/`: `EncoderIntelligenceLayer` (LFM2.5-Encoder-350M via `AutoModelForMaskedLM`→`.lfm2`, mean-pool, warm) + typed `turn_state`; heuristics kept as `Head` impls behind a `Head` interface. GATE: zero behavior change, suite green.
- **F-510** frozen-encoder linear-probe head (v1, NO GPU/forge/egress — fits tonight): persist the labeled set from `scripts/spike_encoder_linear_probe.py` to `data/fixtures/trigger_probe_dataset.jsonl` (gitignored; may hand-add more inline examples — NO teacher/egress), fit a logistic-regression probe on frozen mean-pooled embeddings, eval on a frozen held-out split. GATE: wire the probe as the default trigger `Head` ONLY if it beats the heuristic on held-out (report per-class); else keep the heuristic as default, leave the probe wired-but-off with its numbers. Encoder-only, never causal.
- **F-601** `lib/attribution/resolver.py`: hierarchy (L1 channel, L3 acoustic wrap, L4 roster map from meeting_context, regime detection) + honest degradation. Refactor `session.py` behind it.
- **F-606** conference-room degradation (Others-room bucket / flagged low-confidence).
- **F-604** acoustic diarization fix: speaker-change segmentation, roster-bounded clustering + re-assignment. Validate with synthetic/unit fixtures.
- **F-605** voice-enrollment mechanism (unit-validated).
- **F-506** route-first hot/cold execution + shared turn-state.
- **F-507** notes → `LFM2.5-350M-Extract` (structured fields).
- **F-508** persistent warm-model runtime (load encoder + embedder + instruct once).
- **F-607** lexical speaker-consistency pass.

## WAVE B — head training: PARKED tonight (prerequisite pending)

DO NOT train any head tonight. Reason: the ADR requires encoder-backed heads, and
forge's trainer templates currently load the base causally (a true bidirectional
read is forge's own pending P1.1 work — `compat.py`). That forge change is a
separate, tested task that must land + verify FIRST. Until then, training any head
would mean using the causal tower, which the ADR forbids. So: park F-503/F-504/F-505
entirely tonight. Do the code waves (Wave A) only.

Do not run any of the forge steps below tonight. They are recorded here only as the
spec for the FUTURE dedicated session, and are gated on a prerequisite.

────────────────────────────────────────────────────────────────────────────
FUTURE SESSION ONLY — encoder-head training via forge. NOT part of tonight's run.
────────────────────────────────────────────────────────────────────────────

PREREQUISITE (must land + verify FIRST, in the finetune-quality monorepo):
the forge bidirectional-encoder-classifier change — additive encoder head variants
in `targets.py` (base=LFM2.5-Encoder-350M, backbone=encoder, mean-pool), the
bidirectional loading path in the `loopb.py` sequence/token trainer templates
(`AutoModelForMaskedLM`→`.lfm2`, no causal mask), the compat stamp (already returns
bidirectional for encoder+base), and updated `test_backbones.py`. Verify with the
forge test suite + `--mock` dataloop + a dry-run plan + a real MPS smoke-train —
all BEFORE any GPU.

HARD RULE: **encoder backbone only — NEVER causal.** All three heads train on the
bidirectional LFM2.5-Encoder-350M via the new encoder variants:
- F-503 trigger router — one label ∈ {question, alert, topic, followup, none}.
- F-505 quality gate — K independent flags.
- F-504 evidence span — one label per token (BIO), its own LoRA, never co-trained.
If any head would resolve to `backbone="causal"`, DO NOT train it.

DATA: synthetic-first (forge Loop A teacher generation is the primary corpus).
Notion notes are OPTIONAL conditioning only — never a blocker.

Per-head (forge SKILL.md): S1 contract → S2 (select the ENCODER variant, never the
causal default) → S4 golden LoRA recipe (r=16/α=32, mean-pool, real LFM2
`target_modules`) → author `genspec.yaml` → Loop A to GREEN (never past RED/HALT;
≤3 genspec revisions else park) → `corpuscope approve` (GREEN only) → Loop B
`forge train --execute` (full-auto GPU) → wire in ONLY if it beats the heuristic on
a frozen held-out eval (else keep the heuristic + the artifact). Guardrails: GREEN
before GPU (no exceptions), never override a gate, mean-pool, real LFM2
`target_modules`, frozen holdout, honest per-class reporting.

## PARK — genuinely blocked (log reason, do NOT attempt)

- **F-603** VLM visual context — needs real screenshots (ADR gates on a real-image test).
- **F-608** Zoom SDK mode — needs Zoom dev account + creds.
- **F-602** a11y roster reader — needs a live Zoom window to map the AX tree.
- Upgrading transformers; pushing; touching main / VL models.

## MORNING REPORT — `STATUS-overnight.md` at repo root

Tonight is CODE WAVES ONLY (no head training — that is the future forge session).
Report: per-feature outcome for Wave A (done / partial / parked + why); final
`pytest -q` status; retrieval eval deltas (MiniLM vs LFM, before/after hardening);
the commit list on the `liquid-rearch` branch; and a prioritized **NEEDS-HUMAN**
list (the forge bidirectional-encoder change before any head training; real
screenshots for VLM; live call; Zoom creds). Update `PROGRESS.yaml`. Leave the
branch unpushed for review.
