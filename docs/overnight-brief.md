# Overnight autonomous brief — Liquid re-architecture (full ADR + forge training)

Mission: execute the entire Liquid re-architecture ADR unattended, and train the
encoder heads end-to-end via the **forge** pipeline in **full-auto (incl. remote
GPU)**. Do as much of every stage as its inputs allow; bank what passes its gate;
park only what is genuinely blocked. Never halt the whole run because one item is
blocked. Leave a branch + a morning report for review.

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
- Local only for code. The ONLY external calls allowed: forge's teacher (OpenRouter), the Notion API (read-only), model downloads already on disk, and the forge GPU host. No other egress.
- Per-item stop-and-park: blocked on a missing input → log in `STATUS-overnight.md`, move on. NEVER fake inputs.
- Secrets (`OPENROUTER_API_KEY`, GPU SSH key, `NOTION_API_TOKEN`) are read from their configured paths — NEVER print them.

## Prereqs — verify FIRST; gate dependent work if missing

Run these checks at start and record results in STATUS. Do not fake around a miss.

1. `forge targets` + `corpuscope --help` → toolchain ready (was ready at authoring time).
2. `OPENROUTER_API_KEY` present → forge Loop A teacher. **If missing: park ALL training (Wave B), do Wave A only, report loudly.**
3. Notion access (below). If unreachable: Wave B proceeds with teacher-only generation but **flag that real-data grounding was skipped**.
4. GPU reachability (forge train dry-run emits `train_plan.sh`). **If GPU unreachable: stop each head at `corpuscope approve` + `train_plan.sh`, park the trained artifact for morning, do NOT loop retrying.**

## Data acquisition — Notion meeting notes (grounds the training distribution)

The notes are real meeting *content*, used to make forge's authored data look like
real Liquid meetings — NOT as ready-made labels.

- Source DB: `673508f9ad4645389052980ec0770501` (workspace liquidai).
- Fetch read-only via the project's own client (`lib/notion/`): set `notion.enabled: true`,
  add the DB id to `rag_source_database_ids`, use `NOTION_API_TOKEN`. Convert pages →
  markdown (`lib/notion/parser.py`).
- Store gitignored under `data/fixtures/notion/` — real participant data, NEVER committed.
- Mine realistic utterances/turns from the notes → use as forge `families` / `scenarios`
  (conditioning by default; `in_band` only if the served input carries it) so the teacher
  generates on-distribution examples. Optionally weak-label a subset as a bring-your-own
  corpus (`corpuscope audit … --spec …`).

## WAVE A — code (full ADR, no external inputs). In order, each test-gated.

- **F-502** retrieval swap → `LFM2.5-Embedding-350M` behind `embedder.py` + config. GATE: `tests/eval` holds Hit@1 ≥ 94.4% / MRR ≥ 0.972.
- **F-509** harden the eval: chunk-level labels, 2–3 confusable fixture docs, query/passage prompts. Record MiniLM-vs-LFM margin.
- **F-501** `lib/intelligence/`: `EncoderIntelligenceLayer` (LFM2.5-Encoder-350M via `AutoModelForMaskedLM`→`.lfm2`, mean-pool, warm) + typed `turn_state`; heuristics kept as `Head` impls behind a `Head` interface. GATE: zero behavior change, suite green.
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

--- (retained for when the forge bidirectional-encoder change has landed) ---
HARD RULE: **encoder backbone only — NEVER the causal head.** Follow the ADR, not
forge's default S2 (which maps sequence/multi-label to the causal tower). If a head
would resolve to `backbone="causal"`, DO NOT train it — park it. Do not "fall back"
to causal to make the run complete.

Data is **synthetic-first**: forge's Loop A teacher generation is the primary corpus.
Notion notes are OPTIONAL conditioning only (if reachable, mine utterances into
`families`; if not, pure synthetic — never a blocker).

Heads:
- **F-504 evidence span** — one label per token (BIO). forge's existing `token_classifier`
  is already **encoder** (its own LoRA, never co-trained). Train this tonight.
- **F-503 trigger router** (one label ∈ {question, alert, topic, followup, none}) and
  **F-505 quality gate** (K flags) — these need an **encoder-backed** sequence/multi-label
  head. forge does NOT ship that yet (its `sequence_classifier`/`multi_label` are causal).
  GATE: train these ONLY if the forge encoder-classifier change (targets.py base+backbone,
  loopb dispatch, bidirectional loading in train_sequence_classifier_meanpool.py, compat
  stamp, updated tests) has been LANDED and VERIFIED (forge test suite green + `--mock`
  dataloop + dry-run plan builds with the encoder base). If NOT verified, PARK both heads
  — do not train them on causal.

Per-head procedure (forge):
1. **S1 contract** — numbered selection rules, exclusion zones, tie-breaks, explicit abstain/`null`.
2. **S2 head/backbone/size** — derive from output shape via forge's table. NOTE: forge puts one-label/K-flag heads on the **causal 350M** and token-labels on the **encoder** — this DIVERGES from the ADR's "one shared encoder" assumption. Follow forge, and RECORD the divergence + per-head eval in STATUS so we can compare encoder-vs-causal for the sequence heads in the morning.
3. **S4 recipe** — SFT/LoRA golden recipe (r=16/α=32, mean-pool, real LFM2 `target_modules` `[q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3]`). forge encodes this.
4. **Author `genspec.yaml`** — labels (with definitions/boundaries on subtle classes), families grounded in the Notion notes, two-sided cues for the confusable discriminators.
5. **Loop A**: `forge dataloop --spec genspec.yaml --out run/` (autonomous, no GPU). Open `run/scorecard.html`.
   - **GREEN** → proceed. **RED** → inspect offenders, fix the *distribution*, re-run (bounded: ≤3 genspec revisions). **HALT** → the loop printed a *witness*; **change the genspec it names — re-running is futile**. If not GREEN after the revision budget, **park this head** with its scorecard + offenders and move to the next. NEVER proceed past RED/HALT.
6. **Approve (GREEN only)**: `corpuscope approve run/train.jsonl` — the mechanical GREEN-before-GPU gate.
7. **Loop B (GPU, full-auto)**: `forge train --spec genspec.yaml --run run/ --out artifact/ --head <h> --size <s>` to emit the plan; heed any `⚠ plan lint`; `cat artifact/train_plan.sh` (paths absolute, no `~`/`$HOME`); then `forge train … --execute`. Tail the remote log. On eval-gate failure: **fix the data, re-run Loop A** — do not tune weights.
8. **Wire-in GATE**: replace the heuristic with the trained head ONLY if it beats the heuristic on the held-out eval (macro-F1 / per-class). Else keep the heuristic + keep the artifact + numbers. Either way commit the pipeline + eval. Follow delete-as-you-replace when it wins.

Guardrails (forge, encode as steps — never skip): GREEN before GPU, no exceptions. Never override a gate. Mean-pool on LFM2.5; refuse last-token. Real LFM2 `target_modules` only. Carve a frozen holdout (`holdout_frac`). Report per-class numbers honestly — "done" = gated-and-verified, not "it ran".

## PARK — genuinely blocked (log reason, do NOT attempt)

- **F-603** VLM visual context — needs real screenshots (ADR gates on a real-image test).
- **F-608** Zoom SDK mode — needs Zoom dev account + creds.
- **F-602** a11y roster reader — needs a live Zoom window to map the AX tree.
- Upgrading transformers; pushing; touching main / VL models.

## MORNING REPORT — `STATUS-overnight.md` at repo root

Per-feature outcome (done / partial / parked + why); final `pytest -q` status;
retrieval eval deltas (MiniLM vs LFM, before/after hardening); for EACH head:
Loop A terminal state (GREEN/RED/HALT), scorecard path, whether trained, eval vs
heuristic, wired-in?; the encoder-vs-causal backbone note for the sequence heads;
the commit list on the branch; and a prioritized **NEEDS-HUMAN** list (real
screenshots, live call, Zoom creds, any RED/HALT corpora to inspect). Update
`PROGRESS.yaml`. Leave the branch unpushed for review.
