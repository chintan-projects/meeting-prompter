# STATUS — overnight Liquid re-architecture (Wave A, code-only)

Branch: `liquid-rearch` (off `main`) · **unpushed** · Date: 2026-07-20 (overnight)
Scope tonight: **code waves only** (Wave A + F-510 probe). No head training, no external
egress, no GPU. `pytest -q` kept green after every change.

## Prereqs (verified before starting)
- ✅ `./venv` present; `pytest -q` GREEN on clean `main` (585 passed).
- ✅ Local models on disk: `LFM2.5-Encoder-350M`, `LFM2.5-Embedding-350M`,
  `LFM2.5-350M-Extract-023-v1`, `LFM2.5-1.2B-Instruct` (under `~/Projects/_models`).
- Note: `ruff`/`mypy` are NOT installed in `./venv`; pytest is the enforced gate.
  Auto-format hooks (black/prettier) ran on write.

## Retrieval eval — MiniLM vs LFM2.5-Embedding-350M
| Retriever | Hit@1 | Hit@3 | MRR | Neg-max conf |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (was) | 94.4% | 100% | 0.972 | 0.34 |
| LFM2.5-Embedding-350M (now) | 94.4% | 100% | 0.972 | 0.28 |

Ties exactly (doc-level eval saturates — F-509 hardens it to gain discriminating power).

## Per-feature outcome (Wave A)

### F-502 — retrieval swap → LFM2.5-Embedding-350M — ✅ DONE
Config-driven embedder (`rag.embedding_model` / `embedding_dimension`). Default now the
Liquid retriever (1024-d, local + trust_remote_code); MiniLM still selectable by hub id.
Added a stale-dimension index purge so a model swap can't silently break vector unpack.
GATE MET: Hit@1=94.4% / MRR=0.972 (do-not-regress). Commit `8e5fade`.

### F-509 — harden the retrieval eval — ✅ DONE
New chunk-level harness over a confusable synthetic corpus (3 near-identical router
cards, 2 deploy tiers). Each query names a distinctive fact present only in the correct
chunk. Added asymmetric query/passage prompt support (`embed_query` + document prompt),
backward-compatible. Margin on the confusable corpus (chunk-level; doc-level = 100% all):

| Retriever | chunk@1 | chunk@3 | chunkMRR | neg-max |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 87.5% | 87.5% | 0.887 | 0.178 |
| LFM2.5-Embedding (symmetric) | 81.2% | 93.8% | 0.875 | 0.168 |
| LFM2.5-Embedding (query/passage prompts) | 81.2% | 93.8% | 0.865 | 0.215 |

Honest read: within one query of each other (16 positives) — no clear winner; prompts
gave no lift and slightly worsened negative separation. The swap stays "safe, not proven
better." The harness now has real discriminating power (chunk-level < doc-level). Commit `af7dac9`.

### F-501 — EncoderIntelligenceLayer + typed turn-state — ✅ DONE
New `lib/intelligence/`: `TurnState` (typed per-turn carrier), `Head` interface,
`HeuristicHead` (wraps alert/question/topic triggers as the first heads),
`EncoderBackbone` (warm LFM2.5-Encoder-350M, mean-pool, lazy), `EncoderIntelligenceLayer`
(runs heads priority-sorted, one optional mean-pooled forward/turn). `TriggerEngine` is now
a thin adapter over the layer — order + per-head error isolation preserved. Encoder embedding
off by default → heuristic path never loads the model → **zero behavior change**. Fixed an
engine↔intelligence import cycle via PEP-562 lazy export. Encoder smoke: loads, 1024-d
vectors. 615 fast tests green. Commit `8b4dd72`.

### F-510 — frozen-encoder linear-probe head — ✅ DONE (probe wired-but-OFF, by decision)
Logistic probe on frozen mean-pooled LFM2.5-Encoder-350M embeddings of the 5-way
trigger set. Wired into `TriggerEngine` via a lazy encoder (loads nothing while off).
Honest gate (frozen held-out, per-class F1):

| | macro-F1 | question | alert | topic | followup | none |
|---|---|---|---|---|---|---|
| linear probe | **0.886** | 0.92 | 0.80 | 0.80 | 0.91 | 1.0 |
| heuristic | 0.286 | **1.0** | 0.0 | 0.0 | 0.0 | 0.43 |

Decision: **keep heuristic default, probe wired-but-off.** The probe only wins on macro-F1
because the heuristic has no isolated-text mechanism for alert/topic/followup; it *regresses*
the question class the heuristic owns (0.92 < 1.0) and rests on 70 synthetic examples. The
conservative rule (beats macro-F1 AND no question regression) → off. F-503 forge-LoRA on real
labels is the measured upgrade. Also caught + fixed a train/test **data leak** (re-persisting the
seed into the overlay double-counted rows → inflated to 1.0; load now dedups). Commit `e6eaea4`.

### F-601 — AttributionResolver hierarchy — ✅ DONE
New `lib/attribution/`: resolver composing L1 channel (mic→You / system→Others, ground
truth), L3 acoustic (diarization estimate), L4 roster (name overrides), + regime detection
(solo-endpoint vs conference-room) with honest degradation. `session.py` routes all speaker
decisions through it; meeting-context participants seed the L4 roster. Default regime UNKNOWN
→ behavior unchanged (81 session/diarization tests pass). Commit `4ac3e12`.

### F-606 — conference-room honest degradation — ✅ DONE
`conference_room: true` in meeting context → resolver enters CONFERENCE_ROOM regime →
diarized system turns collapse to a flagged `Others (room)` bucket. A `low_confidence` flag
travels Turn → TranscriptStore → `transcript_final`/`transcript_relabeled` WS messages → the
React transcript (a "~ best guess" badge). WS protocol contract + meeting-context template
updated. 645 fast tests green; tsc + frontend green. Commit `a105ec3`.

### F-604 — acoustic diarization fix — ✅ DONE
Speaker-change segmentation within a turn (`process_turn_segments`: window → embed →
change-point detection → per-slice assignment; `process_turn` returns the dominant speaker,
backward-compatible for single-speaker turns) + roster-bounded clustering (`set_roster_size`
caps clusters at the known participant count with nearest-cluster re-assignment). Session seeds
the roster from meeting context. Validated with synthetic-embedding fixtures. 656 fast tests
green. Commit `3ca884d`.

### F-605 — voice-enrollment mechanism — ✅ DONE (mechanism; record-flow deferred)
New `lib/speaker_enrollment.py`: `VoiceEnrollment` store (name→profile embeddings, nearest-
profile identify, re-enroll averaging, local JSON persistence). Diarizer `set_enrollment` names
matching clusters with the real enrolled name (L4) instead of "Speaker A" (parallel `_names`
list, anonymous fallback). Session loads from `diarization.enrollment_path`. MeetingSetup shows
an honest enrollment note; the live record-a-voice capture flow needs a live session (deferred,
noted in NEEDS-HUMAN). 670 fast tests green; tsc clean. Commit `fd27999`.

### F-506 — route-first hot/cold execution — ✅ DONE
Heads tagged hot vs cold. Hot heads (alert, question — cheap regex) run every turn; the cold
path (topic, RAG-backed) runs only for substantive turns (≥ `cold_path_min_words`), so filler
no longer fires a RAG query. Routing decision recorded on shared `TurnState.ran_cold`. Config-
driven (default 3 words). 673 fast tests green. Commit `d33eef1`.

### F-507 — structured notes via LFM2.5-350M-Extract — ✅ DONE (path built; live promotion deferred)
`StructuredNotes` typed schema + Extract-style prompt (YAML field schema in the system turn) +
robust YAML parser + deterministic markdown renderer — separates structure (model) from rendering
(code). `generate_structured_notes` prefers the extractor and falls back to the instruct path on
empty/failed output. Notes route passes `orchestrator.extract_generator`, which is None today
(instruct path unchanged); pointing it at the real Extract GGUF + verifying quality is a live run.
684 fast tests green. Commit `7a49a0e`.

### F-508 — persistent warm-model runtime — ✅ DONE (registry + encoder; ASR server deferred)
New `WarmModelRuntime`: single owner of load-once models. Constructs the encoder backbone once
and shares it (orchestrator no longer builds it ad-hoc), registers the embedder + instruct
generator for status/teardown, with `warm()`/`status()`/`teardown()` lifecycle. Encoder stays
lazy → unchanged startup cost. ASR still spawns a llama.cpp subprocess per chunk; a persistent
audio server is the remaining step (audio CLI is one-shot) — needs a live run. 692 fast tests
green. Commit `f173f22`.

### F-607 — lexical speaker-consistency pass — ✅ DONE
Model-free correction layer: named hand-off cues ("over to you, Priya") name the next remote
turn; gratitude cues ("thanks, Priya") name the previous one. Only relabels generic remote
turns (Others / Speaker N), scoped to roster first names, never overrides a set name
(confidence 0.6). Wired non-destructively at notes time (`correct_segments`, flags corrected
turns low_confidence). 704 fast tests green. Commit `83b65a2`.

## WAVE B — head training — PARKED (not attempted, per brief)
F-503 / F-504 / F-505 (forge-LoRA encoder heads) were **not** attempted tonight. Reason
(from the brief): the ADR requires encoder-backed heads, but forge's trainer templates load
the base causally today — a true bidirectional read is forge's own pending P1.1 work
(`compat.py`). That forge change must land + verify first. F-510's frozen-encoder probe is the
non-training v1 that decouples encoder intelligence from that forge work.

## PARK — genuinely blocked (not attempted; need a real input)
- **F-603** VLM visual context — needs real speaker-view + shared-slide screenshots (ADR gates on a real-image test).
- **F-608** Zoom SDK high-fidelity mode — needs a Zoom dev account + Meeting SDK creds.
- **F-602** a11y roster reader — needs a live Zoom/Teams window to map the AX tree.

## Final status
- `pytest -q`: **704 passed** (was 585 on clean main; +119 new tests). Green after every change.
- `pytest -m slow`: **10 passed** (RAG eval, hardened chunk eval, probe gate).
- Frontend: `tsc --noEmit` clean; **16** frontend tests pass.
- No external egress; no head training; no push. Branch `liquid-rearch` unpushed, off `main`.

## Commit list (`liquid-rearch`, oldest → newest)
```
8e5fade F-502  swap retrieval to LFM2.5-Embedding-350M behind embedder
af7dac9 F-509  hardened chunk-level retrieval eval + query/passage prompts
8b4dd72 F-501  EncoderIntelligenceLayer + typed turn-state behind Head interface
e6eaea4 F-510  frozen-encoder linear-probe head (wired-but-off, gated)
4ac3e12 F-601  AttributionResolver — hierarchy + regime + honest degradation
a105ec3 F-606  conference-room honest degradation + low-confidence flag
3ca884d F-604  acoustic diarization — speaker-change segmentation + roster-bound
fd27999 F-605  voice-enrollment mechanism — name clusters from profiles
d33eef1 F-506  route-first hot/cold execution on shared turn-state
7a49a0e F-507  structured notes via LFM2.5-350M-Extract path
f173f22 F-508  persistent warm-model runtime
83b65a2 F-607  lexical speaker-consistency correction pass
```

## NEEDS-HUMAN (prioritized)
1. **Forge bidirectional-encoder change** (blocks all head training). Land + verify the
   forge encoder-classifier path (targets.py encoder variants, loopb.py bidirectional load,
   test_backbones.py) in the finetune-quality monorepo BEFORE F-503/504/505. Until then the
   F-510 probe stays the v1 (wired-but-off; heuristic remains default).
2. **Live call** to validate: dual-stream + F-604 segmentation + F-601/F-606 attribution +
   F-607 corrections on real audio; and to promote F-507 (point notes at the real Extract
   GGUF and verify quality) and F-508 (persistent ASR server vs subprocess-per-call).
3. **Voice-enrollment capture UI** (F-605): the record-a-voice flow needs a live session;
   the store + matching + persistence are done and unit-validated.
4. **Real Zoom speaker-view + shared-slide screenshots** to run the VLM real-image test that
   gates F-603.
5. **Zoom Meeting SDK creds / dev account** for F-608; **live Zoom window** for F-602.

## Notes
- `ruff` / `mypy` are not installed in `./venv`; pytest was the enforced gate. Auto-format
  hooks (black/prettier) ran on write. Recommend adding ruff+mypy to the venv for the lint gate.
- One pre-existing benign warning remains (`pytest.mark.slow` unregistered) — left as-is to
  avoid introducing global pytest config mid-run.
