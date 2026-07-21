# Liquid Re-Architecture — meeting-prompter

Status: **Stage 0 complete (spikes run); Stages 1–4 proposed** · Date: 2026-07-20 · Owner: Chintan

## Thesis

One line: **use the right model for each output shape.** Two misalignments, both
introduced when the app was built before we understood the Liquid lineup:

1. **Intelligence is mostly classification / extraction, not generation** — yet
   triggers run on regex heuristics and answers on a generic decoder. The new
   `LFM2.5-Encoder-350M` (bidirectional masked-LM, strong token-level heads) is
   the correct backbone for that work.
2. **Speaker attribution is mostly integration / sensing, not acoustic guessing**
   — yet we lean on turn-level ECAPA clustering as the primary mechanism. The
   strongest signals (channel source, platform SDK) are deterministic; acoustic
   diarization is a fallback, not the lead.

This is a **re-alignment, not a rewrite** — see the decision below.

## Decision: refactor the current project (not a new one)

**Refactor via strangler-fig, inside the current repo.** The system skeleton —
dual-buffer fork (display vs intelligence), turn accumulation, per-source silence,
thread safety, WebSocket contract, consent-gated export, Tauri UI, RAG pipeline —
is sound and model-agnostic. Every change sits **behind an existing seam**
(`embedder.py`, the trigger engine, `session.py` tier logic, `lfm2_wrapper.py`),
so organs are swapped one at a time, each gated by the ~585-test suite, always in
a working state.

- **Cleaner:** a new project is clean only because it is empty; it re-accumulates
  every solved complication (dual-stream threading, ScreenCaptureKit capture, WS
  reconnect, turn edges). This codebase is already modular (<300-line files) and
  tested — mess in refactors comes from bad structure, which this lacks.
- **Faster:** reuses working code and the regression net; green-field re-derives
  the plumbing before arriving at the same place.
- **Discipline that keeps it clean:** *delete-as-you-replace.* When a head lands,
  the heuristic it supersedes is **deleted**, not left beside it. Two parallel
  implementations is the real refactor trap — avoided by policy.
- New clean packages (`lib/intelligence/`, `lib/attribution/`) live *inside* the
  repo; superseded modules are removed. New code, same tests.

A new project would only be right if the skeleton were wrong or the stack changed
fundamentally (e.g. collapsing to a single omni-model). Neither holds.

## Stage 0 — decision-spike findings (2026-07-20)

| Spike | Result | Consequence |
|---|---|---|
| **Encoder smoke** | ✅ GREEN — clean weight load (via `AutoModelForMaskedLM` → `.lfm2` backbone), **~14 ms / turn** mean-pooled on MPS, 1024-dim | Encoder intelligence layer is feasible with large latency headroom. Build it. |
| **Frozen-encoder probe** | ✅ off-the-shelf macro-F1 **0.886** on the 5-way trigger task (frozen encoder + logistic regression; 0.85 zero-fit centroid; 0.21 floor; 70-utt synthetic) | The encoder already separates the classes linearly. A **no-GPU / no-forge linear-probe head is a viable v1** (F-510); forge-LoRA becomes a measured upgrade over it, not a from-scratch bet. |
| **Retrieval swap** | ⚠️ SAFE but unproven-better — `LFM2.5-Embedding-350M` **ties** MiniLM exactly (Hit@1 94.4 / Hit@3 100 / MRR 0.972). Eval saturates (doc-level, ~3 distinct docs) so it **can't discriminate** retrievers | Swap passes "do not regress." Must **harden the eval** (chunk-level, more/confusable docs, query/passage prompts) to claim improvement. |
| **Encoder vs Embedding geometry** | ✅ cos 0.855 (encoder) vs 0.095 (embedding) on unrelated text | Confirms the thesis empirically: Encoder→heads, Embedding→retrieval. Never swap. |
| **a11y active-speaker** | ❌ near-dead-end — no documented "speaking" AX attribute on Zoom/Teams/Meet; no known project reads it; a11y gives **names + own mute state only** | a11y demoted to **roster-name enrichment (L4)**. Not a primary signal. |
| **Platform path (research)** | Zoom Meeting SDK exposes ground-truth active speaker + per-participant PCM, **locally**, host/co-host perm, no cloud | This is the real **L2**. Per-platform (Zoom strong; Teams needs a bot; Meet preview API). |
| **Incumbents** | Granola diarizes **in the cloud**; botless-desktop tools (Otter/Fireflies) share the same acoustic ceiling; Recall.ai uses server bots for separated streams | Our fully-local pipeline is a **privacy edge**. We are not behind the botless-desktop category. |
| **Conference room** | Confirmed **unsolvable in software** (only mic arrays / per-person mics / enrollment) | Honest degradation is the correct and defensible stance. |
| **VLM (base + Extract, 450M)** | ⚠️ **INCONCLUSIVE — requires real images and a real test.** Active-speaker failed on a dense gallery frame and a *synthetic* speaker-view mock; slide extraction scored 4/5 on a *synthetic* slide; `screen_sharing` detected correctly | Do **not** commit VLM scope yet. Gate on a real-image test (real speaker-view + real shared-slide captures) before building `VisualContextCapture`. |

**Net:** the intelligence re-architecture (encoder + heads) is validated and
low-risk. Attribution shifts to **channel → Zoom SDK → acoustic**, with a11y as
roster-names only. The VLM's role (slides→RAG + roster) is a **hypothesis pending
real-image validation**, not a settled decision.

## Why: the output-shape audit

The decisive question (per `liquid-models-architecture`): *"the output of this
step is ___"* — the fill-in picks the head.

| Task | Output shape | Right tool | Today |
|---|---|---|---|
| Question detection | 1 label / input | Encoder seq-classifier (mean-pool) | heuristic scoring |
| Rhetorical / tag / self-answer | K yes/no flags | Encoder multi-label | 3 regex layers (F-201) |
| Trigger-type routing | 1-of-N | Encoder router | priority-sorted heuristics |
| Evidence / answer grounding | 1 label / token | Encoder token-classifier (BIO) | sentence heuristic |
| Noise / hallucination (fuzzy) | binary | Encoder seq-classifier | regex (keep O(1) cases) |
| Literal watch-words | O(1) match | deterministic Python | correct as-is |
| Context retrieval | vector rank | `LFM2.5-Embedding-350M` | all-MiniLM-L6 (non-Liquid) |
| Mode-aware prompt | free-form text | LFM2.5-1.2B-Instruct | correct as-is |
| Structured notes | typed fields | LFM2.5-350M-Extract | generic instruct + prompts |
| Active speaker (remote, per-endpoint) | 1 name | Zoom SDK callback (channel-grade) | not captured |
| Remote-speaker separation (no SDK) | label / segment | acoustic diarization (fallback) | primary mechanism (fragile) |
| Shared-slide / visual context | typed fields | `LFM2.5-VL-Extract` *(pending real test)* | not captured |

Six heuristic/decoder tasks collapse onto **one shared encoder backbone + tiny
heads** (~30–75 KB each, 0.02% of the model). Retrieval stays retrieval-shaped;
generation stays generative. Each model does the job its objective was built for.

## Target architecture (awareness → intelligence → action)

### Awareness layer (sensors — capture ground truth)

- **Audio, dual-channel:** mic (`you`) + system per-app (ScreenCaptureKit) →
  `LFM2.5-Audio-1.5B` ASR. Channel source = ground-truth me/not-me.
- **Attribution hierarchy** (strongest signal first; see below).
- **Visual context (hypothesis, pending real-image test):** `LFM2.5-VL-Extract`
  reads shared-screen/slides into RAG + extracts the roster. **Not** an
  active-speaker locator (Stage-0 signal is negative there).

### Intelligence layer (the encoder brain)

- **Shared backbone:** `LFM2.5-Encoder-350M`, one mean-pooled forward per turn
  (~14 ms measured); per-token vectors for span heads.
- **Heads (distilled, tiny):** trigger router (seq-cls) · quality/rhetorical gate
  (multi-label) · evidence spans (token-cls). Optional semantic-alert head.
- **Two-tier head path:** **v1 = frozen-encoder linear probe** (F-510) — a logistic
  head on frozen mean-pooled embeddings; no GPU, no forge, encoder-only; measured at
  0.886 macro-F1 off-the-shelf. Ships as the first non-heuristic `Head`, gated to beat
  the heuristic on a held-out split. **v2 = forge-LoRA** (F-503/504/505) — the measured
  upgrade once the forge bidirectional-encoder change lands. v1 decouples encoder
  intelligence from that forge work.
- **Route-first, hot/cold:** router/gate decide per turn. Hot path answers or
  suppresses from heads. Cold path runs retrieval + generation **only when a head
  fires** — no more running every trigger every turn.
- **Retrieval:** `LFM2.5-Embedding-350M` over `context/` docs + ingested slides
  + Notion. (Swap is safe; prove improvement via a hardened eval.)
- **Generation:** `LFM2.5-1.2B-Instruct` (mode-aware prompts), cold path only.
- **Extraction:** `LFM2.5-350M-Extract` for structured notes.

### Action layer (outputs)

- TranscriptPane (display) · PromptsPane (coaching) · Notes → consent-gated Notion.
- **Shared turn-state workspace** (not a daisy-chain): `ConversationBuffer` evolves
  into a typed state object carrying transcript window, attribution, encoder
  outputs, route decision, and generation results — debuggable and testable.

### Model map

| Role | Model | On disk |
|---|---|---|
| ASR | LFM2.5-Audio-1.5B (+ 600M-mini) | yes |
| Intelligence heads | LFM2.5-Encoder-350M | yes |
| Retrieval | LFM2.5-Embedding-350M | yes (downloaded 2026-07-20) |
| Generation | LFM2.5-1.2B-Instruct | yes |
| Structured notes | LFM2.5-350M-Extract | yes |
| Visual context *(pending real test)* | LFM2.5-VL-450M-Extract (+ VL-1.6B) | yes |
| Acoustic diarization (fallback) | ECAPA-TDNN (speechbrain, external) | yes |
| Active speaker (ground truth) | Zoom Meeting SDK callback | integration, not a model |

## The attribution hierarchy (the diarization answer)

Attribution quality is bounded by **microphone topology and platform access, not
by the diarizer.** Use the strongest available signal; degrade honestly.

| Layer | Signal | Fidelity | When it wins |
|---|---|---|---|
| L1 channel source | mic vs system | ground truth (me/others) | always available |
| L2 Zoom Meeting SDK | `onUserActiveAudioChange` + per-participant PCM, local | ground truth per remote speaker | on Zoom, host/co-host — the real ground-truth path |
| L2′ VLM visual *(pending real test)* | `LFM2.5-VL-Extract` → shared-slide content + roster names | strong for content; **weak for active-speaker** | screen-share / slides → RAG; roster on clean frames |
| L3 acoustic | speaker-change seg → embed → cluster (roster-bounded) | estimate | no SDK (Teams/Meet/phone/recording) |
| L4 a11y roster + enrollment | AX participant names; voice profiles for known colleagues | names the clusters | native app roster; recurring colleagues |

**a11y is L4, not L2** — it exposes names and own-mute state, not a queryable
"who is speaking." **Active-speaker belongs to L2 (Zoom SDK) or L3 (acoustic), not
vision or a11y.**

**Regime detection** sets expectations: solo-endpoint → high-confidence names;
**conference room** (many people, one far-field mic) → L2/L3 collapse to one room
identity, so degrade honestly to a single `Others (room)` bucket or clearly-flagged
best-effort `Speaker N` — never emit confidently-wrong names. Voice enrollment and
LFM lexical cues ("what do you think, Raj?") are the high-leverage adds here.

## Staging (the re-work)

Each stage: reversible, gated on a number or a decision.

| Stage | Scope | Gate / exit criteria | Features |
|---|---|---|---|
| **0 — Decision spikes** ✅ done | encoder smoke; retrieval eval; a11y + incumbent research; VLM probe | See findings above | F-500 |
| **1 — Structural refactor (no training)** | `EncoderIntelligenceLayer` + typed turn-state; wire encoder backbone; keep heuristics as first head impls behind it; **v1 frozen-encoder linear-probe head** (no GPU/forge); `AttributionResolver` hierarchy + regime scaffold; land retrieval swap; **delete-as-you-replace** | Full suite green; retrieval ≥ baseline; probe head beats heuristic on held-out else stays off; zero behavior regression | F-501, F-502, F-510, F-601 |
| **1b — Harden retrieval eval** | chunk-level relevance, more/confusable docs, query/passage prompts — give the eval discriminating power | Eval separates two retrievers by a meaningful margin | F-509 (new) |
| **2 — Awareness** | acoustic diarization fix (speaker-change seg, roster-bound); voice enrollment; conference-room degradation; a11y roster-name reader | Attribution eval on real calls; honest degradation verified | F-604, F-605, F-606, F-602 |
| **2b — VLM visual context** *(gated on real-image test)* | Only after a real speaker-view + real shared-slide test: `VisualContextCapture` (VL-Extract slides→RAG + roster; screen-share detected). Debounced. **Not** active-speaker | Real-image test passes; slides retrievable in RAG | F-603 |
| **2c — Zoom high-fidelity mode** *(follow-on)* | Zoom Meeting SDK integration: ground-truth active speaker + per-participant audio, local | Ground-truth attribution on Zoom, no cloud | F-608 (new) |
| **3 — Learned heads (distillation)** | Distill heuristics → encoder heads on real meeting data (Granola) + 1.2B teacher; LoRA (LFM2.5 module names, mean-pool, span head on its own LoRA). Ship each head only when it beats the heuristic. Order: router → spans → gate | Per-head eval beats heuristic baseline | F-503, F-504, F-505 |
| **4 — Runtime & polish** | Route-first hot/cold + shared-state; persistent warm-model runtime (replace subprocess-per-call); model-contention mgmt; notes → Extract; lexical speaker-consistency pass | Latency budget met; no cold-spawn per call | F-506, F-507, F-508, F-607 |

## Non-negotiables (from the Liquid architecture skill)

- **Mean-pool, never last-token** — final encoder layer is conv (3-token receptive field). Confirmed in the smoke spike.
- **Do not co-train the span (token) head with sequence heads** — PII collapsed to 9.9% F1; span head gets its own LoRA.
- **LFM2.5 module names** in LoRA (`w1/w2/w3`, `out_proj`, `in_proj`) — LLaMA names train nothing silently. Expect ~92 modules on 350M.
- **O(1)-checkable belongs in code, not a model** — literal watch-words, repeated-char filters, channel routing stay deterministic.
- **VLM is prompt-sensitive** — Extract needs a YAML field schema in the *system* prompt + image-only user turn + enum choices. Free-form prompts fail.
- **Model contention** — 4+ resident models on a laptop during a live call; manage which are warm when (VL-450M light, 1.6B only for slides; encoder is cheap at 14 ms).
- **Local-first** — no cloud diarization/ASR; the only egress remains consent-gated Notion export. (Zoom SDK is local; it does not break this.)
- **Honest degradation** — the conference-room answer is "flag low confidence," not "pretend."

## What we explicitly keep (not rebuilt)

Dual-buffer fork · turn-based accumulation · per-source silence · thread safety ·
WebSocket protocol · consent-gated export · Tauri dual-pane UI · the RAG eval harness.
These are sound and model-agnostic; the re-work happens behind their interfaces.

## Data

The Granola transcript(s) become: Stage 1 real-world test data, and Stage 3
distillation seed labels for the encoder heads (real questions, topic shifts,
attribution examples). Store gitignored (`data/fixtures/` or `tests/eval/real/`) —
real participant data, never committed. **Also needed:** real Zoom speaker-view and
shared-slide screenshots to run the VLM real-image test that gates Stage 2b.
