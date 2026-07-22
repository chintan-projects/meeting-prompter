# Architecture

Why this system is shaped the way it is. Design decisions, the models behind each one,
what has been measured, and what is still open.

- Build and run it → [docs/DEVELOPER-GUIDE.md](docs/DEVELOPER-GUIDE.md)
- Corpus distillation in depth → [docs/distillation.md](docs/distillation.md)
- Live decision register → [docs/architecture/open-decisions-log.md](docs/architecture/open-decisions-log.md)
- The re-architecture plan → [docs/architecture/liquid-rearchitecture.md](docs/architecture/liquid-rearchitecture.md)

---

## 1. The one-line thesis

**Use the right model for each output shape, and keep every one of them on the device.**

Both halves are load-bearing. The first is why there are six models instead of one
general-purpose LLM. The second is why the hard problems — attribution, corpus
preparation, latency — are solved with architecture rather than by calling a bigger
model in a datacenter.

---

## 2. Why the Liquid architecture

### 2.1 The output-shape audit

The system was originally built as most such systems are: regex heuristics for
detection, a generic decoder for everything else. That is one hammer applied to tasks
with very different shapes. The decisive question — *"the output of this step is ___"* —
picks the right tool almost mechanically:

| Task | Output shape | Right tool | Originally |
|---|---|---|---|
| Question detection | 1 label per input | encoder seq-classifier (mean-pooled) | heuristic scoring |
| Rhetorical / tag / self-answer | K yes-no flags | encoder multi-label | 3 regex layers |
| Trigger-type routing | 1-of-N | encoder router | priority-sorted heuristics |
| Evidence / answer grounding | 1 label per token | encoder token-classifier (BIO) | sentence heuristic |
| Noise / hallucination (fuzzy) | binary | encoder seq-classifier | regex |
| Literal watch words | O(1) match | **plain Python** | correct already |
| Context retrieval | vector rank | `LFM2.5-Embedding-350M` | all-MiniLM-L6 (non-Liquid) |
| Mode-aware prompt | free-form text | `LFM2.5-1.2B-Instruct` / 2.6B | correct already |
| Structured notes | typed fields | `LFM2.5-350M-Extract` | generic instruct + prompts |
| Active speaker (remote) | 1 name | Zoom SDK callback | not captured |
| Remote-speaker separation | label per segment | acoustic diarization (**fallback**) | primary mechanism |

Six heuristic-or-decoder tasks collapse onto **one shared encoder backbone plus tiny
heads** — roughly 30–75 KB each, about 0.02% of the model. Retrieval stays
retrieval-shaped. Generation stays generative. Each model does the job its training
objective was built for.

Note row six. **Literal watch words stay plain Python.** The audit is not an argument
that everything should be a model; it is an argument that each task should use the
cheapest tool that fits its shape. Anything checkable in O(1) belongs in code.

### 2.2 Encoder and embedding are not interchangeable

Measured on unrelated text: cosine similarity **0.855** for the encoder versus **0.095**
for the embedding model. The encoder's space is not metric in the way retrieval needs —
it packs syntax and task structure, which is exactly what makes it a good classifier
backbone and a bad retriever. Encoder → heads. Embedding → retrieval. Never swapped.

### 2.3 Small and local, as an architectural stance

Every model here is between 350M and 2.6B parameters, and all of them run on the laptop.
This buys four things a cloud call cannot:

- **Latency budgets that permit per-turn work.** The encoder is ~14 ms mean-pooled per
  turn. That is cheap enough to run on *every* turn, which in turn is what makes
  route-first hot/cold dispatch possible at all.
- **Privacy that is structural rather than promised.** Meeting audio and the user's
  private corpus never leave the machine. The only network egress in the system is a
  consent-gated Notion export the user triggers by hand.
- **A competitive edge that is real.** Granola diarizes in the cloud; Recall.ai uses
  server-side bots for separated streams; botless desktop tools share our acoustic
  ceiling. A fully local pipeline is not a compromise against that field.
- **Cost that does not scale with use.** Re-distilling a corpus is free, so it can be
  re-run whenever documents change.

The trade is honest: small models are worse at open-ended generation. The architecture
responds by **not asking them to do open-ended generation on the critical path** — see
retrieval-first below.

### 2.4 Non-negotiable rules of the stack

Learned the hard way; violating any of them fails silently rather than loudly.

- **Mean-pool, never last-token.** The final encoder layer is a 3-token convolution, so
  the last token sees almost nothing.
- **Never co-train the span head with sequence heads.** Doing so collapsed PII span F1
  to 9.9%. The span head gets its own LoRA.
- **Use LFM2.5 module names in LoRA** (`w1/w2/w3`, `out_proj`, `in_proj`). LLaMA names
  match nothing and train nothing, without an error.
- **The VLM is prompt-sensitive.** Extract needs a YAML field schema in the *system*
  prompt, an image-only user turn, and enum choices. Free-form prompts fail.
- **Manage model contention.** Four-plus resident models on a laptop during a live call
  requires deciding which are warm when. `lib/warm_runtime.py` is the single owner.

---

## 3. The model map

| Role | Model | Size | Latency | Status |
|---|---|---|---|---|
| ASR | LFM2.5-Audio-1.5B (+ 600M-mini fallback) | 1.2 GB | ~300 ms / chunk | shipped |
| Retrieval | LFM2.5-Embedding-350M (1024-dim) | ~350 MB | ~50 ms | shipped |
| Intelligence backbone | LFM2.5-Encoder-350M (frozen, bidirectional) | ~350 MB | ~14 ms / turn | shipped |
| Trigger router head | LFM2.5-TriggerRouter-350M (forge LoRA) | ~75 KB adapter | negligible | shipped, **unproven live** |
| Generation | LFM2.5-2.6B-Q4_K_M (1.2B-Instruct fallback) | ~1.6 GB | ~1.5–3.5 s | on-demand only |
| Structured notes | LFM2.5-350M-Extract | ~350 MB | — | in progress (F-507) |
| Corpus distillation | LFM2.5-2.6B, prompted | ~1.6 GB | offline | shipped default, specialist pending (F-702) |
| Acoustic diarization | ECAPA-TDNN (speechbrain) | ~100 MB | — | optional, off by default |
| Visual context | LFM2.5-VL-450M-Extract | ~450 MB | — | not built, gated (F-603) |

Roughly 4.5 GB resident during a call. Every model is selected in `config.yaml`; there
are no hardcoded model paths in `lib/`.

**Why the 2.6B for generation.** On identical retrieved context, the 1.2B answered
correctly and then appended *"I don't have that information in my documents"* anyway —
refusal boilerplate leaking into good answers — and it proved brittle to prompt
rewrites. The 2.6B gave a graceful, honest partial answer: it acknowledged what the
context covered, flagged what it did not, and still extracted the useful points. That is
5× the latency and the right trade **once generation is user-gated** rather than on the
critical path.

---

## 4. The live pipeline

```
┌──────────────────────────── AWARENESS ────────────────────────────┐
│  Microphone ────────►  AudioCapture  ──► LFM2.5-Audio ──► filters │
│  System audio ──────►  (per-app via ScreenCaptureKit, or          │
│                         BlackHole loopback)                       │
└───────────────────────────────┬───────────────────────────────────┘
                                │  source = ground-truth me/not-me
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        TranscriptBuffer                ConversationBuffer
        (turn accumulation,             (rolling 90 s window,
         2 s pause boundary)             trigger routing)
                 │                             │
                 ▼                    ┌─────── INTELLIGENCE ───────┐
        TranscriptStore               │  LFM2.5-Encoder-350M       │
        (upsert + edit overlay)       │  one mean-pooled forward   │
                 │                    │  ├─ trigger router head    │
                 ▼                    │  ├─ quality / rhetoric gate│
        WebSocket /ws/transcript      │  └─ evidence-span head     │
                 │                    └──────────────┬─────────────┘
                 ▼                        route-first │ hot / cold
          TranscriptPane                              ▼
                                      Hybrid RAG  (FTS5 5% + vector 95%)
                                                      │
                                        ┌─────────────┴────────────┐
                                        ▼                          ▼
                             borrowable unit (default)    generation (on demand)
                             ~16–190 ms, no LLM           LFM2.5-2.6B, user-gated
                                        └─────────────┬────────────┘
                                                      ▼
                                        WebSocket /ws/prompts → PromptsPane
```

### 4.1 Dual-stream capture and attribution

Two independent audio threads feed one thread-safe buffer. Source is ground truth:
`mic → "You"`, `system → "Others"`. That is Tier 1 and it is deterministic.

Attribution beyond that follows a hierarchy of decreasing fidelity — always use the
strongest signal available, and degrade **honestly**:

| Layer | Signal | Fidelity | Wins when |
|---|---|---|---|
| L1 channel source | mic vs system | ground truth (me / others) | always |
| L2 Zoom Meeting SDK | `onUserActiveAudioChange` + per-participant PCM, local | ground truth per speaker | Zoom, host/co-host (F-608, not built) |
| L3 acoustic | speaker-change segmentation → embed → cluster | estimate | no SDK — Teams, Meet, phone |
| L4 a11y roster + voice enrollment | AX participant names; profiles for known colleagues | names the clusters | native rosters, recurring colleagues |

Accessibility APIs are **L4, not L2**: research found no documented "is speaking"
attribute on Zoom, Teams, or Meet. They give names and your own mute state. Vision was
investigated for active-speaker detection and failed; its plausible role is slides →
RAG and roster extraction, and even that is gated on a real-image test.

**Conference rooms are unsolvable in software.** Many people, one far-field microphone —
L2 and L3 both collapse to a single room identity. The correct response is to say so:
degrade to one `Others (room)` bucket, or clearly flagged best-effort `Speaker N` with a
`low_confidence` marker the UI renders as a "~ best guess" badge. Never emit a
confidently wrong name into a record the user will trust and export.

**Echo suppression.** Without headphones, both streams transcribe the same speech.
`StreamDeduplicator` catches near-duplicates via `SequenceMatcher` in an 8 s window.
Suppressed chunks still signal silence, so turn boundaries stay correct.

### 4.2 Turn-based transcript

Raw ASR chunks are ~4 seconds — displaying them directly produces a fragmented,
unreadable transcript. Turns are accumulated on the **backend**, where the pipeline has
the timing information, then streamed as `transcript_update` (partial) /
`transcript_final` (complete) with upsert-by-ID semantics that preserve user edits.

A frontend fix by time-gap grouping was tried and was insufficient: the groupings were
not semantically meaningful, and groups shifted as new chunks arrived.

**Two independent buffers consume the same chunk stream.** `TranscriptBuffer` optimizes
for display (simple pause boundaries); `ConversationBuffer` optimizes for intelligence
(rolling window, trigger evaluation). They share no state, and that separation is what
keeps each one simple.

### 4.3 Hybrid retrieval

| Signal | Method | Weight | Catches |
|---|---|---|---|
| Lexical | FTS5 BM25 | 5% | exact terms, proper nouns, acronyms |
| Semantic | LFM2.5-Embedding-350M cosine | 95% | conceptual similarity across vocabulary |

Section-aware chunking (split on markdown headers, 400 tokens, 50 overlap, header
prepended), fused by weighted sum, then heuristically re-ranked. Semantic scores use
**raw cosine, not min-max normalization** — min-max destroys the confidence
discrimination the silence threshold depends on.

Benchmark (21 queries, `tests/eval/`): **Hit@1 94.4%, MRR 0.972.**

Two components are measurably idle on typical queries: BM25 scores 0.000 on every hit
at 5% weight, and the heuristic re-ranker frequently leaves order unchanged. Both are
known and tracked rather than quietly assumed to be working.

### 4.4 Retrieval-first: the central decision (D-08)

**The default live path contains no LLM.** A trigger fires, retrieval runs, and the card
shows a borrowable span of the corpus verbatim — 16–190 ms warm — with its heading as
provenance and expand-to-source for the full unit. Generation is demoted to an explicit
user action (the ✨ button, `POST /prompts/generate`).

The reasoning, from the lab (E-02): the 1.2B was fluent but unverifiable, the 2.6B was
slow and truncated, Extract is a field extractor rather than an answerer — while
retrieval ranked the right documents 94% of the time. Mid-meeting, a grounded sentence
you can read aloud in 50 ms beats a plausible paragraph in 3 seconds that you have to
audit before you dare say it.

The cost is stated plainly in [docs/distillation.md](docs/distillation.md): corpus
quality becomes a hard ceiling on output quality, because there is no model in the path
to compensate for a weak source. That is what the entire corpus-preparation flow exists
to address.

### 4.5 Four intelligence modes

| Mode | Label | Voice | Max tokens | Persistence |
|---|---|---|---|---|
| ALERT | HEADS UP | direct — what you need to know now | 100 | persistent |
| QUESTION | ANSWER | concise answer + optional coaching suffix | 200 | persistent |
| TOPIC_MATCH | FYI | a **new** fact from the docs, not an echo | 100 | ephemeral (45 s) |
| FOLLOW_UP | SUGGEST | coaching nudge — "Ask about…", "Mention that…" | 75 | standard (90 s) |

Two suppression layers keep the panel quiet. **Rhetorical suppression** (F-201) filters
tag questions, self-answering questions, and rhetorical forms before scoring.
**Dead-end suppression** (F-202) drops empty, low-confidence, or too-short results at
both the generator and session layers, so the user never sees "I don't have that
information."

A prompt-spammy session is a failed session even when every individual card is correct.
Silence is a feature with a maintenance cost.

### 4.6 Quiet by default: the listen gate (D-02)

The suppression layers above filter *bad* cards. They do nothing about the deeper
problem, which the first live call surfaced immediately: **a correctly-detected
question is not the same as a question you want answered on screen**, and in a real
meeting most of them aren't. A perfect trigger router still interrupts on every true
positive. Prompt spam is a **permission problem, not a classification problem.**

So the default is quiet, and the user opens the tap two ways:

| Channel | Control | Behaviour |
|---|---|---|
| **Temporal** | ⌘L / `POST /prompts/listen` | Arms the listen window; automatic cards flow until toggled off. |
| **Spatial** | select transcript text → 💡 Answer this | Answers that exact span on demand, gate-exempt, even while quiet. |
| **Always-on** | `triggers.gating.always_on` | Watch-word ALERTs only — the user pre-authorized them by naming the terms. |

Three implementation properties matter:

- **One choke point.** The gate sits in `MeetingOrchestrator._process_trigger`, which
  both capture pipelines already call. Gating there covers every automatic path by
  construction rather than by remembering to check in two places.
- **Short-circuit, not filter.** A suppressed trigger returns before retrieval runs.
  Triggers fire on most turns, so filtering after the work would pay the full cost of
  a feature whose entire point is doing less.
- **Explicit requests bypass it entirely.** `retrieve_for_text` (select-to-answer) and
  `generate_for_text` (the ✨ button) never consult the gate. The user asking *is* the
  permission, and routing consent through a gate the user just opened by hand would be
  asking twice.

The window has **no timer** (`max_listen_seconds: 0`) — it stays open until toggled
off. That is a deliberate product call, and its known failure mode is a forgotten
window quietly restoring the old always-on behaviour. Two mitigations: the status bar
carries an unmissable green **◉ LISTENING** state, and the safety cap exists as an
opt-in for anyone who wants it. `triggers.gating.enabled: false` restores always-on
push wholesale.

---

## 5. Corpus preparation (D-09, ADR-001)

Because retrieval-first makes the corpus the ceiling, corpus preparation is a **product
step**, not a script: bring your documents → distill into grounded, provenance-tagged
answer-units → readiness score with a gap list → activate for live calls.

Full methodology, measurements, and open limits: **[docs/distillation.md](docs/distillation.md)**.

The short version: source documents are *explainers*, meetings need an *answer bank*,
and the distiller reshapes one into the other offline. Measured lift on a 21-question
held-out set is **76% → 90–95%** using the cloud path. ADR-001 forbids shipping that
path — the user's private corpus must not leave the device — so the shipped default is
an on-device model, and forging a local specialist to close the gap is the critical
path (F-702).

---

## 6. Fallback chains

Every stage degrades to something usable rather than erroring. Named, tested, and
logged with context — never silent.

```
Retrieval        low confidence  →  silence (better than a wrong card)
Generation       failure         →  extraction bullets  →  silence
Distiller        contract reject →  heuristic floor (grounded, verbatim)
ASR              2.5 unavailable →  LFM2 legacy
Encoder heads    model absent    →  heuristic heads
Trigger router   adapter absent  →  heuristics
Embedding        Liquid absent   →  all-MiniLM-L6-v2 (384-dim)
Attribution      no SDK          →  acoustic  →  flagged "Others (room)"
Per-app capture  no permission   →  mic-only, surfaced in the UI
```

---

## 7. Concurrency

The pipeline is genuinely multi-threaded — two capture threads, an asyncio event loop,
and background model warm-up — so thread safety is explicit rather than assumed.

| Mechanism | Where | Guards |
|---|---|---|
| `threading.Lock` | TranscriptBuffer, ConversationBuffer, RAGAnswerGenerator | all public mutations |
| Double-checked locking | `lib/rag/embedder.py`, `lib/intelligence/encoder.py` | lazy model load |
| `loop.call_soon_threadsafe` | `src/api/session.py` | capture thread → asyncio queue |
| `deque(maxlen=1000)` | trigger history | unbounded growth in long sessions |
| try/finally | both capture threads | every setup has a teardown |

The double-checked locking is worth spelling out, because it was a real bug (BUG-006).
Lazy model loading was a plain check-then-set. It was latent for months and went live
the moment session start began pre-warming the embedder on a background thread while
the pipeline queried on another: torch materializes weights from the meta device during
construction, and a concurrent second construction dies with *"Cannot copy out of meta
tensor; no data!"* — on the first query of every session.

**800 tests were green while this was broken.** A test suite proves the paths it
exercises; concurrency bugs live in the paths it does not.

The fix publishes to `self._model` only after construction completes, so the unlocked
fast path can never observe a half-built model, and steady-state embedding stays
lock-free.

---

## 8. Interface contracts

### WebSocket

`/ws/transcript` — `transcript_update` (partial turn), `transcript_final` (complete),
`transcript_relabeled` (attribution changed); client sends `edit`.

`/ws/prompts` — one `prompt` message per card, carrying `trigger_type`, `answer`,
`confidence`, `method`, `latency_ms`, `source`, `heading`, `source_text`, `persistence`,
`dismiss_ms`, and display metadata. In the default retrieval-first path `method` is
`"retrieval"`, `answer` is the glanceable sentence, and `source_text` is the full unit
for expand-to-source.

The authoritative shapes live in [CLAUDE.md](CLAUDE.md#websocket-protocol) — both sides
read that section, and changing a message without updating it is how the frontend and
backend drift apart silently.

### Configuration

Every threshold, weight, timeout, and model choice is in `config.yaml`, loaded through
typed dataclasses in `lib/config.py`. No magic numbers in code, no hardcoded model
paths, `MODELS_DIR` for the registry. This is what makes tuning a config diff rather
than a code change, and what lets the same code run with a different model set.

---

## 9. Open decisions

Tracked live in
[docs/architecture/open-decisions-log.md](docs/architecture/open-decisions-log.md).
The consequential ones:

| ID | Question | Status |
|---|---|---|
| **D-01** | AEC mic capture (macOS Voice-Processing I/O) — cancel speaker→mic echo *at capture* so attribution is correct by construction, independent of headphones | open, high priority, foundational |
| **D-02** | User-gated interaction — default quiet, user opens the tap via armed listen-window or select-to-answer; ALERTs stay the only always-on channel | **decided and built** — see §4.6 |
| **D-03** | Answer model — 2.6B is wired with 1.2B fallback | leaning 2.6B, operator to confirm |
| **D-07** | The transcript refiner currently shares the answer-model instance; with a 2.6B reasoning model, per-turn refinement is slow | open |
| **D-11** | Readiness rater is miscalibrated (57% judge agreement, harsh in 36% of cells) | decided in principle, rater blocked on recalibration |

D-02 is the one that matters most to the product's character. The always-on push model
produced a stream of irrelevant prompts, which is the failure mode of this entire
product category. Whether the answer is temporal (an armed window) or spatial
(select-to-answer) is what the first live calls are meant to settle.

---

## 10. Future scope

Ordered by what unblocks what, not by appeal.

**Near — proving what is already built.** The learned trigger router (F-503) is on and
unvalidated in a real call; it beat the heuristic offline by a wide margin (hybrid
macro-F1 0.846 vs 0.55 probe vs 0.26 heuristic) but offline transcripts are not a
meeting. Structured notes via Extract (F-507) and the persistent warm-model runtime
(F-508) are path-built and awaiting the same validation. AEC at capture (D-01) is
foundational: it improves attribution, ASR quality, and prompt trust simultaneously.

**Next — the corpus loop.** Forge the local distiller specialist (F-702), recalibrate
the readiness rater (F-707), and validate the whole flow on a *messy* corpus (F-708).
F-708 gates F-702: the case for a specialist model rests on corpora that are not already
well-structured explainers, and that case is currently an assumption rather than a
measurement.

**Then — attribution fidelity.** Zoom Meeting SDK integration (F-608) is the real
ground-truth path: `onUserActiveAudioChange` plus per-participant PCM, locally, with
host or co-host permission and no cloud. It would make named remote speakers exact on
Zoom, while Teams and Meet continue to fall back to acoustic clustering.

**Exploratory — visual context.** `LFM2.5-VL-Extract` reading shared slides into RAG and
extracting the roster (F-603). Explicitly gated on a real-image test with genuine
speaker-view and shared-slide captures. The Stage-0 probe was inconclusive on synthetic
images, and vision is **not** the answer to active-speaker detection.

**Continuous — the encoder heads.** The evidence-span head (F-504) and quality gate
(F-505) complete the migration of heuristics onto the shared backbone. The discipline
that keeps this from becoming a mess is *delete-as-you-replace*: when a head ships, the
heuristic it supersedes is deleted, not left running beside it. Two parallel
implementations of the same decision is the real refactor trap.

---

## 11. How we got here

The system has been rebuilt more than once. The pivots that stuck, and what each taught:

**Keyword → ColBERT → hybrid FTS5 + vector.** Jaccard similarity missed every semantic
query ("neural network alternatives" found nothing). ColBERT late-interaction fixed
that, but cost 1.5 GB resident and a PLAID index to maintain. Hybrid FTS5 + a 350M
embedding model matches its quality on the benchmark at a fraction of the footprint,
with SQLite doing the storage. *Late interaction was the right idea at the wrong price.*

**Generation → extraction → hybrid → retrieval-first.** Small models hallucinated when
asked to answer from context, so generation was removed entirely in favor of sentence
extraction — accurate, but choppy and robotic. Hybrid RAG restored fluency by using
extraction as a *grounding* stage that pre-filters what the model can see. Then the lab
showed retrieval alone was good enough for the live path, and generation moved
off it entirely. *Four positions in one direction: less model on the critical path.*

**Chunk-count → timestamp buffering.** Counting audio chunks to detect pauses caused the
"second question triggers the first" bug, because a 4-second chunk's silence can fall at
its start, middle, or end. Real timestamps were the only reliable boundary.

**Heuristics → encoder heads.** The current re-architecture, done as a strangler-fig
inside the existing repo rather than a green-field rewrite. The skeleton — dual-buffer
fork, turn accumulation, per-source silence, thread safety, WebSocket contract, Tauri
UI — is sound and model-agnostic, so organs are swapped one at a time behind existing
seams, each gated by the test suite, always in a working state. A new project would only
be right if the skeleton were wrong.

Some things did not survive contact and are worth recording as dead ends:

- **Accessibility APIs for active-speaker detection** — no such attribute exists on any
  major platform.
- **Vision for active-speaker detection** — failed on gallery view and on speaker view.
- **Prompt engineering as a correctness control for the local distiller** — hardening
  the prompt produced *different* narration rather than none. Structural contract checks
  are what work.
- **Conference-room speaker separation** — genuinely unsolvable with one far-field
  microphone. Honest degradation is the answer, not a better clusterer.
