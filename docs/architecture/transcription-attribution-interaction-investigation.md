# Investigation: Transcription Quality, Speaker Attribution, and Interaction Model

**Status:** OPEN — recorded for decision, no code committed from this investigation yet.
**Date:** 2026-07-21
**Trigger:** Live-call testing (session-018) surfaced three issues that turn out to
share roots and point at one foundational decision.

---

## 0. What prompted this

Running a real Zoom call through the app exposed:

1. **Mediocre ASR quality** — the local LFM2.5-Audio transcript is rough.
2. **Wrong speaker attribution** — "You" appears when the local user has barely
   spoken; the *same* remote talker flips between "You" and "Others".
3. Two product-direction questions from the user:
   - Should transcript chunks go through a **cleanup** pass?
   - Should intelligence be **proactive (push everything through RAG)** or
     **user-driven (select / pin a segment, then ask)**?

The core realization: **1 and 2 share one physical root cause, and fixing it is
also the highest-leverage move for 3.** So this doc reasons from the cause
outward rather than treating them as three separate features.

---

## 1. Attribution — the channel assumption is the bug

### The current model
Attribution is **channel-based** (L1 in the attribution hierarchy):
`source="mic" → "You"`, `source="system"/Zoom-tap → "Others"`. Correct *only if
each channel carries exactly one party.*

### Why it breaks
There is one physical coupling path we cannot design away:
**speakers → microphone (open-air echo).** With no headphones, the laptop mic
re-records the remote participants coming out of the speakers. That audio enters
on the mic channel → gets tagged **"You"**. This single fact explains both
symptoms:

- *"Barely spoke but shows You"* → those turns are remote voices leaking into the mic.
- *"Same talker flips You/Others"* → that person is captured twice — once by the
  Zoom tap (correct "Others") and once by the mic via the speakers (wrong "You").
  Whichever pipeline finalizes the ~4s chunk first wins the label.

It also poisons ASR: the mic channel becomes **overlapping voices**, which is
near-worst-case input for any transcriber. So a large share of "mediocre quality"
is contamination, not the model.

### Why the current mitigation is a band-aid
`StreamDeduplicator` (`lib/stream_dedup.py`) tries to *detect and delete* the
duplicate after the fact via `difflib` text similarity (threshold 0.55). Two
failure modes, both inherent:
- When the two ASR passes diverge (they will — different audio quality), similarity
  drops below threshold → duplicate survives → speaker on both sides.
- When it *does* fire, it keeps whichever arrived first — there is **no rule that the
  Zoom tap wins for remote speech** — so it can delete the correct copy and keep the
  wrong "You" one.

Dedup treats the symptom (two transcripts) with a fuzzy heuristic. The user's
constraint is decisive here: **we cannot control whether people use headphones,
speakers, or BlackHole.** A heuristic that only works in some listening
configurations is not a solution.

### First-principles fix: cancel the echo at capture (AEC)
Stop the mic from *hearing* the remote audio in the first place. We already have
the one input needed to do this correctly — the **reference signal**: we know
exactly what is being played to the speakers (it is the system audio we already
tap). Subtracting that reference's acoustic contribution from the mic leaves
**only the local voice**. This is **Acoustic Echo Cancellation** — standard in
Zoom/Teams/Meet and every telephone.

On macOS it is a built-in OS capability: capture the mic through
**Voice-Processing I/O** (`AVAudioInputNode.setVoiceProcessingEnabled(true)`),
which applies AEC + noise suppression + auto-gain using the system output as the
echo reference, before the app ever sees the samples.

**Why this is the right, non-hacky answer:**
- Removes the **cause** (echo in the mic), not the symptom (duplicate text).
- Makes channel attribution correct **by construction** — no thresholds, no guessing.
- **Invariant to how people listen** — the exact requirement:
  - Headphones → no echo; AEC is a no-op.
  - Speakers → AEC removes the leak.
  - BlackHole/other routing → mic still passes through AEC, still clean.
- Deterministic and reference-based, unlike fuzzy text matching.
- **Also fixes quality:** a clean single-voice mic (plus NS/AGC) sharply improves
  the "You" transcript independent of the model.
- **Demotes dedup** from load-bearing to an optional safety net (or delete it).

**Ceiling (not required):** per-**name** truth ("Alice said X") comes from a
meeting SDK's per-participant stream (Zoom Meeting SDK `onUserActiveAudioChange`,
L2 in the ADR). AEC is the universal default that needs no SDK and works for any
meeting app; SDK per-participant is the optional upgrade for named diarization.

**Implementation reality:** the mic is captured today via Python `sounddevice`,
which cannot do AEC. The change is to move mic capture to a small macOS
Voice-Processing path (Swift/CoreAudio — same pattern as the `audio-tap` helper)
feeding the identical PCM pipeline. Contained and well-trodden.

**Recommendation:** adopt AEC mic capture as the foundational fix. It is the
single highest-leverage change — it improves attribution, quality, *and* the
trustworthiness of every downstream prompt at once.

---

## 2. Idea #1 — Transcript cleanup pass

### What already exists
`lib/text_refiner.py` already runs finalized turns through LFM2.5-1.2B-Instruct
with a conservative cleanup prompt (fix grammar/punctuation, remove stutters,
correct obvious mishearings, preserve meaning). So "run chunks through cleanup"
is **partly built** — the question is how far to lean on it.

### First-principles framing
Cleanup does two *different* jobs — separate them:
- **(a) Readability** — punctuation, capitalization, de-stutter, merge fragments.
  Safe for an LLM; this is what the refiner does well.
- **(b) Error correction** — recover the *right words* the ASR got wrong. This
  requires information the LLM does not have (the acoustics). An LLM asked to
  "correct" a transcript will confidently **rewrite meaning** — it hallucinates
  plausible words, which is worse than a visible ASR error in a meeting record.

**Principle: cleanup can polish, it cannot reconstruct information destroyed at
capture.** Words the ASR never heard correctly cannot be recovered by a text
model without risking fabrication.

### Options
- **Strengthen the LLM refiner** — cheap, but pushing it toward (b) invites
  hallucinated corrections in a record people trust. Keep it scoped to (a).
- **Fix capture upstream** (AEC clean audio + wider context window + overlapping
  chunks so words aren't cut at 4s boundaries) — fixes the cause of most errors.
- **Context-biased ASR** — prime the transcriber with meeting vocabulary (watch
  words, participant names, agenda terms) so domain words transcribe correctly.
  Correction *with* grounding, not guessing.

**Recommendation:** (1) fix capture first (AEC + chunking), (2) keep the refiner
**readability-only** — explicitly *not* a meaning-correction stage, (3) optionally
add domain-term biasing. Do **not** escalate the LLM refiner into an error-fixer.

---

## 3. Idea #2 — Interaction model: proactive-RAG vs user-driven

This is the larger architectural fork.

### Observed failure (session-018, live)
The push model produced a **continuous stream of mostly-irrelevant prompts** — a
distraction with little value. Two compounding causes: (a) bad ASR/attribution
fed wrong prompts, and (b) **structurally, "prompt on every turn" is low-precision
even with perfect audio** — most turns don't warrant a prompt, so always-on push
is noisy by construction, not just because of input errors.

### Current model = PUSH
Every turn is auto-scored by triggers → RAG → generated prompt.
- **Pros:** zero effort, real-time, surfaces things you'd miss.
- **Cons:** interrupts with prompts you didn't ask for; every turn spends
  RAG+LLM compute; and — critically — **ASR/attribution errors propagate into
  wrong prompts.** One confidently-wrong prompt erodes trust in all of them.

### Reframe: who controls the attention budget?
The deepest framing isn't push-vs-pull — it's **who decides to spend the user's
attention.** Today the *system* spends it on every turn. The fix is to hand the
throttle to the *user*, who can open the tap in two dimensions:

- **Select-to-answer (spatial throttle):** user highlights a transcript span →
  RAG/generate runs on *that* selection only. Precise, zero unsolicited output.
  For "I want the answer to *this specific thing*."
- **Armed listen-window (temporal throttle):** a toggle — "from now on, answer
  what you hear" — the user flips ON when they can *predict* a Q&A stretch (about
  to present / be questioned) and OFF when it passes. Key insight: **the user
  often knows in advance when they need help**, so let them declare it rather than
  the system guessing on every turn forever.

Default is **quiet**. The user opens the tap on a phrase (select) or for a stretch
(arm). The one justified always-on exception is hard **ALERTs** (watch words) —
rare, high-value, and the case the user *cannot* anticipate; everything generative
(answers/suggestions/topics) becomes user-gated.

### Proposed model = PULL / hybrid
- **(2a) Select-then-ask:** user highlights a transcript segment and explicitly
  requests an answer/suggestion. High precision, on-demand — but costs attention
  mid-conversation.
- **(2b) Pin-for-later:** user pins segments to revisit in a lull or post-call.
  Async, low-interruption.

### First-principles framing: two axes, not one switch
- **Timing:** real-time (during) vs deferred (lull / after).
- **Initiative:** system-initiated (push) vs user-initiated (pull).

The insight: **not all intelligence has the same urgency, and the reliability of
the input differs by type.**
- **ALERT** (watch word "pricing") → the entire point is real-time push. Keep push.
- **QUESTION → answer** → valuable in real time *only if precise*. A wrong push is
  worse than nothing → push **only above a high confidence bar**, otherwise make
  it pullable.
- **TOPIC / FYI, FOLLOW-UP** → low urgency → better as pull or pinned-for-later.

There is also a **robustness argument** the user intuited: because ASR and
attribution are imperfect, **putting a human in the loop (select/pin) filters
garbage before it is amplified into a prompt.** The pull model is not just calmer
UX — it is structurally more tolerant of upstream errors. And it cuts compute:
RAG/generate runs on the handful of segments the user chose, not every turn.

### Recommendation: user-gated, three primitives
Default **quiet**. Same RAG → generate engine underneath; only the *trigger* changes.
1. **ALERTs stay always-on** — watch words only; rare, high-value, unanticipatable.
2. **Select-to-answer** — highlight a transcript span → answer that selection.
3. **Armed listen-window** — a toggle that runs the proactive pipeline only while
   ON, for a user-declared Q&A stretch; auto-off after N minutes of quiet is an option.
- (Optional) **Pin** — mark a span to revisit in a lull / post-call, no live prompt.

This keeps the one magic-moment (alerts) while removing the always-on noise and the
error-amplification, hands the attention budget to the user, and degrades gracefully
as ASR/attribution improve. The armed-window is the smallest first step: it is the
current push pipeline behind an on/off gate — little new machinery, immediate relief
from the distraction.

---

## 4. Cross-cutting principle and sequencing

**Fix the foundation before the superstructure.** Errors at capture propagate and
get amplified by every stage above them — attribution, cleanup, and the
trustworthiness of prompts all inherit the quality of the audio.

Proposed order:
1. **AEC mic capture** (Section 1) — foundational; unblocks attribution *and*
   quality *and* prompt trust. Do this first.
2. **Refiner scoped to readability** + optional context-biased ASR (Section 2).
3. **Hybrid urgency-gated interaction** with select/pin (Section 3) — prototype
   behind a flag, measure against the current push model.

Re-evaluate 2 and 3 *after* 1 lands, since clean audio changes how much the other
two even need to do.

---

## 5b. Model footprint — right model for each task (raised during review)

Verified wiring:

| Stage | Model | Kind |
|-------|-------|------|
| ASR | LFM2.5-Audio | generative (speech→text) |
| Retrieval embeddings | LFM2.5-Embedding-350M | encoder (retrieval) |
| Trigger classification | LFM2.5-Encoder-350M | encoder (classify) |
| **Answer after RAG** | **LFM2.5-1.2B-Instruct** | **generative** |
| Transcript polish | LFM2.5-1.2B-Instruct (same) | generative |
| Post-meeting notes | LFM2.5-350M-Extract | extractive |

The live "extraction grounding" before the answer is a **heuristic**
(`answer_extractor.py`), not a model. The 350M-Extract model is wired for notes
only (F-507, not yet live).

**The open question:** should the *live RAG answer* stay a 1.2B **generation**, or
become a 350M **extraction** (pull the answer span from retrieved context)?
`answer_extractor.py`'s own thesis — *"small models are weak at 'only use this
context'; extraction structurally prevents hallucination"* — argues extraction is
more faithful, smaller, faster, traceable, at the cost of fluent synthesis. The
interaction redesign (generation becomes rare/user-gated) removes the "must be
instant" pressure, so we can afford to optimize for faithfulness. All three
candidates (350M-Extract, 1.2B-Instruct, 2.6B) are GGUF-runnable through the same
path → decide empirically via experiment **E-01** (see open-decisions-log.md).

## 5. Open questions for decision
- AEC path: extend the existing `audio-tap` Swift helper to add a Voice-Processing
  mic mode, or a separate small helper? (Leaning: extend — one capture toolchain.)
- Do we keep `StreamDeduplicator` as a safety net post-AEC, or delete it once
  channels are clean? (Leaning: keep briefly, then delete to avoid two impls.)
- Interaction: default-quiet with user-gated tap is agreed. Which primitive ships
  first? (Leaning: **armed listen-window** — it is the existing push pipeline behind
  an on/off gate, so smallest build for immediate relief — then add select-to-answer.)
  Do ALERTs remain the only always-on channel, or also gated? (Leaning: always-on.)
- Does named diarization (meeting-SDK per-participant) stay parked, or does the
  AEC clean-channel foundation make basic acoustic diarization good enough?
