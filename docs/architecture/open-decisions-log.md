# Open Decisions & Experiments — Register

Single place to track architecture decisions and experiments so none get lost.
Deep reasoning lives in the linked docs; this is the index + status.

**Status legend:** `open` (undecided) · `experimenting` · `leaning <X>` ·
`decided` · `parked` (blocked on external input).
Each decision, once made, is promoted to a short ADR (Decision / Consequences /
Migration) in `docs/architecture/`.

Reasoning source: [transcription-attribution-interaction-investigation.md](transcription-attribution-interaction-investigation.md) (§ refs below).

---

## Decisions

| ID | Decision | Status | Pri | Needs ADR | Ref |
|----|----------|--------|-----|-----------|-----|
| **D-01** | **AEC mic capture** (macOS Voice-Processing I/O) — cancel speaker→mic echo at capture so channel attribution is correct by construction, invariant to headphones/speakers/BlackHole. Foundational: fixes attribution **and** ASR quality **and** prompt trust. | open | high | yes | §1 |
| **D-02** | **User-gated interaction** — default **quiet**; user opens the tap via **armed listen-window** (temporal) or **select-to-answer** (spatial). ALERTs stay the only always-on channel. Replaces always-on push that produced a stream of irrelevant prompts. | open | high | yes | §3 |
| **D-03** | **Answer model selection** — live RAG answer stays 1.2B **generation** vs becomes 350M-Extract **extraction** (or 2.6B). Decide empirically via **E-01**. | open (blocked on E-01) | high | yes | §5b |
| **D-04** | **Refiner scope = readability only** — never a meaning/error-correction stage (an LLM error-corrector hallucinates into a trusted record). | leaning yes | low | no | §2 |
| **D-05** | **StreamDeduplicator after AEC** — keep as thin safety net vs delete once channels are clean. | open | med | no | §1 |
| **D-06** | **Named diarization** via meeting-SDK per-participant streams (Zoom SDK) — the "who by name" ceiling above AEC. | parked (needs SDK creds) | low | maybe | §1 |

## Experiments

| ID | Experiment | Status | Feeds | Ref |
|----|------------|--------|-------|-----|
| **E-01** | **Select-driven model + retrieval comparison harness** — `scripts/exp_model_retrieval_compare.py`. v1 ran; findings below. | ran v1 | D-03, D-02 (select-to-answer) | §5b |

## Related (tracked elsewhere)
- **BUG-004** Chrome crash on per-app capture — `investigating`, evidence overturns the tap hypothesis (see BUGS.yaml).
- Sequencing (from investigation §4): **D-01 first** (foundation), then re-evaluate D-04/D-02, run E-01 to settle D-03.

---

## E-01 — spec

**Goal:** pick the right model for the live-answer task by seeing the paths and
the candidates side-by-side on real input. Doubles as the select-to-answer prototype.

**Input:** a selected transcript span (from a real/sample transcript).

**Panels shown for that span:**
1. **Classification** — LFM2.5-Encoder-350M trigger router → `{question|alert|topic|followup|none}` + confidence. (Decides *if/what*, not which docs.)
2. **Retrieval introspection** — run the query through the hybrid engine and show the stages:
   - BM25 lexical top-k (with scores)
   - vector semantic top-k (cosine)
   - fused (weighted) ranking
   - re-ranked final → which docs/chunks + scores
3. **Answer, 3-way** — feed the *same* retrieved context to each candidate and show answer + latency:
   - LFM2.5-350M-Extract (`extract-023-v1.gguf`) — extractive
   - LFM2.5-1.2B-Instruct — current generative
   - LFM2.5-2.6B — larger generative

**Delivery:** script/CLI harness first (fastest signal, reproducible), then
promote the winning model + the select flow into the Tauri select-to-answer UI.

**Decision output:** which model for D-03 (per trigger type if it differs), and
whether extraction beats generation on faithfulness/latency for live answers.

### E-01 v1 findings (2026-07-21)
Ran on the curated Finetuning-Strategy corpus, span = *"How should we generate
synthetic data to fine-tune a small model without a GPU?"*

1. **Retrieval works, but it's effectively pure-vector.** Top-5 chunks are the
   right docs (fine-tuning-lessons, Synthetic-Data-Guide, SLM-pipeline), but
   `bm25=0.000` on *every* hit — the lexical arm (weight 0.05) contributes
   nothing for this query. Retrieval quality is fine; the BM25 half is idle.
   → tuning question: is 0.05 too low, or is FTS not matching (stopwords/OR)?
2. **The 3-way model comparison is blocked by the inference runtime, not the
   harness.** With the bundled `llama-cpp-python`:
   - **350M-Extract** (`extract-023-v1.gguf`) → "Failed to load model from file"
     (arch/format not supported by this llama.cpp build).
   - **2.6B** (`Q4_K_M.gguf`) → chat-template parse error at load
     (`unknown tag 'generation'` — newer Jinja tag the bundled minja can't parse).
   - Only **1.2B-Instruct** loads. So comparing Extract/2.6B needs the runtime
     updated (or the models reconverted) **first** — a real D-03 prerequisite.
3. **Live-quality flag:** the 1.2B (current live model) answered *"I don't have
   that information in my documents"* on well-retrieved on-topic context. The
   harness fed raw joined chunks (the app does extraction-grounding first), so
   verify whether it's the prompt, context formatting, or an over-conservative
   model. If it reproduces in-app, it's a live under-answering bug.

**Next for E-01:** (a) update/align the llama.cpp runtime to load Extract + 2.6B,
(b) give Extract its field-schema prompt for a fair test, (c) chase the 1.2B
refusal. Then re-run and record the model choice against D-03.
