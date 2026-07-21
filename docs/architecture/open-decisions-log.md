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
| **D-03** | **Answer model selection** — 2.6B is currently *wired* (config-driven, `models.generation.model_file`, 1.2B fallback) but the call is the operator's, made from the **lab (E-02)**, not by me. 350M-Extract returns structured JSON fields (extraction), not answers → notes/F-507 lane. Prompt tuned: strict grounding + empty-`<think>` prefill + think-strip. **Correction:** I earlier wrote this "DECIDED" and committed the switch without the operator's call — reverted to provisional; 2.6B stays wired pending their judgement via the lab. | leaning 2.6B (provisional; operator to confirm via E-02) | high | yes | §5b, E-01, E-02 |
| **D-07** | **Refiner/answer model coupling** — the transcript refiner shares the answer-model instance. With 2.6B (reasoning, ~1.5–3.5s) that makes *per-turn* refinement slow. Decouple the refiner to a fast small model, or gate/disable it, once D-02 (user-gated) lands. | open | med | no | E-01 |
| **D-04** | **Refiner scope = readability only** — never a meaning/error-correction stage (an LLM error-corrector hallucinates into a trusted record). | leaning yes | low | no | §2 |
| **D-05** | **StreamDeduplicator after AEC** — keep as thin safety net vs delete once channels are clean. | open | med | no | §1 |
| **D-06** | **Named diarization** via meeting-SDK per-participant streams (Zoom SDK) — the "who by name" ceiling above AEC. | parked (needs SDK creds) | low | maybe | §1 |

## Experiments

| ID | Experiment | Status | Feeds | Ref |
|----|------------|--------|-------|-----|
| **E-01** | **Select-driven model + retrieval comparison harness** (CLI) — `scripts/exp_model_retrieval_compare.py`, `exp_pipeline_probe.py`. v1 ran; findings below. Superseded by E-02 (visual). | ran v1 | D-03, D-02 (select-to-answer) | §5b |
| **E-02** | **Visual model & retrieval lab** — `scripts/lab/` (FastAPI + single page). Select a span → see each retrieval stage (BM25 / vector / fused / reranked) with scores **and** all three answer candidates side-by-side; operator marks the winning model. This is the tool for the D-03 call — surfaces options, does not decide. | built + verified live | D-03, D-02 (select-to-answer) | §5b, below |

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

### E-01 findings (2026-07-21)
Ran on the curated Finetuning-Strategy corpus, span = *"How should we generate
synthetic data to fine-tune a small model without a GPU?"*

1. **Retrieval works, but it's effectively pure-vector.** Top-5 chunks are the
   right docs, but `bm25=0.000` on *every* hit — the lexical arm (weight 0.05)
   contributes nothing for this query. Quality is fine; the BM25 half is idle.
   → tuning question: is 0.05 too low, or is FTS not matching (stopwords/OR)?

2. **Model loading — root-caused, mostly fixed (not a model problem).**
   - **2.6B** → the failure was **not** the model: `llama-cpp-python` 0.3.16
     eagerly compiles every embedded chat template with Python `jinja2`, which
     lacks the HF `{% generation %}` tag (the C++ `minja` runtime handles it,
     which is why the CLI loads it fine). Fixed with `lib/llama_compat.py`
     (makes template compile non-fatal; we use raw completion anyway). **2.6B
     now loads + generates**, in-app and in the harness.
   - **350M-Extract** → genuine GGUF↔llama.cpp mismatch:
     `wrong number of tensors; expected 149, got 148` (tied-embedding export
     diff). The bundled llama.cpp can't load *this* GGUF. Fix: run from
     `model.safetensors` via transformers (its native format), or reconvert.
     **Next step, not yet done.**

3. **1.2B refusal — root cause is the PROMPT, not infra.** On context that
   directly answers a question, the 1.2B answers correctly **but appends**
   *"I don't have that information in my documents"* anyway — the refusal
   boilerplate leaks into good answers. A quick prompt rewrite made it *worse*
   (refused everything, duplicated) → the 1.2B is **brittle** to prompt phrasing.
   Proper fix = a `tune-prompts` iteration, not a one-liner.

4. **D-03 signal — 2.6B >> 1.2B on the same context.** Same span, same context:
   - **1.2B** (725 ms): flat *"I don't have that information."*
   - **2.6B** (3401 ms): graceful, honest partial answer — acknowledges what the
     context covers, notes the GPU aspect isn't addressed, still extracts useful
     points. Far more useful for a coach. ~5× latency, acceptable once generation
     is user-gated (D-02). → **leaning: 2.6B for the answer model**, pending the
     Extract comparison + a prompt pass.

**Next for E-01:** (a) add the Extract (transformers/safetensors) runner for a
fair 3-way, (b) add **classifier** (encoder trigger router) + **re-ranker
before/after** panels, (c) `tune-prompts` pass on the answer prompt, then record
the D-03 decision.

### E-01 full pipeline probe (`scripts/exp_pipeline_probe.py`, 2026-07-21)
Detached, one span through **classify → retrieve → rerank → answer**:

- **Classify** works: encoder trigger router (LFM2.5-Encoder-350M + on-disk
  `LFM2.5-TriggerRouter-350M` adapter) → `question` (conf 0.971); heuristic
  question-score 1.000. Agreement.
- **Rerank is idle for this query:** pre- and post-heuristic-rerank order is
  *identical* ("order unchanged"). Combined with `bm25=0.000` on every hit, two
  retrieval components (BM25 arm, heuristic re-ranker) contribute nothing here.
  → revisit whether they earn their place, or need tuning/harder queries.
- **Answer:** 1.2B flat-refuses (~0.8s); **2.6B** gives the graceful, useful
  partial answer (~3.5s). Consistent with the earlier run → **D-03 leans 2.6B.**
- **350M-Extract is runtime-blocked, definitively.** This build is incompatible
  with *both* pinned runtimes: GGUF won't load in llama-cpp-python 0.3.16
  (tensor-count 149≠148); safetensors won't run in transformers 4.56.2 (unknown
  `TokenizersBackend` tokenizer class, then a tensor-shape bug in the LFM2
  remote-code generate path). Works elsewhere on newer runtimes. **To compare
  Extract fairly, run it in an upgraded env (newer transformers) — a runtime
  decision, not a model problem.**

**D-03 status:** leaning **2.6B** for the live answer model on the evidence so far;
final call pending (a) an Extract run on an upgraded runtime and (b) a
`tune-prompts` pass (the 1.2B refusal is prompt-brittleness). BM25 weight and the
heuristic re-ranker are open **tuning** items (both idle on the test query).

### E-01 resolution (2026-07-21) — Extract on upgraded runtime + D-03 decided

**Extract, run properly.** In an isolated overlay venv (`--system-site-packages`
+ `transformers==5.14.1`), the safetensors loaded and generated. Result: it emits
**structured JSON fields** (`{context, strengths[], risks[], teacher_model{…}}`),
grounded and accurate — i.e. it is a **field extractor, not an answerer**. By the
ADR's output-shape-first rule it belongs to the notes/structured path (F-507),
**not** the live answer role. (Confirms the pinned-runtime diagnosis too: it works
on newer transformers, fails on the project's 4.56.2 / llama.cpp 0.3.16.)

**D-03 DECIDED → 2.6B** for the live answer. Same-context 3-way:
- 1.2B: flat-refuses / leaks the refusal into good answers; brittle to prompts.
- 2.6B: grounded, graceful, refuses correctly. **Chosen.**
- Extract: structured JSON → notes lane, not answers.

Wired: `models.generation.model_file: LFM2.5-2.6B-Q4_K_M.gguf` (config-driven,
1.2B fallback in `_resolve_rag_model`).

**tune-prompts (done).** The old prompt had two opposite failures — 1.2B *leaked*
the refusal after answering; 2.6B *ignored grounding* (answered "capital of
France" from world knowledge). Fixed with: strict grounding ("ONLY the CONTEXT …
no outside knowledge") + an empty `<think></think>` prefill (the 2.6B is a
reasoning model; without it it over-reasons and stalls) + `_strip_think` on
output. Validated: answers cleanly + refuses ungrounded questions, ~1.5–2.5s.
Caveat: the prompt is tuned for the 2.6B default; the 1.2B fallback is degraded
by it (already brittle) → acceptable for a fallback. See **D-07** for the refiner
coupling that rides on the same instance.

Still open (tuning, non-blocking): BM25 weight and the heuristic re-ranker (both
idle on the test query).

### E-02 — visual lab (2026-07-21)

Built `scripts/lab/` (FastAPI + one page) because the CLI harnesses (E-01) put the
decision in *my* hands via a findings write-up — the operator asked for a **visual**
harness to make the model call themselves. The lab lays out, for a selected span:
classification (encoder router + heuristic), the four retrieval stages each on its
own (BM25 / vector / fused / reranked, with scores + the sanitized FTS query), and
the three answer candidates side-by-side (1.2B, 2.6B, 350M-Extract) with latency and
a "Pick this" control that records the operator's choice. It surfaces; it does not
decide.

Run:
```
uvicorn scripts.lab.server:app --port 8555            # then open http://localhost:8555
# 350M-Extract panel needs transformers>=5; point it at an overlay venv:
LAB_EXTRACT_PYTHON=/path/to/venv/bin/python uvicorn scripts.lab.server:app --port 8555
```

**Correction the lab surfaced (sharpens the earlier "BM25 idle" claim):** the BM25
arm is *not* dead — on the sample span it returns strong, on-topic lexical hits
(bm25≈13.3 on the right synthetic-data / no-GPU docs). What zeroes it out is the
**fusion math**: min-max normalisation across a handful of hits + a 0.05 weight
collapse the lexical contribution to 0.000 in the fused score. So the tuning lever
is fusion (weight / normalisation), not lexical recall. The reranker is genuinely
inert here (fused order == reranked order). Both now visible at a glance in the lab
rather than asserted — which is the point.
