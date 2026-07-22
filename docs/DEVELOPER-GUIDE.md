# Developer Guide

Everything needed to go from a clean machine to a running build, plus how to work on
each layer. If something here is wrong, that is a bug — fix the doc in the same PR.

- Product overview → [README.md](../README.md)
- Why the system is shaped this way → [ARCHITECTURE.md](../ARCHITECTURE.md)
- How the corpus distiller works → [distillation.md](distillation.md)

---

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| macOS 13+ on Apple Silicon (M1–M4) | Metal inference, ScreenCaptureKit per-app audio | `uname -m` → `arm64` |
| 16 GB RAM minimum | ~4–5 GB of models resident during a call | — |
| Python 3.10+ | backend, pipeline, tests | `python3 --version` |
| Node.js 18+ | Tauri frontend | `node --version` |
| Rust (stable) | Tauri shell | `rustup --version` |
| Xcode command line tools | Swift build for the audio tap | `xcode-select -p` |

The system is Apple-Silicon-only by design, not by accident: ASR runs on Metal via
llama.cpp, and per-app audio capture uses ScreenCaptureKit. There is no CUDA path.

---

## 2. First build

```bash
git clone <repo-url> && cd meeting-prompter
```

### 2.1 Python backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Models

Models live in a **shared registry outside the repo** so several projects can use one
copy. The default location is `~/Projects/_models`; override with `MODELS_DIR`.

```bash
export MODELS_DIR=~/Projects/_models     # add to your shell profile
```

`models/` in the repo is a symlink to that directory. See
[§3 Models](#3-models-what-to-download) for exactly which files are required and which
are optional.

### 2.3 Runner binaries

Two native binaries live in `runners/macos-arm64/` (gitignored — they are built or
downloaded, never committed):

| Binary | Source | Purpose |
|---|---|---|
| `llama-liquid-audio-macos-arm64` | llama.cpp build with Liquid audio support | LFM2.5-Audio ASR |
| `audio-tap` | `tools/audio-tap/build.sh` | ScreenCaptureKit per-app audio capture |

Build the audio tap:

```bash
./tools/audio-tap/build.sh          # → runners/macos-arm64/audio-tap
./runners/macos-arm64/audio-tap --check-permission
```

The permission check must print `granted`. If it prints `denied`, grant **Screen
Recording** to the terminal you are launching from (System Settings → Privacy &
Security → Screen & System Audio Recording). macOS attributes this permission to the
*responsible process* at the top of the spawn chain, so the grant follows whatever
launches the app — launching from a different tool inherits *that* tool's grant, and a
denied one silently degrades the session to mic-only.

### 2.4 Frontend

```bash
cd app && npm install && cd ..
```

### 2.5 Run it

```bash
cd app && npm run tauri dev
```

The Rust shell spawns the Python backend itself on `127.0.0.1:8420`. **Do not** start
uvicorn separately or the port will conflict.

---

## 3. Models: what to download

Set `MODELS_DIR` and place these under it. Sizes are on-disk quantized.

### Required

| Directory / file | Role | Notes |
|---|---|---|
| `LFM2.5-Audio-1.5B-GGUF/` | ASR | multi-file GGUF; needs the `llama-liquid-audio-cli` runner |
| `LFM2.5-Embedding-350M/` | retrieval embeddings | 1024-dim, loaded via `trust_remote_code` |
| `LFM2.5-2.6B-Q4_K_M.gguf` | generation (on-demand) | falls back to `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` if absent |

### Optional (features degrade gracefully without them)

| Directory / file | Enables | Without it |
|---|---|---|
| `LFM2.5-Encoder-350M/` | encoder intelligence layer (F-501/510) | heuristic heads only |
| `LFM2.5-TriggerRouter-350M/` | learned trigger router (F-503) | falls back to heuristics |
| `LFM2.5-350M-Extract-023-v1/` | structured notes extraction (F-507) | generic instruct + prompts |
| `LFM2.5-Audio-600M-mini-GGUF/` | lighter ASR fallback | 1.5B only |
| `LFM2.5-VL-450M-Extract/` | slide/visual context (unbuilt, F-603) | — |
| `speechbrain/spkrec-ecapa-voxceleb` (HF cache, auto) | Tier-2 diarization | You/Others attribution only |

**Everything is optional-by-design at the module level.** The encoder lazy-loads on
first use and the heuristic path never touches it, which is why the whole test suite
runs on a machine with no models present.

Change models via `config.yaml`, never in code — `models.generation.model_file`,
`rag.embedding_model`, `diarization.embedding_model`. There are no hardcoded model
paths in `lib/`.

---

## 4. Running

### Desktop app

```bash
cd app && npm run tauri dev        # dev mode, hot reload
```

### Backend alone (for API work)

```bash
source venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8420 --reload
curl -s localhost:8420/status | jq
```

### CLI (headless, no Tauri)

```bash
python coach.py --mic                        # microphone
python coach.py                              # live meeting (BlackHole)
python coach.py --test audio.wav             # audio file
python coach.py --context meeting_context.yaml
python coach.py --create-context             # write a template
python coach.py --list-devices
python coach.py --verbose                    # debug logging
```

### Corpus lab (offline research harness)

```bash
./scripts/lab/run.sh              # FastAPI + single-page rating/judging UI
```

The lab is a **development instrument**, not part of the product. It is the only place
cloud Claude is used, and only for offline calibration — see
[distillation.md](distillation.md).

---

## 5. Tests, lint, types

Everything below must be green before a commit. There is no exception for
"docs-only" changes that touch code paths.

```bash
pytest                               # ~800 Python tests, no models needed
pytest tests/eval/ -m slow -v        # retrieval eval — needs real docs indexed
ruff check .
mypy --strict lib/ src/
cd app && npx tsc --noEmit && npm test
```

Useful subsets while iterating:

```bash
pytest tests/test_corpus_distiller.py -v
pytest tests/test_transcript_buffer.py tests/test_session.py -v
pytest -k "trigger" -v
```

Hooks auto-format on write (`black --line-length=100` then `ruff check --fix` for
Python; prettier + eslint for TS). If a hook strips an import you were about to use,
re-add it — that has bitten this repo before.

---

## 6. Layout: where things live

```
coach.py                  CLI entry point
config.yaml               every threshold and model choice
lib/
  orchestrator.py         MeetingOrchestrator — the pipeline coordinator
  audio_capture.py        mic streaming (sounddevice)
  system_audio_capture.py per-app capture via the Swift audio-tap
  stream_dedup.py         cross-stream echo suppression
  lfm2_wrapper.py         ASR subprocess wrapper
  warm_runtime.py         single owner of load-once, stay-warm models (F-508)
  intelligence/           encoder backbone + pluggable heads
    encoder.py            LFM2.5-Encoder-350M, mean-pooled, frozen
    heads/                base protocol, heuristic heads, linear probe, trigger router
    turn_state.py         typed per-turn workspace (replaces the daisy-chain)
  attribution/            resolver hierarchy + lexical consistency
  triggers/               question / alert / topic / followup + engine
  conversation/           rolling buffer + meeting context
  generation/             ChatML prompts, mode-aware generator
  corpus/                 distiller, readiness, incremental, active-corpus state
  rag/                    parser → chunker → index (FTS5 + vector) → fusion → rank
  notion/                 export + RAG ingest
src/api/
  main.py                 FastAPI app, router registration
  session.py              bridges the audio pipeline to WebSockets
  transcript_buffer.py    turn accumulation from raw ASR chunks
  transcript_store.py     append-only store with edit overlay
  routes/                 session, transcript, prompts, notes, notion, corpus, context
app/
  src-tauri/src/lib.rs    Rust shell — spawns and supervises the Python backend
  src/App.tsx             root layout + WebSocket wiring
  src/components/         TranscriptPane, PromptsPane, CorpusPrep, MeetingSetup, …
tools/audio-tap/          Swift CLI: ScreenCaptureKit → float32 PCM on stdout
scripts/lab/              offline corpus/retrieval lab (not shipped)
tests/                    colocated tests; tests/eval/ holds the retrieval eval
```

Hard rules, enforced in review: **max 300 lines per file**, composition over
inheritance, full type annotations, no `print()` in `lib/`, no hardcoded config.

---

## 7. Common tasks

### Add a trigger type

Six files in a fixed order. Use the `add-trigger` skill (`/add-trigger <name>`), which
scaffolds enum → engine registration → config threshold → prompt template → session
display metadata → tests. Doing it by hand reliably misses one.

### Add an API endpoint or WebSocket message

Use the `add-api-endpoint` skill. The WebSocket protocol in
[CLAUDE.md](../CLAUDE.md#websocket-protocol) is the contract both sides read — update
it in the same change, or the frontend and backend drift silently.

### Change retrieval quality

```bash
pytest tests/eval/ -m slow -v      # Hit@1, Hit@3, MRR on the 21-query benchmark
```

Tune `rag.lexical_weight` / `rag.semantic_weight` / thresholds in `config.yaml`, never
in code. The `tune-rag` skill runs the loop safely. Baseline to beat: Hit@1 94.4%,
MRR 0.972.

### Re-index documents

Delete `data/rag.db` and restart, or `POST /session/reindex` on a running backend.
Changing `paths.docs_dir` requires a delete — the index does not detect the swap.

### Work on the corpus flow

```bash
curl -s localhost:8420/corpus/status | jq
curl -sX POST localhost:8420/corpus/distill -d '{"backend":"local"}' -H 'content-type: application/json'
curl -s localhost:8420/corpus/distill/status | jq
```

Or use the wizard: Meeting Setup → "Prepare corpus…". Activation writes
`data/corpus_active.json` and applies at the **next session start**, not immediately.

---

## 8. Debugging

| Symptom | First thing to check |
|---|---|
| No remote speakers transcribed | `audio-tap --check-permission` from the launching terminal |
| No audio at all | `python coach.py --list-devices`; confirm the device in Meeting Setup |
| Garbled transcript | hallucination filter thresholds in `lib/filters.py`; noisy input is a known ASR weakness |
| Cards never appear | `triggers.retrieval_first`, `rag_confidence_minimum`, and whether the corpus actually covers the topic |
| Cards appear at nonsense moments | set `triggers.f503_router_enabled: false` and restart — that isolates the router from the rest |
| Wrong answers | delete `data/rag.db`, restart, re-index |
| Slow first query | first embed loads the model (~1–2 s); the embedder is pre-warmed on session start |
| Tauri build errors | `cd app && npm install`; confirm the Rust toolchain |

The `audio-debug` skill (`/audio-check`) walks the dual-stream pipeline end to end.

Logs: the backend uses the `logging` module throughout. Run the CLI with `--verbose`
for routing decisions and trigger scores.

---

## 9. Project state files

These are the source of truth for what is done and what is broken. Read them at
session start; they are loaded automatically by the Claude Code hooks.

| File | Contents |
|---|---|
| `PROGRESS.yaml` | active work only, under ~50 lines |
| `PROGRESS-archive.yaml` | completed items, old sessions |
| `BUGS.yaml` | bug tracker (`/log-bug`, `/fix-bug BUG-NNN`) |
| `FEATURES.yaml` | feature backlog with IDs, status, effort, dependencies |
| `docs/architecture/open-decisions-log.md` | every open decision (D-NN) and experiment (E-NN) |

Feature IDs (F-NNN) appear in code comments and commit messages. When you read
`# F-705` in a file, `FEATURES.yaml` explains what it is and why it exists.

---

## 10. Conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
- Every `setup()` has a matching `teardown()` — model processes, threads, file handles.
- No silent failures. Catch, log with structured context, recover via a named fallback.
- Timeouts on every external operation, values from config.
- Thread safety is explicit: `threading.Lock` in the buffers and the generator,
  `loop.call_soon_threadsafe` for the queue bridge, `deque(maxlen=)` for history.
- All inference is local. The only network egress is consent-gated Notion export.
