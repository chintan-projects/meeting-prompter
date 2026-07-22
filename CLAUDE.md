# CLAUDE.md — meeting-prompter

Real-time meeting intelligence system. Listens to audio, transcribes via LFM2.5-Audio, detects 4 trigger types (questions, topics, alerts, follow-ups), retrieves context via hybrid FTS5 + vector RAG, and generates mode-aware responses using LFM2.5-1.2B-Instruct. Everything runs locally on Apple Silicon. Includes a Tauri desktop app with dual-pane UI (editable transcript + live prompts) and optional Notion export/ingest.

## Key Paths

```
├── coach.py                       # CLI entry point (args, startup)
├── config.yaml                    # All externalized thresholds and settings
├── lib/
│   ├── orchestrator.py            # MeetingOrchestrator — central pipeline coordinator
│   ├── config.py                  # Typed dataclass config loader
│   ├── filters.py                 # Hallucination/noise/normalization filters
│   ├── audio_capture.py           # Streaming mic/BlackHole capture (sounddevice)
│   ├── system_audio_capture.py    # Per-app audio capture via ScreenCaptureKit
│   ├── audio_protocol.py          # Shared AudioCapture protocol for type safety
│   ├── paths.py                   # Project root / runner / output path resolution
│   ├── lfm2_wrapper.py            # LFM2.5-Audio subprocess wrapper (llama.cpp)
│   ├── text_refiner.py            # Post-transcription text polishing via LLM
│   ├── stream_dedup.py             # Cross-stream echo suppression (SequenceMatcher)
│   ├── diarization.py             # Neural speaker diarization (Tier 2)
│   ├── answer_extractor.py        # Sentence extraction for grounding (fallback)
│   ├── rag_generator.py           # LFM2.5-1.2B-Instruct generation (ChatML)
│   ├── rag_engine.py              # Hybrid RAG adapter (FTS5 + vector → same query() API)
│   ├── dashboard.py               # CLI dashboard with trigger-type coloring
│   ├── triggers/                  # Multi-mode trigger engine
│   │   ├── types.py               # TriggerType, Trigger, RAGQueryable protocol
│   │   ├── engine.py              # Orchestrator: runs all triggers, priority sort
│   │   ├── question_trigger.py    # Question detection + rhetorical suppression
│   │   ├── alert_trigger.py       # Watch word scanning with cooldown
│   │   ├── topic_trigger.py       # RAG-backed topic detection
│   │   └── followup_trigger.py    # Pause-based follow-up suggestions
│   ├── conversation/              # Conversation intelligence
│   │   ├── buffer.py              # Rolling 90s transcript + trigger routing
│   │   └── meeting_context.py     # YAML meeting context loader (watch words, agenda)
│   ├── generation/                # Mode-aware generation
│   │   ├── prompts.py             # ChatML prompt templates per trigger type
│   │   ├── generator.py           # ModeAwareGenerator — trigger-routed generation
│   │   └── types.py               # GenerationResult dataclass
│   ├── corpus/                    # Corpus prep (D-09/ADR-001): distiller (heuristic/local/cloud),
│   │                              #   readiness score, incremental re-distill, active-corpus state
│   ├── notion/                    # Notion integration (export + RAG ingest)
│   │   ├── client.py              # Notion API client (retry/backoff, rate-limit safe)
│   │   ├── exporter.py            # Meeting notes/transcript → Notion page
│   │   ├── parser.py              # Notion page → markdown
│   │   └── block_converter.py     # Notion block tree ↔ markdown
│   └── rag/                       # Hybrid retrieval pipeline
│       ├── storage/               # SQLite schema + migrations
│       ├── parser/                # Document parsers (text, PDF, composite)
│       ├── chunker/               # Token-based chunking with overlap
│       ├── index/                 # FTS5 lexical + vector semantic indexing
│       ├── retrieval/             # Weighted fusion engine
│       ├── rank/                  # Heuristic re-ranking
│       ├── embedder.py            # all-MiniLM-L6-v2 (384-dim, lazy-load)
│       ├── config.py              # RAGConfig dataclass (14 tunables)
│       └── types.py               # Citation, RetrievalResult, FusedHit
├── src/api/                       # FastAPI backend for Tauri app
│   ├── main.py                    # FastAPI server + WebSocket endpoints (router registration)
│   ├── session.py                 # Session manager (bridges audio pipeline → WebSocket)
│   ├── transcript_buffer.py       # Turn-based ASR chunk accumulator
│   ├── transcript_store.py        # Append-only transcript with edit overlay + upsert
│   ├── notes_generator.py         # Structured meeting notes via LLM
│   └── routes/
│       ├── session.py             # POST /session/start|stop, GET /status, POST /reindex
│       ├── transcript.py          # WebSocket /ws/transcript (turn updates + edits)
│       ├── prompts.py             # WebSocket /ws/prompts (trigger results)
│       ├── notes.py               # Notes generate/export/save/download endpoints
│       ├── notion.py              # GET /notion/status, POST /notion (export + RAG sync)
│       ├── corpus.py              # Prepare-corpus flow: status/upload/distill/readiness/activate
│       └── context.py             # Meeting context upload
├── app/                           # Tauri + React frontend
│   ├── src-tauri/src/lib.rs       # Rust shell: spawns Python backend, manages lifecycle
│   ├── src/App.tsx                # Root component, layout, WebSocket connections
│   ├── src/components/
│   │   ├── TranscriptPane.tsx     # Left pane: turn-based editable transcript
│   │   ├── PromptsPane.tsx        # Right pane: live trigger results
│   │   ├── StatusBar.tsx          # Session controls, audio health, elapsed time
│   │   ├── MeetingSetup.tsx       # Pre-meeting context config dialog
│   │   └── NoteEditor.tsx         # Post-meeting structured notes editor (export + consent)
│   ├── src/hooks/
│   │   ├── useWebSocket.ts        # WebSocket connection + reconnect hook
│   │   ├── useTranscript.ts       # Transcript state with turn-based upsert
│   │   └── useKeyboardShortcuts.ts # Keyboard shortcut bindings
│   └── src/styles/global.css      # Theme variables and animations
├── tests/                         # Colocated Python tests (see Testing below)
│   └── eval/                      # RAG retrieval quality eval harness
│       ├── rag_eval_dataset.yaml  # 21 queries against real context docs
│       └── test_rag_eval.py       # Hit@1, Hit@3, MRR, confidence analysis (@slow)
├── tools/audio-tap/               # Swift CLI for per-app audio capture (ScreenCaptureKit)
│   ├── Sources/AudioTap.swift     # ScreenCaptureKit stream → raw float32 PCM stdout
│   └── build.sh                   # Build script → runners/audio-tap binary
├── scripts/                       # Utility scripts (backend launcher, packaging)
├── models/                        # Symlink → ~/Projects/_models
├── runners/                       # llama.cpp + audio-tap binaries (gitignored)
├── context/                       # Source documents for RAG (PDF + Markdown)
├── data/                          # SQLite RAG index (rag.db, gitignored)
├── output/                        # Saved meeting notes (gitignored)
└── .claude/                       # Claude Code setup — see "Claude Code Workflow" below
```

## Commands

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# CLI modes
python coach.py                              # Live meeting (BlackHole)
python coach.py --mic                        # Test with microphone
python coach.py --test audio.wav             # Test with audio file
python coach.py --context meeting_context.yaml  # With meeting context
python coach.py --create-context             # Create template YAML
python coach.py --list-devices               # List audio devices
python coach.py --verbose                    # Debug logging

# Tauri app
cd app && npm run tauri dev                  # Dev mode (hot reload)

# API server (standalone)
uvicorn src.api.main:app --host 0.0.0.0 --port 8420
curl -s localhost:8420/notion/status | jq    # Notion integration readiness

# Re-index documents (delete data/rag.db and restart, or POST /session/reindex)

# Tests
pytest                                       # Full unit suite (~585 tests)
pytest tests/test_transcript_buffer.py -v    # Buffer tests only
pytest tests/eval/ -m slow -v                # RAG retrieval eval (requires real docs)
cd app && npx tsc --noEmit                   # TypeScript check
```

## Architecture

### Two-Model Pipeline (+ embedding model)

| Model | Size | Stage | Latency |
|-------|------|-------|---------|
| LFM2.5-Audio-1.5B | 1.2 GB | Speech → text | ~300ms |
| all-MiniLM-L6-v2 | 80 MB | Hybrid retrieval (FTS5 + vector) | ~50ms |
| LFM2.5-1.2B-Instruct | 700 MB | Mode-aware generation | ~500ms |

### Pipeline Flow

```
Audio → Transcribe → Noise/Hallucination Filter
                            │
                    ┌───────┴───────┐
                    │               │
            TranscriptBuffer   ConversationBuffer
            (turn accumulation)  (trigger routing)
                    │               │
            WebSocket push    Trigger Engine
          (update/final)    ┌────┬────┬────┐
                    │     ALERT  Q  TOPIC FOLLOW-UP
                    │       │    │    │    │
                    ▼       └────┴────┴────┘
            TranscriptPane         │
            (Tauri UI)    RAG → Generator → PromptsPane
```

### Dual-Stream Audio Pipeline

Captures mic (you) and system audio (others) simultaneously via two independent
audio threads feeding a shared, thread-safe TranscriptBuffer:

```
Mic AudioCapture ──→ LFM2 ASR ──→ source="mic"    ──→ TranscriptBuffer ──→ "You"
                                                        (threading.Lock)
System Audio     ──→ LFM2 ASR ──→ source="system" ──→ TranscriptBuffer ──→ "Others"
 (ScreenCaptureKit)
```

System audio capture supports two backends:
- **BlackHole**: Virtual audio device (loopback) — captures all system audio
- **ScreenCaptureKit** (per-app): Swift CLI (`tools/audio-tap`) captures audio from a specific app via PID

Speaker attribution is two-tier:
- **Tier 1** (deterministic): source="mic" → "You", source="system" → "Others"
- **Tier 2** (neural): Optional diarization on system audio to distinguish individual remote speakers

Cross-stream echo suppression: `StreamDeduplicator` (lib/stream_dedup.py) detects
when both streams produce near-duplicate ASR text from the same speech (acoustic
coupling without headphones). Uses `difflib.SequenceMatcher` with config-driven
thresholds (dual_stream section in config.yaml). Suppressed chunks still signal
silence to maintain turn boundaries.

Per-source silence: TranscriptBuffer tracks silence flags per source (`dict[str, bool]`),
so system audio silence doesn't prematurely finalize mic turns and vice versa.

Thread safety: TranscriptBuffer guards all mutations with `threading.Lock`. LFM2Wrapper
uses `subprocess.run()` per call (independent subprocesses). RAGAnswerGenerator has an
internal lock for generation. Session pipeline includes try/finally cleanup for both
audio capture threads. StreamDeduplicator uses its own lock for concurrent check() calls.

### Turn-Based Transcript Architecture

Raw ASR chunks (~4s each) are accumulated into coherent speech turns before
reaching the UI. This happens at the `TranscriptBuffer` layer:

| Component | Role |
|-----------|------|
| `TranscriptBuffer` | Accumulates chunks into turns via pause detection (2s gap) |
| `TranscriptStore` | Persists turns with upsert semantics + edit overlay |
| `Session._on_turn_update/final` | Bridges buffer callbacks → WebSocket queue |
| WebSocket `/ws/transcript` | Streams `transcript_update` (partial) and `transcript_final` (complete) |
| `useTranscript` hook | Upserts by turn ID — updates existing or creates new |
| `TranscriptPane` | Renders turns as paragraphs with active turn indicator |

Two independent buffers receive the same raw chunks:
- **TranscriptBuffer**: For display (turn accumulation → UI)
- **ConversationBuffer**: For intelligence (trigger detection → RAG → generation)

### Intelligence Modes (priority order)

| Mode | Label | Persona | Max Tokens | Persistence |
|------|-------|---------|------------|-------------|
| ALERT | HEADS UP | Direct alert — what you need to know now | 100 | persistent |
| QUESTION | ANSWER | Concise answer + optional coaching suffix | 200 | persistent |
| TOPIC_MATCH | FYI | Surface new fact from docs (not conversation echo) | 100 | ephemeral (45s) |
| FOLLOW_UP | SUGGEST | Coaching nudge — "Ask about...", "Mention that..." | 75 | standard (90s) |

Adding a new mode touches six files in a fixed order — use the `add-trigger` skill.

### Post-Meeting Notes & Notion Export

After a session stops, the session is kept alive so notes/transcript remain available
for export (see BUG-003). `notes_generator.py` produces structured meeting notes via the
LLM; `NoteEditor.tsx` lets the user review/edit before anything leaves the machine.
Export is **consent-gated** — notes leave the device only when the user exports.

Notion integration (`lib/notion/`, `routes/notion.py`) does two independent things:
- **Export**: `POST /notion` creates a child page under `export_parent_page_id` from
  notes + optional transcript (`include_transcript` opt-in). Returns the page URL.
- **RAG ingest**: pages/databases in `rag_source_page_ids` / `rag_source_database_ids`
  are fetched, converted to markdown, chunked, and indexed alongside `context/` docs.

Disabled by default; enable in `config.yaml` (`notion.enabled`) + `NOTION_API_TOKEN`
env var. Rate limits (429) handled via `max_retries` / `initial_backoff_s`. Use the
`notion-sync` skill for setup and operation.

### Prepare Corpus (F-701..F-706, D-08/D-09, ADR-001)

Retrieval-first product: live answers are borrowable spans of the user's corpus, so
corpus quality is the ceiling. The "Prepare corpus" flow (`CorpusPrep.tsx` from
Meeting Setup → `routes/corpus.py`) runs: add sources → **distill** into grounded,
provenance-tagged answer-units (`lib/corpus/distiller.py`; backends: `local`
on-device model = shipped default, `heuristic` floor, `cloud` offline-dev only) →
**readiness score** (`lib/corpus/readiness.py`, local rater: answer-shapedness +
retrieval confidence + term overlap; `POST /corpus/readiness`) → **activate**
(`data/corpus_active.json`, own index DB, applies at next session start).
Re-distills are incremental per content hash (`lib/corpus/incremental.py`).
Held-out eval set: `tests/eval/corpus_questions.yaml` (21 questions — never tune
the distiller against it). Cloud judge (`scripts/lab/judge.py`) stays offline for
calibration.

### Key Design Decisions

- **Turn-based transcript buffering**: Backend accumulates raw ASR chunks into speech turns via pause detection. Frontend receives coherent paragraphs, not fragmented chunks.
- **Dual buffer architecture**: TranscriptBuffer (display) and ConversationBuffer (triggers) operate independently on the same chunk stream.
- **4 intelligence modes** with coaching voice persona. Dead-end suppression (F-202): empty/low-quality answers silently filtered at generator and session layers. Persistence tiers control auto-dismiss (configurable in config.yaml).
- **Rolling 90s transcript window** provides conversation context for generation.
- **Context budget split**: 30% conversation, 70% RAG context in prompts.
- **Two-stage question pipeline**: extraction grounding → generation. Falls through to direct generation when extraction confidence is low. Rhetorical question suppression (F-201): tag questions, self-answering, rhetorical forms filtered before scoring.
- **Hybrid RAG**: FTS5 BM25 (5%) + vector cosine (95%) weighted fusion with heuristic re-ranking. Raw cosine similarity (not min-max) for semantic scores to preserve confidence discrimination. FTS5 queries use OR with stop word removal. SQLite-backed with incremental indexing. Citations carry document path, section heading, page range. Eval: Hit@1=94.4%, MRR=0.972 on 21-query benchmark.
- **Section-aware chunking**: Split on markdown headers, 400 tokens, 50 overlap.
- **KV cache reset**: Reset model state before each generation.
- **ChatML format**: `<|im_start|>` / `<|im_end|>` delimiters for LFM2.5-Instruct.
- **Config externalization**: All thresholds in `config.yaml`, typed dataclass loader.
- **Session lifecycle**: Session kept alive after stop for export access; fresh session created on next start.
- **Consent-gated export**: Notes/transcript stay local until the user explicitly exports (e.g. to Notion).
- **Per-app audio capture**: ScreenCaptureKit via Swift CLI (`audio-tap`). Captures audio from a specific app by PID. Requires macOS 13+ and Screen Recording permission. Falls back to BlackHole device capture.
- **Thread-safe dual-stream**: TranscriptBuffer uses `threading.Lock` on all public methods. Session pipeline uses try/finally for mic capture cleanup. `_trigger_history` bounded with `deque(maxlen=1000)` to prevent memory leaks in long sessions.

### WebSocket Protocol

**`/ws/transcript`** — Turn-based transcript streaming:
- Server → Client: `{"type": "transcript_update", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": false, "speaker": "...", "source": "mic|system", "low_confidence": false}`
- Server → Client: `{"type": "transcript_final", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": true, "speaker": "...", "source": "...", "low_confidence": false}`
- Server → Client: `{"type": "transcript_relabeled", "id": "turn-1", ..., "speaker": "Others (room)", "low_confidence": true}` — attribution updated (diarization / conference-room degradation, F-606)
- Client → Server: `{"type": "edit", "id": "turn-1", "text": "corrected text"}`
- `low_confidence: true` marks a flagged best-effort speaker label (conference-room regime) — the UI shows a "~ best guess" badge.

**`/ws/prompts`** — Intelligence results with display metadata:
- Server → Client: `{"type": "prompt", "trigger_type": "question", "trigger_text": "...", "answer": "...", "confidence": 0.75, "method": "retrieval", "latency_ms": 480, "source": "deployment.md", "heading": "Part 1 > 1.3", "source_text": "full borrowable unit ...", "persistence": "persistent", "dismiss_ms": 0, "display_label": "ANSWER", "display_emoji": "💡"}`
- The default live path is **retrieval-first** (F-705/D-08, `triggers.retrieval_first`): `method: "retrieval"`, `answer` = glanceable sentence(s) of the best borrowable unit, `source_text` = the full unit for expand-to-source, `heading` = provenance. LLM answers (`method: "hybrid"/...`) appear only from the legacy path (`retrieval_first: false`) or on-demand.
- Client → Server (HTTP, not WS): `POST /prompts/generate {"trigger_text": "...", "trigger_type": "question"}` → `{"answer", "confidence", "method", "latency_ms", "source"}` — the demoted, user-gated generation path (D-02).
- Dead-end results (`no_match`, `no_context`, `suppressed`, or empty answer) are filtered server-side and never sent to the client.

When adding or changing a message shape, update this section in the same change — it is
the contract both sides read. Use the `add-api-endpoint` skill.

### Fallback Chains

Hybrid retrieval (FTS5 + vector) → low-confidence silence. Generation → extraction bullets. Extraction → "no match" (suppressed). LFM2.5 models → LFM2 legacy fallback.

## Model Registry

Models in `~/Projects/_models/` (shared). Set `MODELS_DIR` env var to override.

| Model | Path | Purpose |
|-------|------|---------|
| LFM2.5-Audio-1.5B | `${MODELS_DIR}/LFM2.5-Audio-1.5B-GGUF/` | ASR via `llama-liquid-audio-cli` |
| all-MiniLM-L6-v2 | HuggingFace cache (auto-download) | Sentence embeddings for hybrid RAG |
| LFM2.5-1.2B-Instruct | `${MODELS_DIR}/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | Generation (ChatML) |

## Configuration

All thresholds externalized to `config.yaml`. Loader: `lib/config.py` with typed dataclasses. Falls back to defaults if no YAML present.

Key settings: n_ctx=4096, max_context_chars=6000, pause_threshold=1.5s, question_score_threshold=0.25, topic_match_threshold=0.50, rag_confidence_minimum=0.35, turn_pause=2.0s, max_turn_duration=30s, watch_words configurable per meeting. RAG: lexical_weight=0.05, semantic_weight=0.95, max_chunk_tokens=400, db_path=data/rag.db. Intelligence panel: min_answer_length=10, dismiss_persistent_ms=0, dismiss_standard_ms=90000, dismiss_ephemeral_ms=45000. Dual-stream echo suppression: window_seconds=8.0, similarity_threshold=0.55, short_text_threshold=0.75. Notion: enabled=false, api_token_env=NOTION_API_TOKEN, max_retries=3.

## Testing

~585 Python tests across ~22 files (colocated in `tests/`), plus 16 frontend tests
and TypeScript strict mode. Notable suites:

- `test_question_trigger.py` — rhetorical/tag/self-answer suppression
- `test_stream_dedup.py` — cross-stream echo detection
- `test_transcript_buffer.py` / `test_transcript_store.py` — turn accumulation, upsert, edit overlay, export
- `test_session.py` — thread-safe queue bridge + turn callbacks
- `test_prompt_experience.py` — coaching persona + dead-end suppression contracts
- `test_notion_exporter.py` / `test_notion_parser.py` / `test_notion_block_converter.py` — Notion export + ingest
- `test_notes_generator.py`, `test_diarization.py`, `test_system_audio_capture.py`, `test_rag_engine.py`, `test_rag_pipeline_document.py`
- `tests/eval/` — RAG retrieval eval (`@pytest.mark.slow`): Hit@1, Hit@3, MRR

Run `pytest` for the fast suite; `pytest -m slow` for the retrieval eval (needs real docs).

## Claude Code Workflow

This repo is set up for Claude Code (`.claude/`). Keep it healthy — it is how sessions
stay grounded across handoffs.

### Tracking files (source of truth for state)
- `PROGRESS.yaml` — LEAN active state only (current phase, active work, blockers). Must stay under ~50 lines.
- `PROGRESS-archive.yaml` — completed items and old sessions.
- `BUGS.yaml` — bug tracker (severity + status), captured via `/log-bug`, fixed via `/fix-bug`.
- `FEATURES.yaml` — feature/workstream backlog with IDs (F-NNN), status, effort, dependencies.

### Slash commands (`.claude/commands/`)
- `/progress` — status dashboard (read-only) from PROGRESS.yaml + BUGS.yaml.
- `/log-bug` — interrupt-safe bug capture; appends to BUGS.yaml and returns to work.
- `/fix-bug BUG-NNN` — TDD micro-session: regression test → minimal fix → verify.
- `/session-end` — checkpoint: update PROGRESS.yaml, archive old sessions, commit state.
- `/rag-eval` — run the retrieval eval and report Hit@1 / Hit@3 / MRR.
- `/add-trigger <name>` — scaffold a new intelligence trigger type end-to-end.
- `/audio-check` — diagnose the dual-stream audio pipeline.
- `/meeting-setup` — author a meeting_context.yaml to prime the coach.

### Skills (`.claude/skills/`, project-local)
- `architecture-map`, `principles-check`, `prd-to-features`, `bug-fix`
- `add-trigger` — new trigger type across enum, engine, config, prompts, tests.
- `tune-rag` — run the eval harness, tune fusion weights/thresholds safely.
- `audio-debug` — diagnose devices, levels, echo, speaker labels.
- `add-api-endpoint` — FastAPI route/WS message + matching React hook, protocol kept in sync.
- `tune-prompts` — refine the coaching-voice prompts per intelligence mode.
- `meeting-context` — build meeting_context.yaml (watch words, agenda, participants).
- `notion-sync` — configure/operate Notion export + RAG ingest.

(Personal workflow skills — gstack `ship`/`review`/`qa`/etc. — are symlinked in and are
not project-specific.)

### Agents (`.claude/agents/`)
`architecture-scout`, `bug-triager`, `code-reviewer`, `test-writer`.

### Hooks (`.claude/settings.json`)
SessionStart loads state; PreToolUse validates Bash + protects files; PostToolUse
auto-formats (ruff/black) and notifies on test failures; Stop runs a smoke test and a
diff-review agent; TaskCompleted validates completion.

## Conventions

- Python 3.10+ (Apple Silicon required)
- All inference runs locally — no external API calls (Notion export is the only opt-in, consent-gated network egress)
- Models resolved via `MODELS_DIR` env var
- Thread safety: `threading.Lock()` in TranscriptBuffer, ConversationBuffer, RAGAnswerGenerator; `loop.call_soon_threadsafe` for queue bridge; `deque(maxlen=)` for bounded history
- All files under 300 lines
- `logging` module throughout (no `print()` in lib/)
- ~585 Python tests, 16 frontend tests, TypeScript strict mode
