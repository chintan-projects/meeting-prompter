# CLAUDE.md — meeting-prompter

Real-time meeting intelligence system. Listens to audio, transcribes via LFM2.5-Audio, detects 4 trigger types (questions, topics, alerts, follow-ups), retrieves context via hybrid FTS5 + vector RAG, and generates mode-aware responses using LFM2.5-1.2B-Instruct. Everything runs locally on Apple Silicon. Includes a Tauri desktop app with dual-pane UI (editable transcript + live prompts).

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
│   │   └── meeting_context.py     # YAML meeting context loader
│   ├── generation/                # Mode-aware generation
│   │   ├── prompts.py             # ChatML prompt templates per trigger type
│   │   ├── generator.py           # ModeAwareGenerator — trigger-routed generation
│   │   └── types.py               # GenerationResult dataclass
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
│   ├── main.py                    # FastAPI server + WebSocket endpoints
│   ├── session.py                 # Session manager (bridges audio pipeline → WebSocket)
│   ├── transcript_buffer.py       # Turn-based ASR chunk accumulator
│   ├── transcript_store.py        # Append-only transcript with edit overlay + upsert
│   ├── notes_generator.py         # Structured meeting notes via LLM
│   └── routes/
│       ├── session.py             # POST /session/start|stop, GET /status, POST /reindex
│       ├── transcript.py          # WebSocket /ws/transcript (turn updates + edits)
│       ├── prompts.py             # WebSocket /ws/prompts (trigger results)
│       ├── notes.py               # Notes generate/export/save/download endpoints
│       └── context.py             # Meeting context upload
├── app/                           # Tauri + React frontend
│   ├── src-tauri/src/lib.rs       # Rust shell: spawns Python backend, manages lifecycle
│   ├── src/App.tsx                # Root component, layout, WebSocket connections
│   ├── src/components/
│   │   ├── TranscriptPane.tsx     # Left pane: turn-based editable transcript
│   │   ├── PromptsPane.tsx        # Right pane: live trigger results
│   │   ├── StatusBar.tsx          # Session controls, audio health, elapsed time
│   │   ├── MeetingSetup.tsx       # Pre-meeting context config dialog
│   │   └── NoteEditor.tsx         # Post-meeting structured notes editor
│   ├── src/hooks/
│   │   ├── useWebSocket.ts        # WebSocket connection + reconnect hook
│   │   └── useTranscript.ts       # Transcript state with turn-based upsert
│   └── src/styles/global.css      # Theme variables and animations
├── tests/                         # Colocated Python tests
│   ├── test_audio_capture.py      # Audio level detection + health diagnostics
│   ├── test_lfm2_wrapper.py       # LFM2.5-Audio output parsing
│   ├── test_question_trigger.py   # Rhetorical/tag/self-answer suppression (46 tests)
│   ├── test_session.py            # Thread-safe queue bridge + turn callbacks
│   ├── test_transcript_buffer.py  # Turn accumulation, boundaries, callbacks
│   ├── test_transcript_store.py   # Append, upsert, edit overlay, export
│   └── eval/                      # RAG retrieval quality eval harness
│       ├── rag_eval_dataset.yaml  # 21 queries against real context docs
│       └── test_rag_eval.py       # Hit@1, Hit@3, MRR, confidence analysis
├── tools/audio-tap/               # Swift CLI for per-app audio capture (ScreenCaptureKit)
│   ├── Sources/AudioTap.swift     # ScreenCaptureKit stream → raw float32 PCM stdout
│   └── build.sh                   # Build script → runners/audio-tap binary
├── scripts/                       # Utility scripts (build, setup, diagnostics)
├── models/                        # Symlink → ~/Projects/_models
├── runners/                       # llama.cpp + audio-tap binaries (gitignored)
├── context/                       # Source documents for RAG (PDF + Markdown)
├── data/                          # SQLite RAG index (rag.db, gitignored)
└── output/                        # Saved meeting notes (gitignored)
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

# Re-index documents (delete data/rag.db and restart, or POST /session/reindex)

# Tests
pytest                                       # All tests (464 tests)
pytest tests/test_transcript_buffer.py -v    # Buffer tests only
pytest tests/eval/ -m slow -v               # RAG retrieval eval (requires real docs)
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

Thread safety: TranscriptBuffer guards all mutations with `threading.Lock`. LFM2Wrapper
uses `subprocess.run()` per call (independent subprocesses). RAGAnswerGenerator has an
internal lock for generation. Session pipeline includes try/finally cleanup for both
audio capture threads.

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
- **Per-app audio capture**: ScreenCaptureKit via Swift CLI (`audio-tap`). Captures audio from a specific app by PID. Requires macOS 13+ and Screen Recording permission. Falls back to BlackHole device capture.
- **Thread-safe dual-stream**: TranscriptBuffer uses `threading.Lock` on all public methods. Session pipeline uses try/finally for mic capture cleanup. `_trigger_history` bounded with `deque(maxlen=1000)` to prevent memory leaks in long sessions.

### WebSocket Protocol

**`/ws/transcript`** — Turn-based transcript streaming:
- Server → Client: `{"type": "transcript_update", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": false}`
- Server → Client: `{"type": "transcript_final", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": true}`
- Client → Server: `{"type": "edit", "id": "turn-1", "text": "corrected text"}`

**`/ws/prompts`** — Intelligence results with display metadata:
- Server → Client: `{"type": "prompt", "trigger_type": "question", "trigger_text": "...", "answer": "...", "confidence": 0.75, "method": "hybrid", "latency_ms": 480, "source": "deployment.md", "persistence": "persistent", "dismiss_ms": 0, "display_label": "ANSWER", "display_emoji": "💡"}`
- Dead-end results (`no_match`, `no_context`, `suppressed`, or empty answer) are filtered server-side and never sent to the client.

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

Key settings: n_ctx=4096, max_context_chars=6000, pause_threshold=1.5s, question_score_threshold=0.25, topic_match_threshold=0.50, rag_confidence_minimum=0.35, turn_pause=2.0s, max_turn_duration=30s, watch_words configurable per meeting. RAG: lexical_weight=0.05, semantic_weight=0.95, max_chunk_tokens=400, db_path=data/rag.db. Intelligence panel: min_answer_length=10, dismiss_persistent_ms=0, dismiss_standard_ms=90000, dismiss_ephemeral_ms=45000.

## Conventions

- Python 3.10+ (Apple Silicon required)
- All inference runs locally — no external API calls
- Models resolved via `MODELS_DIR` env var
- Thread safety: `threading.Lock()` in TranscriptBuffer, ConversationBuffer, RAGAnswerGenerator; `loop.call_soon_threadsafe` for queue bridge; `deque(maxlen=)` for bounded history
- All files under 300 lines
- `logging` module throughout (no `print()` in lib/)
- 464 Python tests, 16 frontend tests, TypeScript strict mode
