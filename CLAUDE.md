# CLAUDE.md — meeting-prompter

Real-time meeting intelligence system. Listens to audio, transcribes via LFM2.5-Audio, detects 4 trigger types (questions, topics, alerts, follow-ups), retrieves context via ColBERT RAG, and generates mode-aware responses using LFM2.5-1.2B-Instruct. Everything runs locally on Apple Silicon. Includes a Tauri desktop app with dual-pane UI (editable transcript + live prompts).

## Key Paths

```
├── coach.py                       # CLI entry point (args, startup)
├── config.yaml                    # All externalized thresholds and settings
├── lib/
│   ├── orchestrator.py            # MeetingOrchestrator — central pipeline coordinator
│   ├── config.py                  # Typed dataclass config loader
│   ├── filters.py                 # Hallucination/noise/normalization filters
│   ├── audio_capture.py           # Streaming mic/BlackHole capture (sounddevice)
│   ├── lfm2_wrapper.py            # LFM2.5-Audio subprocess wrapper (llama.cpp)
│   ├── answer_extractor.py        # Sentence extraction for grounding (fallback)
│   ├── rag_generator.py           # LFM2.5-1.2B-Instruct generation (ChatML)
│   ├── rag_engine.py              # ColBERT + Jaccard fallback orchestration
│   ├── dashboard.py               # CLI dashboard with trigger-type coloring
│   ├── triggers/                  # Multi-mode trigger engine
│   │   ├── types.py               # TriggerType enum, Trigger dataclass
│   │   ├── engine.py              # Orchestrator: runs all triggers, priority sort
│   │   ├── question_trigger.py    # Question detection (patterns + keywords)
│   │   ├── alert_trigger.py       # Watch word scanning with cooldown
│   │   ├── topic_trigger.py       # ColBERT-backed topic detection
│   │   └── followup_trigger.py    # Pause-based follow-up suggestions
│   ├── conversation/              # Conversation intelligence
│   │   ├── buffer.py              # Rolling 90s transcript + trigger routing
│   │   └── meeting_context.py     # YAML meeting context loader
│   ├── generation/                # Mode-aware generation
│   │   ├── prompts.py             # ChatML prompt templates per trigger type
│   │   ├── generator.py           # ModeAwareGenerator — trigger-routed generation
│   │   └── types.py               # GenerationResult dataclass
│   └── colbert/                   # Semantic retrieval module
│       ├── retriever.py           # LFM2-ColBERT-350M + PLAID index
│       ├── chunker.py             # Section-aware markdown chunking
│       ├── index_manager.py       # Index persistence/cache
│       └── normalizer.py          # Sigmoid score normalization
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
│   ├── test_session.py            # Thread-safe queue bridge + turn callbacks
│   ├── test_transcript_buffer.py  # Turn accumulation, boundaries, callbacks
│   └── test_transcript_store.py   # Append, upsert, edit overlay, export
├── models/                        # Symlink → ~/Projects/_models
├── runners/                       # llama.cpp binaries (gitignored)
├── docs/                          # Source documents for RAG (PDF + Markdown)
├── data/colbert_index/            # PLAID index cache (gitignored)
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

# Re-index documents
rm -rf data/colbert_index/

# Tests
pytest                                       # All tests (76 tests)
pytest tests/test_transcript_buffer.py -v    # Buffer tests only
cd app && npx tsc --noEmit                   # TypeScript check
```

## Architecture

### Three-Model Pipeline

| Model | Size | Stage | Latency |
|-------|------|-------|---------|
| LFM2.5-Audio-1.5B | 1.2 GB | Speech → text | ~300ms |
| LFM2-ColBERT-350M | 1.4 GB | Semantic retrieval | ~100ms |
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

### Trigger Types (priority order)

| Type | Priority | Description | Max Tokens |
|------|----------|-------------|------------|
| ALERT | 1 | Watch word detected | 100 |
| QUESTION | 2 | Direct question in speech | 200 |
| TOPIC_MATCH | 3 | Discussion matches docs | 100 |
| FOLLOW_UP | 4 | Pause after topic | 75 |

### Key Design Decisions

- **Turn-based transcript buffering**: Backend accumulates raw ASR chunks into speech turns via pause detection. Frontend receives coherent paragraphs, not fragmented chunks.
- **Dual buffer architecture**: TranscriptBuffer (display) and ConversationBuffer (triggers) operate independently on the same chunk stream.
- **4 trigger types** replace single question detection. Each gets a purpose-built prompt template.
- **Rolling 90s transcript window** provides conversation context for generation.
- **Context budget split**: 30% conversation, 70% RAG context in prompts.
- **Two-stage question pipeline**: extraction grounding → generation. Falls through to direct generation when extraction confidence is low.
- **Section-aware chunking**: Split on markdown headers, 400 tokens, 50 overlap.
- **KV cache reset**: Reset model state before each generation.
- **ChatML format**: `<|im_start|>` / `<|im_end|>` delimiters for LFM2.5-Instruct.
- **Config externalization**: All thresholds in `config.yaml`, typed dataclass loader.
- **Session lifecycle**: Session kept alive after stop for export access; fresh session created on next start.

### WebSocket Protocol

**`/ws/transcript`** — Turn-based transcript streaming:
- Server → Client: `{"type": "transcript_update", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": false}`
- Server → Client: `{"type": "transcript_final", "id": "turn-1", "text": "...", "timestamp": ..., "end_timestamp": ..., "is_final": true}`
- Client → Server: `{"type": "edit", "id": "turn-1", "text": "corrected text"}`

**`/ws/prompts`** — Trigger results:
- Server → Client: `{"type": "prompt", "trigger_type": "question", "trigger_text": "...", "answer": "...", "confidence": 0.75, "method": "hybrid", "latency_ms": 480, "source": "ColBERT + hybrid"}`

### Fallback Chains

ColBERT → Jaccard keyword search. Generation → extraction bullets. Extraction → "no match". LFM2.5 models → LFM2 legacy fallback. Set `RAG_USE_FALLBACK=1` for keyword-only mode.

## Model Registry

Models in `~/Projects/_models/` (shared). Set `MODELS_DIR` env var to override.

| Model | Path | Purpose |
|-------|------|---------|
| LFM2.5-Audio-1.5B | `${MODELS_DIR}/LFM2.5-Audio-1.5B-GGUF/` | ASR via `llama-liquid-audio-cli` |
| LFM2-ColBERT-350M | HuggingFace cache (auto-download) | Semantic retrieval (PLAID) |
| LFM2.5-1.2B-Instruct | `${MODELS_DIR}/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | Generation (ChatML) |

## Configuration

All thresholds externalized to `config.yaml`. Loader: `lib/config.py` with typed dataclasses. Falls back to defaults if no YAML present.

Key settings: n_ctx=4096, max_context_chars=6000, pause_threshold=1.5s, question_score_threshold=0.25, topic_match_threshold=0.50, turn_pause=2.0s, max_turn_duration=30s, watch_words configurable per meeting.

## Conventions

- Python 3.10+ (Apple Silicon required)
- All inference runs locally — no external API calls
- Models resolved via `MODELS_DIR` env var
- Thread safety: `threading.Lock()` in ConversationBuffer, `loop.call_soon_threadsafe` for queue bridge
- All files under 300 lines
- `logging` module throughout (no `print()` in lib/)
- 76 Python tests, TypeScript strict mode
