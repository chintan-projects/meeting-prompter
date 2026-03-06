# Meeting Prompter

Real-time meeting intelligence that transcribes audio, detects questions and topics, retrieves context via ColBERT RAG, and generates mode-aware responses. Runs entirely on Apple Silicon using [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) for transcription and [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) for generation. Ships as a native Tauri desktop app with live transcript editing and a CLI mode for headless use.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![TypeScript](https://img.shields.io/badge/typescript-5.x-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Quick Start

### Desktop App (Tauri)

```bash
git clone <repo-url>
cd meeting-prompter

# Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd app && npm install && cd ..

# Launch (starts Python backend + React UI)
cd app && npm run tauri dev
```

### CLI Mode

```bash
source venv/bin/activate

# Microphone
python coach.py --mic

# Live meeting (BlackHole virtual audio)
python coach.py

# Test with audio file
python coach.py --test audio.wav

# With meeting context (agenda, watch words)
python coach.py --context meeting_context.yaml
```

## Architecture

The system uses a **three-model pipeline** with four trigger types. Each trigger gets a purpose-built prompt template for generation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Audio Pipeline                              │
│                                                                     │
│  Microphone ──► Audio Capture ──► LFM2.5-Audio ──► Filter Chain     │
│                  (4s chunks)      (transcription)   (hallucination  │
│                                                      + noise)       │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                  TranscriptBuffer            ConversationBuffer
                  (turn accumulation)         (rolling 90s window)
                         │                           │
                  Store + WebSocket            Trigger Engine
                  (transcript_update /    ┌────┬────┬────┐
                   transcript_final)    ALERT  Q  TOPIC FOLLOW-UP
                         │                │    │    │    │
                   Tauri App UI           └────┴────┴────┘
                   (TranscriptPane)              │
                                       RAG (ColBERT top-k)
                                                 │
                                       Mode-Aware Generator
                                                 │
                                       Dashboard / PromptsPane
```

### Three-Model Pipeline

| Model | Size | Stage | Latency |
|-------|------|-------|---------|
| LFM2.5-Audio-1.5B | 1.2 GB | Speech-to-text | ~300ms |
| LFM2-ColBERT-350M | 1.4 GB | Semantic retrieval | ~100ms |
| LFM2.5-1.2B-Instruct | 700 MB | Mode-aware generation | ~500ms |

### Four Trigger Types

| Type | Priority | Description | Max Tokens |
|------|----------|-------------|------------|
| ALERT | 1 | Watch word detected | 100 |
| QUESTION | 2 | Direct question in speech | 200 |
| TOPIC_MATCH | 3 | Discussion matches docs | 100 |
| FOLLOW_UP | 4 | Pause after topic | 75 |

### Turn-Based Transcript

Raw ASR chunks (~4 seconds each) are accumulated into coherent speech turns via pause detection on the backend. A dual-buffer architecture keeps display and intelligence independent:

- **TranscriptBuffer** accumulates chunks into turns for the UI (2s pause boundary, 30s max duration)
- **ConversationBuffer** maintains a rolling 90s window for trigger detection

Turns stream to the frontend via WebSocket (`transcript_update` for partial, `transcript_final` for completed) with upsert semantics that preserve user edits.

### Hybrid RAG Pipeline

Each trigger feeds through ColBERT semantic retrieval, then a trigger-specific prompt template:

1. **Retrieval**: ColBERT top-k with section-aware chunking (400 tokens, 50 overlap)
2. **Grounding**: Sentence extraction scores and filters relevant context
3. **Generation**: LFM2.5-1.2B-Instruct with ChatML format, context budget split (30% conversation, 70% RAG)

Fallback chain: ColBERT -> Jaccard keyword search. Generation -> extraction bullets. Extraction -> "no match".

## Desktop App

The Tauri app provides a dual-pane interface:

```
┌─────────────────────────────────────────────────┐
│ Meeting Prompter    [Recording]   00:12:34       │
├──────────────────────┬──────────────────────────┤
│   TRANSCRIPT         │   PROMPTS                │
│                      │                          │
│  [12:01]             │  ALERT: "pricing"        │
│  What about the      │  Our pricing model is... │
│  deployment timeline │                          │
│  for Edge SDK?       │  ANSWER                  │
│  ●                   │  Q: Deployment timeline?  │
│                      │  Edge SDK ships Q2...    │
│  [12:02]             │                          │
│  We're targeting     │  TOPIC: compliance       │
│  Q2 for the beta     │  SOC2 audit completed... │
│  release.            │                          │
│                      │  FOLLOW-UP               │
│  [12:03]             │  Ask about HIPAA status  │
│  And compliance?     │                          │
├──────────────────────┴──────────────────────────┤
│  Settings  Export Notes  Mic: BlackHole          │
└─────────────────────────────────────────────────┘
```

- **Left pane**: Live transcript with turn-based display. Active turns show a pulsing indicator. Double-click finalized turns to edit.
- **Right pane**: Trigger results color-coded by type and priority-sorted (alerts on top).
- **Meeting setup**: Configure agenda, watch words, and upload docs before recording.
- **Meeting notes**: After recording, generate structured notes (summary, decisions, action items) and export to Markdown.

## ColBERT: Why Late Interaction?

Traditional keyword matching fails for semantic queries:

| Query | Keyword Result | ColBERT Result |
|-------|----------------|----------------|
| "What is Liquid AI?" | Found | Found (77%) |
| "neural network alternatives" | **MISS** | **Found (74%)** |
| "compete with OpenAI" | **MISS** | **Found (76%)** |

ColBERT creates **one vector per token** (128-dim) and uses **MaxSim** scoring to find token-level semantic matches. "Neural" matches "model", "alternatives" matches "architecture" -- even without keyword overlap.

Documents are chunked with section awareness: split on markdown headers first, then into 400-token segments with 50-token overlap. Section headers prepended to each chunk for retrieval context.

## Models

### Local Models

| Model | Size | Purpose |
|-------|------|---------|
| [LFM2.5-Audio-1.5B-Q4_0.gguf](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) | ~1.2 GB | Speech-to-text (4 files) |
| [LFM2.5-1.2B-Instruct-Q4_K_M.gguf](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) | 700 MB | Mode-aware generation |

Download to `~/Projects/_models/` (or set `MODELS_DIR` env var). Also need the `llama-liquid-audio-cli` runner binary in `runners/macos-arm64/`.

### Auto-Downloaded Models

| Model | Size | Purpose |
|-------|------|---------|
| [LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 1.4 GB | Semantic document retrieval |

ColBERT downloads automatically on first run via `pylate`. The PLAID index is built once and cached in `data/colbert_index/`.

## Project Structure

```
meeting-prompter/
├── coach.py                          # CLI entry point
├── config.yaml                       # Externalized thresholds and settings
├── lib/
│   ├── orchestrator.py               # MeetingOrchestrator — pipeline coordinator
│   ├── config.py                     # Typed dataclass config loader
│   ├── filters.py                    # Hallucination/noise/normalization
│   ├── audio_capture.py              # Streaming mic/BlackHole capture
│   ├── lfm2_wrapper.py               # LFM2.5-Audio subprocess wrapper
│   ├── answer_extractor.py           # Sentence extraction (grounding fallback)
│   ├── rag_generator.py              # LFM2.5-1.2B-Instruct wrapper (ChatML)
│   ├── rag_engine.py                 # ColBERT + Jaccard orchestration
│   ├── dashboard.py                  # CLI dashboard with trigger coloring
│   ├── triggers/                     # Multi-mode trigger engine
│   │   ├── types.py                  # TriggerType enum, Trigger dataclass
│   │   ├── engine.py                 # Runs all triggers, priority sort
│   │   ├── question_trigger.py       # Question detection (patterns + keywords)
│   │   ├── alert_trigger.py          # Watch word scanning with cooldown
│   │   ├── topic_trigger.py          # ColBERT-backed topic detection
│   │   └── followup_trigger.py       # Pause-based follow-up suggestions
│   ├── conversation/                 # Conversation intelligence
│   │   ├── buffer.py                 # Rolling 90s transcript + triggers
│   │   └── meeting_context.py        # YAML meeting context loader
│   ├── generation/                   # Mode-aware generation
│   │   ├── prompts.py                # ChatML templates per trigger type
│   │   ├── generator.py              # ModeAwareGenerator
│   │   └── types.py                  # GenerationResult dataclass
│   └── colbert/                      # Semantic retrieval
│       ├── retriever.py              # LFM2-ColBERT-350M + PLAID index
│       ├── chunker.py                # Section-aware markdown chunking
│       ├── index_manager.py          # Index persistence/cache
│       └── normalizer.py             # Sigmoid score normalization
├── src/api/                          # FastAPI backend (for Tauri app)
│   ├── main.py                       # Server + WebSocket endpoints
│   ├── session.py                    # Session lifecycle + queue bridge
│   ├── transcript_buffer.py          # Turn-based ASR chunk accumulator
│   ├── transcript_store.py           # Append-only store + edit overlay
│   ├── notes_generator.py            # Structured notes via LLM
│   └── routes/
│       ├── session.py                # Start/stop/status/reindex
│       ├── transcript.py             # WebSocket transcript stream
│       ├── prompts.py                # WebSocket trigger results
│       ├── notes.py                  # Notes generate/export/download
│       └── context.py                # Meeting context upload
├── app/                              # Tauri + React frontend
│   ├── src-tauri/src/lib.rs          # Rust shell, spawns Python backend
│   ├── src/
│   │   ├── App.tsx                   # Root layout, WebSocket connections
│   │   ├── components/
│   │   │   ├── TranscriptPane.tsx    # Turn-based editable transcript
│   │   │   ├── PromptsPane.tsx       # Live trigger results
│   │   │   ├── StatusBar.tsx         # Session controls, audio health
│   │   │   ├── MeetingSetup.tsx      # Pre-meeting context config
│   │   │   └── NoteEditor.tsx        # Post-meeting notes editor
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts       # WS connection + reconnect
│   │   │   └── useTranscript.ts      # Transcript state with upsert
│   │   └── styles/global.css         # Dark theme, pulse animation
│   └── package.json
├── tests/                            # 76 tests across 5 files
│   ├── test_audio_capture.py
│   ├── test_transcript_buffer.py
│   ├── test_transcript_store.py
│   ├── test_session.py
│   └── conftest.py
├── models/                           # Symlink to ~/Projects/_models
├── runners/                          # llama.cpp binaries (gitignored)
├── docs/                             # Source documents for RAG
└── data/colbert_index/               # PLAID index cache (gitignored)
```

## Live Meeting Setup (BlackHole)

To capture audio from video calls:

1. Install BlackHole: `brew install blackhole-2ch`
2. Create Multi-Output Device in Audio MIDI Setup (check both BlackHole 2ch and speakers)
3. Set your meeting app's speaker to "Multi-Output Device"
4. Run: `python coach.py` (CLI) or launch the Tauri app

## Adding Your Own Docs

Drop PDF or Markdown files in `docs/`. The system loads all documents at startup and builds a ColBERT index.

```bash
cp your-product-docs.pdf docs/
rm -rf data/colbert_index/   # Force re-index
python coach.py --mic
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM (~4.5GB for all three models)
- Python 3.10+
- Node.js 18+ and Rust (for Tauri app)

## Troubleshooting

**No audio detected**: Check mic with `--list-devices` or verify BlackHole config for meeting mode.

**Model not found**: Ensure GGUF files are in `models/` (or `MODELS_DIR`) and `llama-liquid-audio-cli` is in `runners/macos-arm64/`.

**Garbled transcriptions**: LFM2.5-Audio can hallucinate on background noise. The hallucination filter catches common patterns, but noisy environments may cause issues.

**Wrong answers**: Delete `data/colbert_index/` and restart to rebuild the index.

**ColBERT not loading**: Check `pylate` installation. On 8GB Macs, set `RAG_USE_FALLBACK=1` for keyword search.

**Slow first startup**: First run downloads ColBERT (~1.4GB) and builds the PLAID index (~30-60s). Subsequent runs load from cache (~6s).

**Tauri build errors**: Ensure Rust toolchain and Node.js 18+ are installed. Run `cd app && npm install` to install frontend dependencies.

## License

MIT
