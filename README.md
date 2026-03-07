# Meeting Prompter

Real-time meeting intelligence that transcribes audio, detects questions and topics, retrieves context via hybrid FTS5 + vector RAG, and generates mode-aware responses. Runs entirely on Apple Silicon using [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) for transcription and [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) for generation. Ships as a native Tauri desktop app with live transcript editing and a CLI mode for headless use.

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

The system uses a **three-model pipeline** with four trigger types and **dual-stream audio capture** for speaker attribution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Dual-Stream Audio Pipeline                       │
│                                                                     │
│  Microphone ─────► Audio Capture ──► LFM2.5-Audio ──► Filter Chain  │
│  BlackHole  ─────►  (dual-stream)    (transcription)   (halluc.     │
│                                                         + noise)    │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                  TranscriptBuffer            ConversationBuffer
                  (turn accumulation)         (rolling 90s window)
                         │                           │
                  Store + WebSocket            Trigger Engine
                  (update/final/            ┌────┬────┬────┐
                   polished/relabeled)    ALERT  Q  TOPIC FOLLOW-UP
                         │                  │    │    │    │
                   Tauri App UI             └────┴────┴────┘
                   (TranscriptPane)                │
                                         RAG (hybrid FTS5 + vector)
                                                   │
                                         Mode-Aware Generator
                                                   │
                                         Dashboard / PromptsPane
```

### Two-Model Pipeline

| Model | Size | Stage | Latency |
|-------|------|-------|---------|
| LFM2.5-Audio-1.5B | 1.2 GB | Speech-to-text | ~300ms |
| LFM2.5-1.2B-Instruct | 700 MB | Mode-aware generation | ~500ms |

Retrieval uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (80 MB) for vector embeddings, combined with SQLite FTS5 for lexical search. No separate model process needed.

### Four Intelligence Modes

The system operates as a **meeting coach**, not an FAQ bot. Each mode has a distinct voice, persistence tier, and visual style:

| Mode | Label | Emoji | What It Does | Persistence |
|------|-------|-------|-------------|-------------|
| ALERT | HEADS UP | :warning: | Key term detected — states what you need to know **right now** | Persistent (manual dismiss) |
| QUESTION | ANSWER | :bulb: | Answers a question heard in conversation, with optional coaching suffix | Persistent (manual dismiss) |
| TOPIC_MATCH | FYI | :pushpin: | Surfaces a **new** fact from your docs that hasn't been mentioned yet | Ephemeral (45s auto-dismiss) |
| FOLLOW_UP | SUGGEST | :speech_balloon: | Coaching nudge — what to say or ask next ("Ask about...", "Mention that...") | Standard (90s auto-dismiss) |

Dead-end suppression: if the system can't produce a useful answer (empty result, low confidence, no matching context), it stays silent rather than showing "I don't have that information."

### Dual-Stream Speaker Attribution

The system captures two audio streams simultaneously for automatic speaker identification:

- **Microphone stream** — captures your voice, labeled as "You"
- **System audio stream** (BlackHole) — captures remote participants

Source-based attribution (Tier 1) gives immediate "You" vs "Others" labels. Neural diarization (Tier 2) further distinguishes individual remote speakers (Speaker A, Speaker B, etc.) using ECAPA-TDNN embeddings on the system audio stream.

Users can **click any speaker label** to rename it (e.g., "Speaker A" → "Alice"). The rename applies retroactively to all past turns and persists for future diarizer results via a name mapping.

### Turn-Based Transcript

Raw ASR chunks (~4 seconds each) are accumulated into coherent speech turns via pause detection on the backend. A dual-buffer architecture keeps display and intelligence independent:

- **TranscriptBuffer** accumulates chunks into turns for the UI (2s pause boundary, 30s max duration)
- **ConversationBuffer** maintains a rolling 90s window for trigger detection

Turns stream to the frontend via WebSocket (`transcript_update` for partial, `transcript_final` for completed) with upsert semantics that preserve user edits.

### Hybrid RAG Pipeline

Each trigger feeds through hybrid retrieval (FTS5 lexical + vector semantic), then a trigger-specific prompt template:

1. **Retrieval**: FTS5 BM25 lexical search (5% weight) + vector cosine similarity (95% weight), fused via weighted sum with min-max normalization. SQLite-backed with section-aware chunking (400 tokens, 50 overlap).
2. **Citations**: Each result carries document path, section heading, heading hierarchy, and page range (for PDFs).
3. **Generation**: LFM2.5-1.2B-Instruct with ChatML format, context budget split (30% conversation, 70% RAG).

Generation falls through to extraction bullets when confidence is low, then to silence (dead-end suppression).

## Desktop App

The Tauri app provides a dual-pane interface: live transcript on the left, intelligence panel on the right.

```
┌──────────────────────────────────────────────────────────────┐
│ Meeting Prompter    ● Recording   00:12:34                    │
├──────────────────────────┬───────────────────────────────────┤
│   TRANSCRIPT             │   INTELLIGENCE               (3) │
│                          │                                   │
│  [12:01] You             │  ⚠️ HEADS UP            75% 📌 ✕ │
│  What about the          │  "pricing"                        │
│  deployment timeline     │  Competitor quoted $2M last       │
│  for Edge SDK?           │  quarter; your offer is $1.5M.    │
│  ●                       │  📎 context/competitive.md        │
│                          │                                   │
│  [12:02] Alice           │  💡 ANSWER               82% 📌  │
│  We're targeting         │  Q: Deployment timeline?          │
│  Q2 for the beta         │  Edge SDK ships Q2 2024 for       │
│  release.                │  mobile and web.                  │
│                          │  You could mention: the March     │
│  [12:03] Speaker B       │  milestone for the Edge SDK.      │
│  And compliance?         │  📎 context/roadmap.md            │
│                          │                                   │
│                          │  📌 FYI                    60%    │
│                          │  pricing model                    │
│                          │  SOC2 Type II completed Jan       │
│                          │  2024 — not yet raised.           │
│                          │  📎 context/compliance.md         │
│                          │                                   │
│                          │  💬 SUGGEST                50% 📌 │
│                          │  migration approach               │
│                          │  Ask about their preferred        │
│                          │  migration timeline for the       │
│                          │  data layer.                      │
│                          │  📎 context/migration.md          │
├──────────────────────────┴───────────────────────────────────┤
│  ⌘R Record  ⌘P Pause  ⌘T Transcript  ⌘N Notes               │
└──────────────────────────────────────────────────────────────┘
```

### Transcript Pane (Left)

- Live transcript with speaker labels (You / Speaker A / Speaker B / etc.)
- Active turn shows a pulsing dot indicator
- Double-click any finalized turn to edit the text
- Click any speaker label to rename it (e.g. "Speaker A" to "Alice")
- Resizable: drag the divider between panes (width persists across sessions)

### Intelligence Panel (Right)

The right pane acts as a **meeting coach** — not an FAQ bot. It provides four kinds of interventions, each with distinct behavior:

**Card anatomy**: Every card shows the mode label + emoji, the trigger text (what was detected), the response, a confidence score, pin/dismiss controls, and the source document.

**Card types**:

- **HEADS UP** (amber border) — A watch word was detected. Direct and urgent. "Competitor quoted $2M last quarter; your offer is $1.5M." Stays on screen until you dismiss it.
- **ANSWER** (blue border) — A question was heard in conversation. Concise factual answer with an optional coaching suffix ("You could mention: ..."). Stays on screen until you dismiss it.
- **FYI** (gray border, compact) — Discussion matches your docs. Surfaces a **new** fact not yet mentioned in conversation — not a summary of what's being said. Auto-dismisses after 45 seconds.
- **SUGGEST** (purple border, italic) — Coaching nudge during a natural pause. Starts with an action verb: "Ask about...", "Mention that...", or "Clarify whether...". Auto-dismisses after 90 seconds.

**Pin and dismiss**:

- Click 📌 on any card to **pin** it — pinned cards appear in a dedicated section at the top and never auto-dismiss
- Click ✕ on persistent or pinned cards to **dismiss** them from view
- Ephemeral and standard cards auto-dismiss based on their timer, unless pinned

**Dead-end suppression**: If the system can't produce a useful answer (empty result, no matching context, answer too short), the card is silently suppressed. You never see "I don't have that information."

**Empty state**: When no intelligence is active, the panel shows: *"Listening for questions, topics, and opportunities to help."*

### Meeting Setup

Before recording, a setup dialog lets you configure:
- **Audio devices**: Select microphone (your voice) and system audio device (remote participants via BlackHole)
- **Meeting title**: Optional label shown in the status bar
- **Watch words**: Terms that trigger HEADS UP alerts (e.g. competitor names, pricing terms)
- **Agenda items**: Topics to track during the meeting
- **Participants**: Expected attendees

Quick Start bypasses setup with default devices for fast iteration.

### Post-Meeting Notes

After stopping a recording, click **Export Notes** (or `Cmd+N`) to generate structured meeting notes:
- Summary of the discussion
- Key decisions made
- Action items (attributed to speakers)
- Export to Markdown or copy to clipboard

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+R` | Toggle recording (start/stop) |
| `Cmd+P` | Pause / resume session |
| `Cmd+T` | Toggle transcript pane visibility |
| `Cmd+N` | Toggle notes editor (after recording) |
| `Escape` | Close active modal (notes or setup) |

## Hybrid Retrieval: Why FTS5 + Vector?

Pure keyword search misses semantic matches. Pure embedding search misses exact terminology. The hybrid approach combines both:

| Signal | Method | Weight | Strength |
|--------|--------|--------|----------|
| Lexical | FTS5 BM25 | 5% | Exact keyword matches, proper nouns, acronyms |
| Semantic | all-MiniLM-L6-v2 cosine | 95% | Conceptual similarity across vocabulary |

Both signals are min-max normalized to [0, 1] and fused via weighted sum. The lexical signal acts as a safety net — when someone says "SOC2", BM25 finds it even if the embedding doesn't rank it top-k.

Documents are parsed with structure awareness (markdown headings, PDF pages), chunked into 400-token segments with 50-token overlap, and stored in SQLite with FTS5 virtual tables and vector embeddings. The index supports incremental updates without full rebuilds.

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
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | ~80 MB | Sentence embeddings for hybrid RAG retrieval |
| [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ~100 MB | Speaker diarization embeddings |

Both models download automatically on first use via HuggingFace cache. The RAG index is stored in `data/rag.db` (SQLite) and rebuilt incrementally.

## Project Structure

```
meeting-prompter/
├── coach.py                          # CLI entry point
├── config.yaml                       # Externalized thresholds and settings
├── lib/
│   ├── orchestrator.py               # MeetingOrchestrator — pipeline coordinator
│   ├── config.py                     # Typed dataclass config loader
│   ├── filters.py                    # Hallucination/noise/normalization
│   ├── audio_capture.py              # Dual-stream mic/BlackHole capture
│   ├── text_refiner.py               # LLM-powered transcript polishing
│   ├── lfm2_wrapper.py               # LFM2.5-Audio subprocess wrapper
│   ├── answer_extractor.py           # Sentence extraction (grounding fallback)
│   ├── rag_generator.py              # LFM2.5-1.2B-Instruct wrapper (ChatML)
│   ├── rag_engine.py                 # Hybrid RAG adapter (FTS5 + vector)
│   ├── dashboard.py                  # CLI dashboard with trigger coloring
│   ├── triggers/                     # Multi-mode trigger engine
│   │   ├── types.py                  # TriggerType, Trigger, RAGQueryable protocol
│   │   ├── engine.py                 # Runs all triggers, priority sort
│   │   ├── question_trigger.py       # Question detection (patterns + keywords)
│   │   ├── alert_trigger.py          # Watch word scanning with cooldown
│   │   ├── topic_trigger.py          # RAG-backed topic detection
│   │   └── followup_trigger.py       # Pause-based follow-up suggestions
│   ├── conversation/                 # Conversation intelligence
│   │   ├── buffer.py                 # Rolling 90s transcript + triggers
│   │   └── meeting_context.py        # YAML meeting context loader
│   ├── generation/                   # Mode-aware generation
│   │   ├── prompts.py                # ChatML templates per trigger type
│   │   ├── generator.py              # ModeAwareGenerator
│   │   └── types.py                  # GenerationResult dataclass
│   └── rag/                          # Hybrid retrieval pipeline
│       ├── storage/                  # SQLite schema, migrations
│       ├── parser/                   # Document parsers (text, PDF, composite)
│       ├── chunker/                  # Token-based chunking with overlap
│       ├── index/                    # FTS5 + vector indexing
│       ├── retrieval/                # Hybrid fusion engine
│       ├── rank/                     # Heuristic re-ranking
│       ├── embedder.py               # all-MiniLM-L6-v2 sentence embeddings
│       ├── config.py                 # RAGConfig dataclass
│       └── types.py                  # Shared types (Citation, RetrievalResult)
├── src/api/                          # FastAPI backend (for Tauri app)
│   ├── main.py                       # Server + WebSocket endpoints
│   ├── session.py                    # Session lifecycle + speaker diarization
│   ├── transcript_buffer.py          # Turn-based ASR chunk accumulator
│   ├── transcript_store.py           # Append-only store + edit overlay + rename
│   ├── notes_generator.py            # Structured notes via LLM
│   └── routes/
│       ├── session.py                # Start/stop/pause/resume/status/reindex
│       ├── transcript.py             # WebSocket transcript stream + edits + rename
│       ├── prompts.py                # WebSocket trigger results
│       ├── notes.py                  # Notes generate/export/download
│       └── context.py                # Meeting context upload
├── app/                              # Tauri + React frontend
│   ├── src-tauri/src/lib.rs          # Rust shell, spawns Python backend
│   ├── src/
│   │   ├── App.tsx                   # Root layout, WebSocket connections
│   │   ├── components/
│   │   │   ├── TranscriptPane.tsx    # Turn-based transcript + speaker rename
│   │   │   ├── PromptsPane.tsx       # Intelligence panel (pin/dismiss, persistence tiers)
│   │   │   ├── StatusBar.tsx         # Session controls, audio health
│   │   │   ├── MeetingSetup.tsx      # Pre-meeting config + device selection
│   │   │   └── NoteEditor.tsx        # Post-meeting notes editor
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts       # WS connection + reconnect
│   │   │   ├── useTranscript.ts      # Transcript state with upsert
│   │   │   └── useKeyboardShortcuts.ts # Global keyboard shortcut handler
│   │   └── styles/global.css         # Dark theme, pulse animation
│   └── package.json
├── tests/                            # 348 Python tests across 11 files
│   ├── test_audio_capture.py         # Dual-stream capture + health diagnostics
│   ├── test_transcript_buffer.py     # Turn accumulation, boundaries
│   ├── test_transcript_store.py      # Append, upsert, edit, rename, export
│   ├── test_session.py               # Queue bridge, turns, diarization, rename
│   ├── test_text_refiner.py          # Transcript polishing
│   ├── test_notes_generator.py       # Structured notes generation
│   ├── test_diarization.py           # Speaker embedding + clustering
│   ├── test_rag_engine.py            # Hybrid RAG pipeline + parsers + fusion
│   ├── test_filters.py               # Hallucination/noise filters
│   ├── test_lfm2_wrapper.py          # LFM2.5-Audio output parsing
│   └── conftest.py
├── models/                           # Symlink to ~/Projects/_models
├── runners/                          # llama.cpp binaries (gitignored)
├── context/                          # Source documents for RAG
└── data/                             # SQLite RAG index (rag.db, gitignored)
```

## Live Meeting Setup (BlackHole)

To capture audio from video calls:

1. Install BlackHole: `brew install blackhole-2ch`
2. Create Multi-Output Device in Audio MIDI Setup (check both BlackHole 2ch and speakers)
3. Set your meeting app's speaker to "Multi-Output Device"
4. Run: `python coach.py` (CLI) or launch the Tauri app

The app captures both your microphone and system audio simultaneously. Your speech is labeled "You", remote participants are attributed via neural diarization (Speaker A, Speaker B, etc.). Click any speaker label to assign a real name.

## Adding Your Own Docs

Drop PDF or Markdown files in `context/`. The system indexes all documents at startup into a SQLite database (`data/rag.db`).

```bash
cp your-product-docs.pdf context/
python coach.py --mic        # New files indexed automatically on startup
```

To force a full re-index during a running session: `POST /session/reindex`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM (~2GB for models + embeddings)
- Python 3.10+
- Node.js 18+ and Rust (for Tauri app)

## Troubleshooting

**No audio detected**: Check mic with `--list-devices` or verify BlackHole config for meeting mode. The app selects mic and system audio devices independently in the setup dialog.

**Model not found**: Ensure GGUF files are in `models/` (or `MODELS_DIR`) and `llama-liquid-audio-cli` is in `runners/macos-arm64/`.

**Garbled transcriptions**: LFM2.5-Audio can hallucinate on background noise. The hallucination filter catches common patterns, but noisy environments may cause issues.

**Wrong answers**: Delete `data/rag.db` and restart to rebuild the index from scratch.

**Slow first startup**: First run downloads all-MiniLM-L6-v2 (~80MB) and ECAPA-TDNN (~100MB) embeddings. Subsequent runs load from HuggingFace cache.

**Tauri build errors**: Ensure Rust toolchain and Node.js 18+ are installed. Run `cd app && npm install` to install frontend dependencies.

## License

MIT
