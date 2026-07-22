# Meeting Prompter

Real-time meeting intelligence that runs entirely on your laptop. It listens to both
sides of a call, transcribes locally, notices when a question or a watched term comes
up, and surfaces a sentence from *your own documents* that you can read aloud — with a
link back to where it came from.

No audio leaves the machine. No transcript leaves the machine. No corpus leaves the
machine. The only network egress in the whole system is a Notion export you trigger by
hand.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![TypeScript](https://img.shields.io/badge/typescript-5.x-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

- **Build it** → [docs/DEVELOPER-GUIDE.md](docs/DEVELOPER-GUIDE.md)
- **Why it's built this way** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Corpus distillation** → [docs/distillation.md](docs/distillation.md)

---

## What it does

```
┌──────────────────────────────────────────────────────────────┐
│ Meeting Prompter    ■ Stop  ❚❚ Pause  ◉ LISTENING   00:12:34  │
├──────────────────────────┬───────────────────────────────────┤
│   TRANSCRIPT             │   INTELLIGENCE               (3)  │
│                          │                                   │
│  [12:01] You             │  ⚠️  HEADS UP           75% 📌 ✕  │
│  What about the          │  "pricing"                        │
│  deployment timeline     │  Competitor quoted $2M last       │
│  for the Edge SDK?       │  quarter; your offer is $1.5M.    │
│  ●                       │  📎 competitive.md › Pricing      │
│                          │                                   │
│  [12:02] Others          │  💡 ANSWER              82% 📌    │
│  We're targeting Q2      │  Edge SDK ships Q2 2026 for       │
│  for the beta.           │  mobile and web; the March        │
│                          │  milestone covers the SDK core.   │
│  [12:03] Others          │  📎 roadmap.md › Part 2 > 2.1     │
│  And compliance?         │  ▸ expand to source     ✨        │
│                          │                                   │
│                          │  📌 FYI                    60%    │
│                          │  SOC2 Type II completed Jan       │
│                          │  2026 — not yet raised.           │
│                          │  📎 compliance.md › Certifications│
├──────────────────────────┴───────────────────────────────────┤
│  ⌘L listen · ⌘⇧R rec · Space pause · ⌘\ pane                 │
└──────────────────────────────────────────────────────────────┘
```

## It's quiet until you ask

The default is **silence**. Nothing pushes at you while you're talking. You open the
tap two ways, and both are deliberate:

| | How | What happens |
|---|---|---|
| **Temporal** | **⌘L** — arms the listen window | Automatic cards flow while it's open. The status bar shows a green **◉ LISTENING**. ⌘L again closes it. |
| **Spatial** | **Select any transcript text** → 💡 Answer this | Answers that exact span, instantly, whether or not it looked like a question. Works even while quiet. |

The one exception is **watch words** — those alert you always, because you named them
yourself in Meeting Setup. That's the only channel you pre-authorized.

Why this shape: a correctly-detected question is not the same as a question you want
answered on screen, and most of them aren't. A perfect trigger classifier still
interrupts on every true positive. **Prompt spam is a permission problem, not a
classification problem** — so the fix is asking, not better guessing.

Restore the old always-on behavior with `triggers.gating.enabled: false`.

**Four kinds of intervention**, each with its own voice and lifetime:

| | What triggers it | Behavior |
|---|---|---|
| ⚠️ **HEADS UP** | one of your watch words | direct, urgent, stays until dismissed |
| 💡 **ANSWER** | a question detected in the conversation | concise answer, stays until dismissed |
| 📌 **FYI** | discussion matches your docs | a **new** fact, not an echo — auto-dismisses in 45 s |
| 💬 **SUGGEST** | a natural pause | a nudge: "Ask about…", "Mention that…" — 90 s |

If it can't produce something useful, **it says nothing.** You will never see "I don't
have that information." A panel that chatters is worse than a panel that is quiet.

---

## The core idea: retrieval-first

Most meeting assistants send your audio to a server and ask a large model to write
something helpful. This one does the opposite.

**The default live path contains no LLM at all.** A trigger fires, hybrid retrieval runs
against your indexed documents, and the card shows a borrowable span of your own corpus
verbatim — 16–190 ms, with its source heading attached and expand-to-source for the full
passage. Generation exists but is demoted to an explicit button press.

Mid-meeting, a grounded sentence you can say out loud in 50 ms beats a plausible
paragraph in 3 seconds that you have to fact-check before you dare repeat it.

The trade is real and worth stating: **your corpus becomes the ceiling on output
quality**, because no model sits in the path to compensate for a weak source. That is
exactly what the corpus preparation step exists to fix.

---

## Prepare corpus

Source documents are usually *explainers* — prose that teaches a topic across
paragraphs and tables. A meeting needs an *answer bank* — statements that stand alone.
The "Prepare corpus" flow converts the first into the second, offline, on-device:

```
bring your documents  →  distill into answer-units  →  readiness score  →  activate
      (.md/.pdf/.txt)     (provenance-tagged,           (+ gap list)       (next session)
                           on-device model)
```

The **readiness score** is the part worth caring about. Before your first real call it
tells you which of your likely questions your corpus *cannot* answer — while there is
still time to add a document.

Measured lift on a 21-question held-out set: **76% → 90–95%** coverage. Caveats,
methodology, and what is still unproven: [docs/distillation.md](docs/distillation.md).

---

## Models

Everything runs locally on Apple Silicon. Roughly 4.5 GB resident during a call.

| Role | Model | Size | Latency |
|---|---|---|---|
| Speech → text | [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) | 1.2 GB | ~300 ms / chunk |
| Retrieval embeddings | LFM2.5-Embedding-350M (1024-dim) | ~350 MB | ~50 ms |
| Intelligence backbone | LFM2.5-Encoder-350M (frozen, bidirectional) | ~350 MB | ~14 ms / turn |
| Trigger routing | LFM2.5-TriggerRouter-350M (LoRA adapter) | ~75 KB | negligible |
| Generation (on demand) | LFM2.5-2.6B-Q4_K_M | ~1.6 GB | ~1.5–3.5 s |
| Structured notes | LFM2.5-350M-Extract | ~350 MB | — |
| Corpus distillation | LFM2.5-2.6B, prompted (offline) | ~1.6 GB | — |
| Speaker diarization | ECAPA-TDNN (speechbrain, optional) | ~100 MB | — |

**Why six small models instead of one big one?** Because these are six different tasks
with six different output shapes. Question detection produces one label. Rhetorical
filtering produces yes-no flags. Evidence grounding produces per-token spans. Retrieval
produces a ranking. Only one step — writing a mode-aware prompt — is actually generative.
Using a decoder for all of them means paying generation cost for classification work
and getting classification quality from a model that was never trained for it.

Six of those tasks now share **one 350M encoder backbone plus tiny task heads** of about
75 KB each. And literal watch words stay plain Python, because anything checkable in O(1)
does not need a model at all.

Full reasoning, the output-shape audit, and the measured numbers behind each choice:
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick start

Full setup, including model downloads and runner binaries:
**[docs/DEVELOPER-GUIDE.md](docs/DEVELOPER-GUIDE.md)**.

```bash
git clone <repo-url> && cd meeting-prompter

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export MODELS_DIR=~/Projects/_models      # shared model registry
./tools/audio-tap/build.sh                # per-app audio capture

cd app && npm install
npm run tauri dev                         # starts backend + UI
```

CLI mode, no Tauri:

```bash
python coach.py --mic                     # microphone only
python coach.py                           # live meeting (BlackHole)
python coach.py --test audio.wav          # audio file
python coach.py --list-devices
```

---

## Capturing a call

The app listens to two streams at once, which is what makes speaker attribution
deterministic rather than guessed:

- **Your microphone** → labeled "You"
- **System audio** → labeled "Others"

System audio has two backends. **Per-app capture** (ScreenCaptureKit, via a small Swift
CLI) grabs audio from one application by PID — it needs macOS 13+ and Screen Recording
permission. **BlackHole** is the fallback loopback device:

```bash
brew install blackhole-2ch
# Audio MIDI Setup → create a Multi-Output Device (BlackHole 2ch + your speakers)
# Set your meeting app's speaker to that Multi-Output Device
```

Optional neural diarization then splits "Others" into individual speakers. Click any
label to rename it; the rename applies retroactively and persists.

**On conference rooms, honestly:** many people sharing one far-field microphone cannot
be separated in software. Rather than inventing names, the app degrades to a single
`Others (room)` label or a clearly flagged best guess.

---

## Adding your documents

Drop `.md`, `.pdf`, or `.txt` files into your documents directory (`paths.docs_dir` in
`config.yaml`). They're indexed on startup into SQLite with FTS5 and vector embeddings.

```bash
cp your-product-docs.pdf context/
python coach.py --mic          # new files indexed automatically
```

Force a rebuild by deleting `data/rag.db` and restarting, or `POST /session/reindex`.
Then run **Prepare corpus** from Meeting Setup to distill and score them.

Notion works as a source too: pages and databases listed in `notion.rag_source_page_ids`
are fetched, converted to markdown, and indexed alongside local files. Disabled by
default; needs `NOTION_API_TOKEN`.

---

## After the meeting

Stop recording and the session stays alive so your notes and transcript remain
available. Meeting notes — summary, decisions, action items attributed to speakers —
are generated locally by the LLM and open in an editor first.

**Export is consent-gated.** Nothing leaves the device until you press export. The
transcript is a separate opt-in from the notes.

---

## Configuration

Every threshold, weight, and model choice lives in `config.yaml` — no magic numbers in
code, no hardcoded model paths. The ones you'll actually touch:

| Setting | Default | Effect |
|---|---|---|
| `paths.docs_dir` | `context` | where your documents live |
| `triggers.watch_words` | `[]` | terms that fire HEADS UP (usually per-meeting) |
| `triggers.gating.enabled` | `true` | D-02 quiet-by-default; `false` = always-on push |
| `triggers.gating.always_on` | `["alert"]` | trigger types that fire without arming |
| `triggers.gating.max_listen_seconds` | `0` | `0` = armed window stays open until ⌘L again |
| `triggers.retrieval_first` | `true` | live cards are retrieved spans, no LLM |
| `triggers.f503_router_enabled` | `true` | learned trigger router; `false` = pure heuristics |
| `detection.rag_confidence_minimum` | `0.35` | below this, stay silent |
| `buffer.turn_pause` | `2.0` | seconds of silence that end a speech turn |
| `diarization.enabled` | `false` | Tier-2 per-speaker separation |
| `notion.enabled` | `false` | export + ingest |

Per-meeting context — watch words, agenda, participants — goes in a
`meeting_context.yaml` (`python coach.py --create-context` writes a template) or through
the Meeting Setup dialog.

---

## Requirements

- macOS 13+ on Apple Silicon (M1–M4) — Metal inference and ScreenCaptureKit; no CUDA path
- 16 GB RAM or more
- Python 3.10+, Node.js 18+, Rust (for the Tauri shell)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Remote speakers never transcribed | Grant Screen Recording to the terminal you launch from, then `./runners/macos-arm64/audio-tap --check-permission` |
| No audio at all | `python coach.py --list-devices`; check the device in Meeting Setup |
| Cards never appear | Check the corpus actually covers the topic — run the readiness score |
| Cards at nonsense moments | Set `triggers.f503_router_enabled: false` and restart |
| Wrong answers | Delete `data/rag.db` and restart to rebuild the index |
| Garbled transcript | ASR hallucinates on background noise; the filter catches common patterns |
| Slow first query | First embed loads the model; it's pre-warmed on session start |
| Tauri build errors | `cd app && npm install`; confirm the Rust toolchain |

More, including the dual-stream audio walkthrough:
[docs/DEVELOPER-GUIDE.md](docs/DEVELOPER-GUIDE.md#8-debugging).

---

## Testing

```bash
pytest                              # ~800 Python tests, no models required
pytest tests/eval/ -m slow -v       # retrieval eval: Hit@1, Hit@3, MRR
cd app && npx tsc --noEmit && npm test
```

Retrieval benchmark on 21 queries: **Hit@1 94.4%, MRR 0.972.**

---

## License

MIT
