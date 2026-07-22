# WS-14 — live call run sheet

The product has never run in a real meeting. This is the checklist to do that, and the
list of things to watch. Preflight was run 2026-07-22; everything below is verified except
the one blocker in step 1.

## Verified in preflight (you don't need to check these)

| Component | State |
|---|---|
| ASR model + runner | `LFM2.5-Audio-1.5B-Q4_0` + `llama-liquid-audio-cli` resolve |
| Generation model | `LFM2.5-2.6B-Q4_K_M.gguf` present |
| Embedding model | `LFM2.5-Embedding-350M` present |
| `audio-tap` binary | present, executable |
| Audio devices | BlackHole 2ch, MacBook Pro Microphone, ZoomAudioDevice |
| Backend | starts on 127.0.0.1:8420 exactly as the Tauri shell spawns it |
| WebSockets | `/ws/transcript` and `/ws/prompts` both connect |
| Rust shell | `cargo check` clean |
| Frontend | `tsc --noEmit` clean, 16 tests pass |
| **Live retrieval path** | **works — 174ms cold, then 51ms / 16ms; off-topic speech correctly returns no card** |

## 1. Grant Screen Recording — THE BLOCKER

Per-app capture needs it. Without it the app **silently degrades to mic-only**, so remote
speakers are never transcribed — which is the entire point of this test (BUG-005).

macOS grants this per *responsible binary*, so grant it to the app you actually launch
from. Preflight reported `denied`, but that was under a different parent process, so
**check it yourself from the same terminal you'll use**:

```bash
./runners/macos-arm64/audio-tap --check-permission
```

If denied: System Settings → Privacy & Security → Screen & System Audio Recording → enable
for your terminal (dev mode) or the packaged app. Then re-run the check. In the app, the
Meeting Setup dialog also surfaces this and offers an explicit "Start mic-only instead".

## 2. Start it

```bash
cd app && npm run tauri dev
```

The Rust shell spawns the Python backend itself — don't start uvicorn separately or port
8420 will conflict.

## 3. Set up the meeting

In Meeting Setup: pick your meeting app (Zoom/Teams/browser) under **Meeting App**, confirm
the **Microphone**, and set **Watch Words** to something you'll actually say (that's the
only always-on prompt channel).

Corpus state: live retrieval currently runs on the **original** playbook at
`paths.docs_dir`, not the distilled corpus. To test the distilled one instead, open
"Prepare corpus…" → "Use distilled corpus for live meetings", then restart the session
(activation applies at next session start).

## 4. What to watch, in priority order

1. **Speaker attribution** — is "You" vs "Others" right? This is the dual-stream guarantee.
2. **Turn segmentation** — do turns break at sensible pauses, or fragment mid-sentence?
3. **Trigger timing** — do cards appear when you'd want them, or at random moments? This is
   the real product question (D-02), and it matters more than answer quality.
4. **Card usefulness** — is the borrowable span something you'd actually read aloud?
   Expand-to-source: does the provenance look right?
5. **Latency** — cards should feel instant. Preflight says 16–174ms; confirm under real load.
6. **Silence** — does it stay quiet during ordinary chat? A prompt-spammy session is a
   failed session even if every card is correct.

## 5. Known-unvalidated switches

- `triggers.f503_router_enabled: true` — the encoder trigger router is **ON and unproven
  live**. If triggers fire at nonsense moments, set it to `false` and restart: that
  isolates "the router is wrong" from "the product is wrong."
- `triggers.retrieval_first: true` — live cards are retrieved spans, no LLM. The ✨ generate
  button on a card is the on-demand LLM path.
- `diarization.enabled: false` — Tier 2 per-speaker separation is off. You get You/Others,
  not individual remote speakers.

## 6. Afterwards

Capture: the transcript export, any screenshot of a bad/mistimed card, and the backend log.
Then the decisions that have been waiting on this call:

- Promote or revert F-503 (`f503_router_enabled`)
- Promote F-507 (Extract) / F-508 (persistent ASR)
- D-02 — is user-gated interaction (quiet by default, select-to-answer) the right model, or
  is always-on acceptable?
- Merge `liquid-rearch` (58 commits, unreviewed) once the call proves the path
