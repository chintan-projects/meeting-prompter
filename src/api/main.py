"""FastAPI server for the Meeting Prompter Tauri app.

Exposes REST and WebSocket endpoints for:
- Session management (start/stop/status)
- Meeting context loading
- Real-time transcript streaming (WebSocket)
- Real-time prompt/trigger streaming (WebSocket)
- Meeting notes editing and export

Run with:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8420
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

# Ensure project root is on path for lib imports
# In packaged mode, MEETING_PROMPTER_ROOT points to source tree.
# In dev mode, walk up from src/api/ to project root.
_project_root = os.environ.get("MEETING_PROMPTER_ROOT") or str(Path(__file__).parent.parent.parent)
sys.path.insert(0, _project_root)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.api.routes import context, corpus, notes, notion, prompts, session, transcript  # noqa: E402

app = FastAPI(
    title="Meeting Prompter",
    description="Real-time meeting intelligence API",
    version="2.0.0",
)

# Allow Tauri webview and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost:1420", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(session.router)
app.include_router(context.router)
app.include_router(corpus.router)
app.include_router(transcript.router)
app.include_router(prompts.router)
app.include_router(notes.router)
app.include_router(notion.router)


@app.get("/health")
async def health() -> dict:
    """Health check for Tauri backend readiness."""
    return {"status": "ok"}
