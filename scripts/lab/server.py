"""Model & retrieval comparison lab — web server.

A standalone, visual harness (detached from the Tauri app) for making the
answer-model call by eye. Serves a single page that, for a selected transcript
span, shows every retrieval stage and all three answer candidates side-by-side.

Run:
    uvicorn scripts.lab.server:app --port 8555
    # then open http://localhost:8555

To enable the 350M-Extract panel, point it at an interpreter with transformers>=5:
    LAB_EXTRACT_PYTHON=/path/to/overlay-venv/bin/python uvicorn scripts.lab.server:app --port 8555

This surfaces options; it does not decide. Feeds D-03 / E-01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from scripts.lab.pipeline import SAMPLE_SPANS, LabEngine

app = FastAPI(title="meeting-prompter model lab")

_PAGE = Path(__file__).with_name("page.html")
_engine: Optional[LabEngine] = None


def engine() -> LabEngine:
    """Lazy singleton — warm the RAG engine on first request, not import."""
    global _engine
    if _engine is None:
        _engine = LabEngine()
    return _engine


class AnalyzeRequest(BaseModel):
    span: str
    max_tokens: Optional[int] = None


class RateRequest(BaseModel):
    span: str
    chunk_id: int
    doc: str
    rating: str  # good | partial | wrong | noise


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE.read_text(encoding="utf-8")


@app.get("/samples")
def samples() -> dict[str, Any]:
    return {"samples": SAMPLE_SPANS}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> JSONResponse:
    span = (req.span or "").strip()
    if not span:
        return JSONResponse({"error": "empty span"}, status_code=400)
    result = engine().analyze(span, max_tokens=req.max_tokens)
    return JSONResponse(result)


@app.post("/rate")
def rate(req: RateRequest) -> JSONResponse:
    if req.rating not in ("good", "partial", "wrong", "noise"):
        return JSONResponse({"error": f"bad rating: {req.rating}"}, status_code=400)
    eng = engine()
    eng.record_rating(req.span, req.chunk_id, req.doc, req.rating)
    return JSONResponse(eng.coverage())


@app.get("/coverage")
def coverage() -> JSONResponse:
    return JSONResponse(engine().coverage())
