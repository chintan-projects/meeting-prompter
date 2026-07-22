"""Corpus preparation routes — readiness scoring (F-703).

POST /corpus/readiness runs a question set against a corpus directory and
returns the fit-for-purpose score + gap list (see lib/corpus/readiness.py).
Scoring is local (heuristic rater, ADR-001) and offline — it builds a throwaway
index for the requested directory, so it never disturbs the live rag.db.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lib.config import load_config
from lib.corpus.readiness import DEFAULT_TOP_K, readiness
from lib.paths import get_docs_dir

router = APIRouter(prefix="/corpus", tags=["corpus"])


class ReadinessRequest(BaseModel):
    questions: List[str] = Field(min_length=1)
    corpus_dir: str = ""  # default: the configured docs dir
    top_k: int = DEFAULT_TOP_K


class GapRow(BaseModel):
    question: str
    best: str
    reason: str
    doc: str
    heading: str


class ReadinessResponse(BaseModel):
    score_pct: int
    questions: int
    good: int
    partial: int
    gap: int
    gaps: List[GapRow]
    rows: List[GapRow]


@router.post("/readiness")
def score_readiness(req: ReadinessRequest) -> ReadinessResponse:
    """Score a corpus directory against a question set (sync — offline prep step)."""
    questions = [q.strip() for q in req.questions if q.strip()]
    if not questions:
        raise HTTPException(status_code=400, detail="questions must be non-empty")
    docs_dir = (
        Path(req.corpus_dir) if req.corpus_dir else get_docs_dir(load_config().paths.docs_dir)
    )
    if not docs_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"corpus dir not found: {docs_dir}")
    result = readiness(docs_dir, questions, top_k=req.top_k)
    return ReadinessResponse(
        score_pct=result["score_pct"],
        questions=result["questions"],
        good=result["good"],
        partial=result["partial"],
        gap=result["gap"],
        gaps=[GapRow(**row) for row in result["gaps"]],
        rows=[GapRow(**row) for row in result["rows"]],
    )
