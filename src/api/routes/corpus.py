"""Corpus preparation routes — the "Prepare corpus" flow (F-703/F-704).

The wizard's backend: add sources → distill (on-device, background job with
progress) → readiness score + gap list → activate the distilled corpus as the
live retrieval source. Everything is local (ADR-001); readiness builds a
throwaway index so it never disturbs the live rag.db.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from lib.config import load_config
from lib.corpus.active import get_active_dir, set_active_dir
from lib.corpus.incremental import MANIFEST_NAME, SUPPORTED_SUFFIXES, distill_dir
from lib.corpus.readiness import DEFAULT_TOP_K, readiness
from lib.paths import get_docs_dir

router = APIRouter(prefix="/corpus", tags=["corpus"])

DISTILLED_DIR = Path("data/distilled")


def _sources_dir() -> Path:
    return get_docs_dir(load_config().paths.docs_dir)


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
    merged: bool = False  # best answer is a multi-unit (top-2) merged card


class ReadinessResponse(BaseModel):
    score_pct: int
    questions: int
    good: int
    partial: int
    gap: int
    gaps: List[GapRow]
    rows: List[GapRow]


def _distilled_state() -> Dict[str, Any]:
    """Describe the distilled corpus, however it was produced.

    Two paths write it: the wizard (`distill_dir`, one manifest for the whole
    directory) and the lab CLI (`scripts.lab.distiller`, a per-file `.meta.json`
    sidecar). Reading only the manifest made the wizard report "no distilled
    corpus" while one sat on disk, which silently blocks its distill/readiness/
    activate steps. Fall back to the sidecars, then to the files themselves.
    """
    state: Dict[str, Any] = {"dir": str(DISTILLED_DIR), "exists": False}
    if not DISTILLED_DIR.is_dir():
        return state
    manifest_path = DISTILLED_DIR / MANIFEST_NAME
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            docs = manifest.get("docs", {})
            return {
                "dir": str(DISTILLED_DIR),
                "exists": True,
                "backend": manifest.get("backend", ""),
                "mode": manifest.get("mode", ""),
                "docs": len(docs),
                "units": sum(int(d.get("units") or 0) for d in docs.values()),
                "source": "manifest",
            }
        except (json.JSONDecodeError, OSError):
            state["error"] = "unreadable manifest"
    corpora = sorted(DISTILLED_DIR.glob("*.distilled.md"))
    if not corpora:
        return state
    backends, units = set(), 0
    for f in corpora:
        meta = f.with_name(f".{f.name}.meta.json")
        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                backends.add(str(m.get("backend", "")))
                units += int(m.get("units") or 0)
            except (json.JSONDecodeError, OSError):
                continue
    return {
        "dir": str(DISTILLED_DIR),
        "exists": True,
        "backend": "+".join(sorted(b for b in backends if b)) or "unknown",
        "mode": "",
        "docs": len(corpora),
        "units": units,
        "source": "sidecar" if backends else "files",
    }


# --- status + sources -------------------------------------------------------
@router.get("/status")
def corpus_status() -> Dict[str, Any]:
    """Everything the wizard needs to render: sources, distilled state, active dir."""
    docs_dir = _sources_dir()
    sources = (
        [
            {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in sorted(docs_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if docs_dir.is_dir()
        else []
    )
    distilled = _distilled_state()
    active = get_active_dir()
    return {
        "docs_dir": str(docs_dir),
        "sources": sources,
        "distilled": distilled,
        "active_dir": str(active) if active else None,
        "distilled_active": active is not None and active.resolve() == DISTILLED_DIR.resolve(),
    }


@router.post("/sources/upload")
async def upload_source(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Add a source document (.md/.txt/.pdf) to the corpus sources directory."""
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f"unsupported file type (allowed: {SUPPORTED_SUFFIXES})"
        )
    docs_dir = _sources_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    (docs_dir / name).write_bytes(content)
    return {"status": "added", "name": name, "size_kb": round(len(content) / 1024, 1)}


# --- distill (background job) ------------------------------------------------
class DistillRequest(BaseModel):
    backend: str = "local"  # shipped default (ADR-001); heuristic = no-model floor
    mode: str = "consolidated"
    force: bool = False


_job_lock = threading.Lock()
_job: Dict[str, Any] = {"state": "idle"}


def _run_distill_job(backend: str, mode: str, force: bool) -> None:
    def cb(name: str, i: int, total: int) -> None:
        with _job_lock:
            _job["progress"] = {"current": name, "done": i - 1, "total": total}

    try:
        result = distill_dir(
            _sources_dir(), DISTILLED_DIR, backend=backend, mode=mode, force=force, progress_cb=cb
        )
        with _job_lock:
            _job.update({"state": "done", "result": result})
    except Exception as e:  # noqa: BLE001 — job boundary: surface any failure to the UI
        with _job_lock:
            _job.update({"state": "error", "error": str(e)})


@router.post("/distill")
def start_distill(req: DistillRequest) -> Dict[str, Any]:
    """Start distilling the sources dir into data/distilled (409 if running).

    backend="cloud" is deliberately rejected here: the product path is local-only
    (ADR-001); cloud distillation stays an offline dev-lab operation.
    """
    if req.backend not in ("local", "heuristic"):
        raise HTTPException(status_code=400, detail="backend must be 'local' or 'heuristic'")
    if not _sources_dir().is_dir():
        raise HTTPException(status_code=400, detail=f"sources dir not found: {_sources_dir()}")
    with _job_lock:
        if _job.get("state") == "running":
            raise HTTPException(status_code=409, detail="a distill job is already running")
        _job.clear()
        _job.update({"state": "running", "backend": req.backend, "mode": req.mode})
    threading.Thread(
        target=_run_distill_job, args=(req.backend, req.mode, req.force), daemon=True
    ).start()
    return {"status": "started", "backend": req.backend, "mode": req.mode}


@router.get("/distill/status")
def distill_status() -> Dict[str, Any]:
    """Poll the background distill job (state: idle|running|done|error)."""
    with _job_lock:
        return dict(_job)


# --- activate ----------------------------------------------------------------
class ActivateRequest(BaseModel):
    corpus_dir: str = str(DISTILLED_DIR)


@router.post("/activate")
def activate_corpus(req: ActivateRequest) -> Dict[str, Any]:
    """Make a prepared corpus the live retrieval source (next session start)."""
    target = Path(req.corpus_dir)
    if not target.is_dir() or not any(
        p.suffix.lower() in SUPPORTED_SUFFIXES for p in target.iterdir() if p.is_file()
    ):
        raise HTTPException(status_code=400, detail=f"no corpus documents in {target}")
    set_active_dir(target)
    return {"status": "active", "corpus_dir": str(target), "applies": "next session start"}


@router.delete("/activate")
def deactivate_corpus() -> Dict[str, Any]:
    """Revert the live retrieval source to the configured docs dir."""
    set_active_dir(None)
    return {"status": "deactivated"}


# --- readiness ---------------------------------------------------------------
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
        gaps=[GapRow.model_validate(row) for row in result["gaps"]],
        rows=[GapRow.model_validate(row) for row in result["rows"]],
    )
