"""LLM-as-judge for corpus fitness — cloud Claude grades borrowability.

OFFLINE EVAL ONLY. The meeting-prompter product stays 100% local; this harness
sends the *question + a retrieved chunk* to the Anthropic API to auto-rate whether
the chunk is a borrowable answer, so the coverage loop scales past hand-rating.

The judge is a proxy — trust it only after checking its agreement with the human
ratings already collected (see LabEngine.calibration). Auth is read from the
environment by the standard client (ANTHROPIC_API_KEY, or an `ant auth login`
profile); this module never handles the key. Enable with a key in the env:

    ANTHROPIC_API_KEY=sk-ant-... uvicorn scripts.lab.server:app --port 8555
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

# Opus 4.8 — the strongest judge = the most valid eval. Override with LAB_JUDGE_MODEL.
JUDGE_MODEL = os.environ.get("LAB_JUDGE_MODEL", "claude-opus-4-8")

JUDGE_SYSTEM = """You grade whether a retrieved document chunk is a BORROWABLE answer \
to a meeting question — text a speaker could read aloud or lightly paraphrase on the \
spot to answer the question.

Choose exactly one rating:
- good: directly and sufficiently answers the question; the speaker could borrow it as-is and a listener would be satisfied.
- partial: relevant and correct but incomplete — answers only part of the question, or would need another chunk to be a full answer.
- wrong: on-topic but does not answer THIS question, or would mislead if borrowed as the answer.
- noise: off-topic, or not answer-shaped (a bare heading, a table/code fragment, boilerplate).

Judge ONLY whether the chunk answers the question. Do not use outside knowledge, and \
do not reward fluent prose that misses the question. Be strict about 'good'. Give a \
one-sentence reason and a confidence in [0,1]."""

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rating": {"type": "string", "enum": ["good", "partial", "wrong", "noise"]},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["rating", "reason", "confidence"],
    "additionalProperties": False,
}

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile from env
    return _client


def judge(span: str, chunk_text: str, max_tokens: int = 400) -> dict[str, Any]:
    """Return {rating, reason, confidence} for one chunk, or {error} on failure."""
    hint = credential_hint()
    if hint is not None:
        return {"error": hint}
    import anthropic

    user = (
        f"QUESTION:\n{span}\n\nRETRIEVED CHUNK:\n{chunk_text[:6000]}\n\n"
        "Rate whether the chunk is a borrowable answer to the question."
    )
    try:
        resp = _get_client().messages.create(
            model=JUDGE_MODEL,
            max_tokens=max_tokens,
            system=JUDGE_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError:
        return {"error": "no Anthropic credential — set ANTHROPIC_API_KEY (or ant auth login)"}
    except anthropic.APIStatusError as e:  # noqa: BLE001 — surface API errors to the UI
        return {"error": f"API error {e.status_code}: {getattr(e, 'message', str(e))}"}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}

    try:
        text = next(b.text for b in resp.content if b.type == "text")
        data: dict[str, Any] = json.loads(text)
    except Exception as e:  # noqa: BLE001
        return {"error": f"bad judge output: {e!r}"}
    data["confidence"] = round(float(data.get("confidence", 0.0)), 3)
    return data


def credential_hint() -> Optional[str]:
    """Best-effort readiness check for the UI. None = looks ready."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "anthropic SDK not installed"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "no ANTHROPIC_API_KEY in env (or run `ant auth login`)"
    return None
