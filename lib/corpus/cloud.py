"""Cloud (Anthropic) access for the corpus pipeline — OFFLINE / OPT-IN ONLY.

Per ADR-001 the shipped distiller runs on a local model; cloud Claude is used
offline during development (validation, training-data generation) and may later
be an optional consent-gated quality toggle. This module is the single place the
corpus package touches the network, so the egress surface stays auditable.

Auth is read from the environment by the standard client (ANTHROPIC_API_KEY or
an `ant auth login` profile); this module never handles the key itself.
"""

from __future__ import annotations

import os
from typing import Any, Optional

#: Model used for cloud distillation. Overridable for cheaper experimentation;
#: falls back to the lab judge's override so lab and library stay consistent.
CLOUD_MODEL: str = os.environ.get(
    "CORPUS_CLOUD_MODEL", os.environ.get("LAB_JUDGE_MODEL", "claude-opus-4-8")
)

_client: Any = None


def get_client() -> Any:
    """Return a lazily constructed Anthropic client (reads auth from the env)."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def credential_hint() -> Optional[str]:
    """Best-effort readiness check. None = a cloud call looks possible."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "anthropic SDK not installed"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "no ANTHROPIC_API_KEY in env (or run `ant auth login`)"
    return None
