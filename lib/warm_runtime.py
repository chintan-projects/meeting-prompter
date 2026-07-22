"""Persistent warm-model runtime (F-508).

Real-time meeting intelligence keeps several models resident at once (ASR,
encoder, embedder, instruct). Loading them per-call — or letting different call
sites each construct their own copy — burns latency and memory. ``WarmModelRuntime``
is the single owner of the load-once, stay-warm models: it lazily constructs each
one, hands the SAME instance to every consumer, reports what is resident, and
tears everything down together (every setup has a matching teardown).

Tonight this centralizes the encoder backbone (previously constructed ad-hoc for
the probe head) and provides a registry + status/teardown surface for the
embedder and instruct generator the orchestrator already keeps warm. Replacing
the ASR subprocess-per-call with a persistent llama.cpp server is the remaining
step and needs a live verification run — see STATUS-overnight.md.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from lib.intelligence.encoder import EncoderBackbone

logger = logging.getLogger(__name__)


class WarmModelRuntime:
    """Owns warm, load-once model singletons and a small registry."""

    def __init__(self) -> None:
        self._encoder: Optional["EncoderBackbone"] = None
        self._registry: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ─── Encoder backbone (owned singleton) ──────────────────────────────

    def encoder(self) -> "EncoderBackbone":
        """Return the shared, warm encoder backbone (constructed once)."""
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    from lib.intelligence.encoder import EncoderBackbone

                    self._encoder = EncoderBackbone()
        return self._encoder

    # ─── Registry for externally-constructed warm models ─────────────────

    def register(self, name: str, model: Any) -> None:
        """Register a warm model (e.g. 'embedder', 'instruct') for status/teardown."""
        with self._lock:
            self._registry[name] = model
        logger.info("Warm runtime: registered %s", name)

    def get(self, name: str) -> Optional[Any]:
        return self._registry.get(name)

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def warm(self, encoder: bool = True) -> None:
        """Eagerly load the models that should be resident before a session.

        Only the encoder is force-warmed here (it is cheap — ~14 ms/turn once
        loaded); registered models are assumed already warm by their owners.
        """
        if encoder:
            try:
                self.encoder()._load()
            except Exception as exc:
                logger.warning("Encoder warm-up failed (will lazy-load): %s", exc)

    def status(self) -> Dict[str, bool]:
        """Report which models are currently resident.

        For the encoder, 'resident' means the weights are actually loaded (not
        merely that the object exists). Registered models report presence.
        """
        resident: Dict[str, bool] = {}
        enc = self._encoder
        resident["encoder"] = bool(enc is not None and getattr(enc, "_backbone", None) is not None)
        for name, model in self._registry.items():
            resident[name] = model is not None
        return resident

    def teardown(self) -> None:
        """Release references so models can be reclaimed. Matches warm()."""
        with self._lock:
            self._encoder = None
            self._registry.clear()
        logger.info("Warm runtime torn down")
