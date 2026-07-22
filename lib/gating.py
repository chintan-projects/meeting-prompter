"""Listen gating — default quiet, user opens the tap (D-02).

The always-on push model produced a stream of irrelevant prompts: a trigger
firing correctly is not the same as the user wanting a card for it. Most
correctly-detected questions in a meeting do not need an answer on screen.
That is a permission problem, not a classification problem — a perfect trigger
router still interrupts on every true positive.

So the default is **quiet**. Automatic cards are suppressed unless the user has
armed the listen window (temporal). Two paths bypass the gate entirely because
the user asked for them by hand:

- **select-to-answer** (spatial) — the user selects a transcript span and asks
  for an answer to *that*.
- **on-demand generation** — the ✨ button on an existing card.

One trigger type stays always-on: ALERT (literal watch words). It is the only
channel the user pre-authorized by naming the terms themselves, and it exists
precisely for things you must know without asking.

The gate is a policy object, not a scheduler: it answers "does this pass right
now" and owns no threads. Thread-safe because the two capture pipelines call it
from their own threads while the API arms it from the event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

#: Trigger types that fire regardless of the gate. Watch-word alerts are
#: pre-authorized by the user naming the terms; nothing else is.
DEFAULT_ALWAYS_ON: tuple[str, ...] = ("alert",)


class ListenGate:
    """Decides whether an automatically-fired trigger may produce a card.

    Args:
        enabled: master switch. ``False`` restores the always-on behaviour
            (every trigger passes) — the pre-D-02 product.
        always_on: trigger type values that bypass the gate even when disarmed.
        max_listen_seconds: safety cap. ``0`` (default) means the armed window
            stays open until explicitly disarmed, which is the chosen product
            behaviour; a positive value auto-disarms after that many seconds so
            a forgotten window cannot silently reintroduce prompt spam.
    """

    def __init__(
        self,
        enabled: bool = True,
        always_on: Sequence[str] = DEFAULT_ALWAYS_ON,
        max_listen_seconds: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.always_on = frozenset(t.lower() for t in always_on)
        self.max_listen_seconds = max(0.0, max_listen_seconds)
        self._armed_at: Optional[float] = None
        self._lock = threading.Lock()

    # --- state ---------------------------------------------------------------
    def _expired(self, armed_at: float, now: float) -> bool:
        """True when a capped window has run out (uncapped windows never expire)."""
        return self.max_listen_seconds > 0 and (now - armed_at) >= self.max_listen_seconds

    def is_armed(self, now: Optional[float] = None) -> bool:
        """Whether the listen window is currently open (honours the safety cap)."""
        now = time.time() if now is None else now
        with self._lock:
            if self._armed_at is None:
                return False
            if self._expired(self._armed_at, now):
                # Lazy expiry: no timer thread to leak, and the state is correct
                # for whoever asks first.
                self._armed_at = None
                logger.info("listen window auto-disarmed after %.0fs", self.max_listen_seconds)
                return False
            return True

    def state(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Serialisable state for the API and the `listen_state` WS message."""
        now = time.time() if now is None else now
        armed = self.is_armed(now)
        with self._lock:
            armed_at = self._armed_at
        expires_at = (
            armed_at + self.max_listen_seconds
            if armed_at is not None and self.max_listen_seconds > 0
            else None
        )
        return {
            "armed": armed,
            "enabled": self.enabled,
            "since": armed_at,
            "expires_at": expires_at,
            "always_on": sorted(self.always_on),
        }

    # --- control -------------------------------------------------------------
    def arm(self, now: Optional[float] = None) -> bool:
        """Open the listen window. Re-arming an open window restarts its clock."""
        now = time.time() if now is None else now
        with self._lock:
            self._armed_at = now
        logger.info("listen window ARMED")
        return True

    def disarm(self) -> bool:
        """Close the listen window. Idempotent."""
        with self._lock:
            was = self._armed_at is not None
            self._armed_at = None
        if was:
            logger.info("listen window disarmed")
        return False

    def toggle(self, now: Optional[float] = None) -> bool:
        """Flip the window and return the resulting armed state."""
        return self.disarm() if self.is_armed(now) else self.arm(now)

    # --- policy --------------------------------------------------------------
    def allows(self, trigger_type: str, now: Optional[float] = None) -> bool:
        """Whether a trigger of this type may produce a card right now.

        Explicit user requests (select-to-answer, on-demand generation) do not
        call this — they bypass the gate by construction, since the user asking
        for an answer *is* the permission.
        """
        if not self.enabled:
            return True
        if trigger_type.lower() in self.always_on:
            return True
        return self.is_armed(now)
