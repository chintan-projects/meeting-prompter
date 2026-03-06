"""Alert trigger — fires when watch words are detected in speech.

Watch words are configured per-meeting (competitor names, pricing terms,
compliance keywords, etc.) via meeting_context.yaml or config.yaml.
"""
import time
from typing import Dict, List, Optional

from .types import Trigger, TriggerType


class AlertTrigger:
    """Scans transcript for configured watch words.

    Highest priority trigger (priority=1). Fires immediately when a watch
    word is found. Includes cooldown to avoid re-alerting on the same word
    within a short window.
    """

    def __init__(
        self,
        watch_words: Optional[List[str]] = None,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.watch_words = [w.lower() for w in (watch_words or [])]
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: Dict[str, float] = {}

    def set_watch_words(self, words: List[str]) -> None:
        """Update watch words (e.g. when loading meeting context)."""
        self.watch_words = [w.lower() for w in words]
        self._last_fired.clear()

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]:
        """Check text for watch words. Returns Trigger on match."""
        if not self.watch_words:
            return None

        text_lower = text.lower()
        now = time.time()

        for word in self.watch_words:
            if word not in text_lower:
                continue

            # Cooldown check
            last = self._last_fired.get(word, 0.0)
            if now - last < self.cooldown_seconds:
                continue

            self._last_fired[word] = now
            return Trigger(
                type=TriggerType.ALERT,
                text=text,
                confidence=1.0,
                source_context=conversation_context,
                metadata={"watch_word": word},
                timestamp=now,
            )

        return None
