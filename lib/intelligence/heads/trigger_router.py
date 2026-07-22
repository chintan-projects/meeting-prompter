"""F-503 trigger-router head — forge-LoRA encoder classifier.

The v2 upgrade over the F-510 frozen probe: a LoRA-trained sequence classifier on
the bidirectional LFM2.5-Encoder-350M (mean-pool, CE), routing one turn to a
single trigger shape ∈ {question, alert, topic, followup, none}. Trained via forge
on a shape-first synthetic corpus (content held maximally diverse so only the
pragmatic SHAPE predicts the label) — see forge/f503-trigger-router/.

Gated honestly. On a frozen out-of-domain held-out AND on real meeting transcript
turns it beats the F-510 probe and the heuristic on macro-F1; its one weakness is
`question` (the class the heuristic owns), which the HYBRID here fixes: the router
is primary, and a `question_rescue` evaluator reclaims a question the router filed
as none/topic. On the real held-out this lifts question 0.80→0.88 (matching the
heuristic) while keeping alert 0.92 / topic 0.88 — macro-F1 0.846.

Wired behind the ``Head`` interface, config-gated (``f503_router_enabled``) and
default OFF: proven offline, but promotion to the live default is gated on a live
call (WS-14). If the adapter or the ML deps are absent it degrades to inert
(``evaluate`` returns ``None``) so the heuristic heads carry the load unchanged.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger, TriggerType

logger = logging.getLogger(__name__)

_ENCODER_DIRNAME = "LFM2.5-Encoder-350M"
_ADAPTER_DIRNAME = "LFM2.5-TriggerRouter-350M"
_MAX_TOKENS = 128

# Predicted label → emitted trigger type. "none" emits nothing.
_LABEL_TO_TYPE: Dict[str, Optional[TriggerType]] = {
    "question": TriggerType.QUESTION,
    "alert": TriggerType.ALERT,
    "topic": TriggerType.TOPIC_MATCH,
    "followup": TriggerType.FOLLOW_UP,
    "none": None,
}
# Labels the question-rescue may override to `question` (never overrides a
# confident alert/followup — only the abstain/assertion classes).
_RESCUABLE = {"none", "topic"}


class _QuestionEvaluator(Protocol):
    """The heuristic question surface used for the hybrid rescue."""

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]: ...


class TriggerRouterHead:
    """Route-first shape classifier (F-503) with a heuristic question-rescue."""

    #: Nominal type for logging; a router emits whichever type it predicts.
    trigger_type = TriggerType.QUESTION

    def __init__(
        self,
        models_dir: Path,
        question_rescue: Optional[_QuestionEvaluator] = None,
        enabled: bool = False,
        min_confidence: float = 0.0,
        encoder_dirname: str = _ENCODER_DIRNAME,
        adapter_dirname: str = _ADAPTER_DIRNAME,
        device: Optional[str] = None,
        name: str = "f503_router",
    ) -> None:
        self.name = name
        self._models_dir = models_dir
        self._question_rescue = question_rescue
        self._enabled = enabled
        self._min_confidence = min_confidence
        self._encoder_path = models_dir / encoder_dirname
        self._adapter_path = models_dir / adapter_dirname
        self._device = device
        # Dynamically-typed HF/torch objects (untyped third-party); Any is intentional.
        self._tok: Any = None
        self._model: Any = None
        self._id2label: Dict[int, str] = {}
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def is_available(self) -> bool:
        """True if both the base encoder and the adapter are present on disk."""
        return self._encoder_path.exists() and self._adapter_path.exists()

    def can_load(self) -> bool:
        """True if the weights are present AND torch/transformers/peft import.

        Checked without loading the 350M model, so the engine can fall back to the
        heuristic heads at construction rather than wiring a router that would
        silently no-op at inference (e.g. peft missing from the runtime venv).
        """
        if not self.is_available():
            return False
        import importlib.util

        return all(
            importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft")
        )

    def _load(self) -> None:
        """Lazy-load tokenizer + adapted classifier once; thread-safe, warm after.

        Heavy deps (torch / transformers / peft) import here, never at module
        load, so the test suite and the heuristic path run without them.
        """
        if self._model is not None or self._load_failed:
            return
        with self._lock:
            if self._model is not None or self._load_failed:
                return
            try:
                import json

                import torch
                import torch.nn as nn
                from peft import PeftModel
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                if not self.is_available():
                    raise FileNotFoundError(
                        f"F-503 router weights missing: {self._encoder_path} / {self._adapter_path}"
                    )
                schema = json.loads((self._adapter_path / "label_schema.json").read_text())
                self._id2label = {int(k): str(v) for k, v in schema["id2label"].items()}
                num_labels = int(schema["num_labels"])

                device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
                tok = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                    str(self._encoder_path), trust_remote_code=True
                )
                full = AutoModelForMaskedLM.from_pretrained(
                    str(self._encoder_path), trust_remote_code=True
                )
                backbone = getattr(full, "lfm2", None)
                if backbone is None:
                    raise RuntimeError("MLM wrapper missing `.lfm2` bidirectional backbone")

                class _SeqClf(nn.Module):
                    """Mean-pooled encoder + linear classifier (matches the trainer)."""

                    def __init__(self, bb: Any, hs: int, nl: int) -> None:
                        super().__init__()
                        self.backbone = bb
                        self.dropout = nn.Dropout(0.1)
                        self.classifier = nn.Linear(hs, nl)

                    def forward(self, input_ids: Any, attention_mask: Any, **kw: Any) -> Any:
                        h = self.backbone(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).last_hidden_state
                        m = attention_mask.unsqueeze(-1).to(h.dtype)
                        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
                        return self.classifier(self.dropout(pooled))

                model = PeftModel.from_pretrained(
                    _SeqClf(backbone, backbone.config.hidden_size, num_labels),
                    str(self._adapter_path),
                )
                model.eval().to(device)
                self._tok = tok
                self._model = model
                self._device = device
                logger.info("F-503 router ready (device=%s, labels=%s)", device, self._id2label)
            except Exception as exc:  # missing deps/weights → inert, heuristics carry on
                self._load_failed = True
                logger.warning("F-503 router unavailable, degrading to heuristics: %s", exc)

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        """Return (label, probability) for one turn, or None if unavailable."""
        self._load()
        if self._model is None:
            return None
        import torch

        enc = self._tok(
            text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS, padding=True
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax(-1).item())
        return self._id2label.get(idx, "none"), float(probs[idx].item())

    def evaluate(self, state: TurnState) -> Optional[Trigger]:
        """Route the turn to a trigger (or stay silent). Inert when disabled.

        Hybrid: the router is primary; a `question_rescue` reclaims a question the
        router filed as an abstain/assertion label (none/topic).
        """
        if not self._enabled:
            return None
        text = state.text
        if not text or not text.strip():
            return None
        pred = self.predict(text)
        if pred is None:  # weights/deps unavailable — let the heuristic heads answer
            return None
        label, prob = pred

        # Hybrid question-rescue: router said none/topic but the heuristic sees a question.
        if label in _RESCUABLE and self._question_rescue is not None:
            q = self._question_rescue.evaluate(text, state.conversation_context)
            if q is not None:
                return Trigger(
                    type=TriggerType.QUESTION,
                    text=text,
                    confidence=max(prob, q.confidence),
                    source_context=state.conversation_context,
                    metadata={"head": self.name, "label": "question", "rescued_from": label},
                    timestamp=state.timestamp,
                )

        trig_type = _LABEL_TO_TYPE.get(label)
        if trig_type is None or prob < self._min_confidence:
            return None
        return Trigger(
            type=trig_type,
            text=text,
            confidence=prob,
            source_context=state.conversation_context,
            metadata={"head": self.name, "label": label},
            timestamp=state.timestamp,
        )
