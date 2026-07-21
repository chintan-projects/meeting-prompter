"""Frozen-encoder linear-probe head (F-510).

The v1 encoder head that needs NO model training: mean-pool the FROZEN
LFM2.5-Encoder-350M to 1024-d, then fit a logistic-regression probe on a small
labeled set (sklearn, seconds on CPU). Encoder-only, no forge, no GPU, no egress.

Ships behind the ``Head`` interface. It is wired as the default trigger router
ONLY if it beats the heuristic on a frozen held-out split; otherwise it stays
``enabled=False`` with its numbers recorded, and the heuristic heads remain the
default. ``evaluate_probe_gate`` computes that comparison honestly (per-class).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from lib.intelligence.heads.probe_data import (
    PROBE_LABELS,
    load_probe_examples,
)
from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger, TriggerType

if TYPE_CHECKING:
    from lib.intelligence.encoder import EncoderBackbone

logger = logging.getLogger(__name__)

# Predicted label → emitted trigger type. "none" emits nothing.
_LABEL_TO_TYPE: Dict[str, Optional[TriggerType]] = {
    "question": TriggerType.QUESTION,
    "alert": TriggerType.ALERT,
    "topic": TriggerType.TOPIC_MATCH,
    "followup": TriggerType.FOLLOW_UP,
    "none": None,
}


class _EmbedderLike:
    """Minimal structural type: something with embed / embed_batch."""

    def embed(self, text: str) -> List[float]: ...  # pragma: no cover
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...  # pragma: no cover


class LinearProbeHead:
    """Logistic-regression router over frozen mean-pooled encoder embeddings."""

    #: Nominal type for logging; a router emits whichever type it predicts.
    trigger_type = TriggerType.QUESTION

    def __init__(
        self,
        encoder: "EncoderBackbone | _EmbedderLike",
        examples: Optional[Sequence[Tuple[str, str]]] = None,
        enabled: bool = False,
        min_confidence: float = 0.0,
        random_state: int = 42,
        name: str = "linear_probe_router",
    ) -> None:
        self.name = name
        self._encoder = encoder
        self._examples = list(examples) if examples is not None else load_probe_examples()
        self._enabled = enabled
        self._min_confidence = min_confidence
        self._random_state = random_state
        self._clf: Optional[object] = None
        self._classes: List[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def fit(self) -> None:
        """Fit the logistic probe on frozen embeddings of the labeled set."""
        if self._clf is not None:
            return
        from sklearn.linear_model import LogisticRegression

        texts = [t for t, _ in self._examples]
        labels = [y for _, y in self._examples]
        features = self._encoder.embed_batch(texts)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(features, labels)
        self._clf = clf
        self._classes = list(clf.classes_)
        logger.info(
            "Linear-probe head fitted on %d examples, classes=%s",
            len(texts),
            self._classes,
        )

    def predict(self, embedding: Sequence[float]) -> Tuple[str, float]:
        """Return (label, probability) for a mean-pooled embedding."""
        self.fit()
        assert self._clf is not None
        probs = self._clf.predict_proba([list(embedding)])[0]  # type: ignore[attr-defined]
        best = max(range(len(probs)), key=lambda i: probs[i])
        return self._classes[best], float(probs[best])

    def evaluate(self, state: TurnState) -> Optional[Trigger]:
        """Head interface: route the turn to a trigger type (or stay silent).

        Returns ``None`` when disabled — so wiring it into the layer is inert
        until the gate flips it on. When enabled, "none" and below-threshold
        predictions also return ``None``.
        """
        if not self._enabled:
            return None
        text = state.text
        if not text or not text.strip():
            return None
        embedding = state.embedding
        if embedding is None:
            embedding = self._encoder.embed(text)
        label, prob = self.predict(embedding)
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


# ─── Honest gate: probe vs heuristic on a frozen held-out split ──────────


def _heuristic_route(text: str, question_trigger: object) -> str:
    """Heuristic-router label for one isolated utterance.

    The production heuristics only classify questions from isolated text — alert
    needs meeting watch-words, topic needs a RAG match, follow-up needs a pause —
    so on the 5-way isolated-utterance task the heuristic can only answer
    question vs none. Reported per-class so this asymmetry is transparent.
    """
    result = question_trigger.evaluate(text, "")  # type: ignore[attr-defined]
    return "question" if result is not None else "none"


def evaluate_probe_gate(
    encoder: "Optional[EncoderBackbone]" = None,
    test_size: float = 0.4,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit the probe on a train split and compare to the heuristic on held-out.

    Returns a dict with per-class + macro-F1 for both the probe and the
    heuristic router, plus the wiring decision. Encoder-only; no GPU/egress.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split

    from lib.config import TriggerConfig
    from lib.intelligence.encoder import EncoderBackbone
    from lib.triggers.question_trigger import QuestionTrigger

    enc = encoder or EncoderBackbone()
    examples = load_probe_examples()
    texts = [t for t, _ in examples]
    labels = [y for _, y in examples]
    features = enc.embed_batch(texts)

    x_tr, x_te, y_tr, y_te, _t_tr, t_te = train_test_split(
        features,
        labels,
        texts,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_tr)
    probe_pred = list(clf.predict(x_te))
    probe_macro_f1 = float(f1_score(y_te, probe_pred, average="macro"))

    qt = QuestionTrigger(TriggerConfig())
    heur_pred = [_heuristic_route(t, qt) for t in t_te]
    heur_macro_f1 = float(f1_score(y_te, heur_pred, average="macro", zero_division=0))

    probe_report = classification_report(
        y_te, probe_pred, labels=list(PROBE_LABELS), zero_division=0, output_dict=True
    )
    heur_report = classification_report(
        y_te, heur_pred, labels=list(PROBE_LABELS), zero_division=0, output_dict=True
    )

    beats = probe_macro_f1 > heur_macro_f1
    probe_q = probe_report["question"]["f1-score"]
    heur_q = heur_report["question"]["f1-score"]
    # Conservative wiring rule: beating macro-F1 is necessary but not sufficient.
    # The probe only wins overall because the heuristic has no isolated-text
    # mechanism for alert/topic/followup. Do NOT replace the heuristic default
    # unless the probe also does not regress the ONE class the heuristic owns
    # (question). On this synthetic 70-example set the probe regresses question
    # (~0.92 < 1.0), so the honest call is: keep heuristic default, probe wired-off.
    question_regression = probe_q < heur_q
    wire_as_default = bool(beats and not question_regression)
    return {
        "n_total": len(examples),
        "n_test": len(y_te),
        "probe_macro_f1": round(probe_macro_f1, 4),
        "heuristic_macro_f1": round(heur_macro_f1, 4),
        "probe_beats_heuristic": beats,
        "question_regression": question_regression,
        "wire_as_default": wire_as_default,
        "probe_per_class_f1": {k: round(probe_report[k]["f1-score"], 3) for k in PROBE_LABELS},
        "heuristic_per_class_f1": {k: round(heur_report[k]["f1-score"], 3) for k in PROBE_LABELS},
    }
