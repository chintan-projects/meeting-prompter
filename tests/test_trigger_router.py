"""Tests for the F-503 trigger-router head (routing + hybrid question-rescue).

The model itself is never loaded here — ``predict`` is monkeypatched — so these
run without torch/transformers/peft or the adapter present, exercising the
routing/rescue/gating logic that is the head's contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from lib.intelligence.heads.base import Head
from lib.intelligence.heads.trigger_router import TriggerRouterHead
from lib.intelligence.turn_state import TurnState
from lib.triggers.types import Trigger, TriggerType


def _head(**kw: object) -> TriggerRouterHead:
    return TriggerRouterHead(Path("/nonexistent-models"), **kw)  # type: ignore[arg-type]


def _q(conf: float = 0.7) -> Trigger:
    return Trigger(type=TriggerType.QUESTION, text="x", confidence=conf)


def test_satisfies_head_protocol() -> None:
    assert isinstance(_head(), Head)


def test_disabled_returns_none() -> None:
    h = _head(enabled=False)
    h.predict = lambda t: ("question", 0.99)  # type: ignore[assignment,method-assign]
    assert h.evaluate(TurnState(text="what is the timeline")) is None


def test_is_available_false_for_missing_weights() -> None:
    assert _head().is_available() is False


def test_routes_each_active_label(monkeypatch) -> None:
    cases = {
        "question": TriggerType.QUESTION,
        "alert": TriggerType.ALERT,
        "topic": TriggerType.TOPIC_MATCH,
        "followup": TriggerType.FOLLOW_UP,
    }
    for label, ttype in cases.items():
        h = _head(enabled=True)
        monkeypatch.setattr(h, "predict", lambda t, _l=label: (_l, 0.9))
        trig = h.evaluate(TurnState(text="a turn"))
        assert trig is not None and trig.type == ttype
        assert trig.metadata["label"] == label
        assert trig.confidence == 0.9


def test_none_label_stays_silent(monkeypatch) -> None:
    h = _head(enabled=True)
    monkeypatch.setattr(h, "predict", lambda t: ("none", 0.95))
    assert h.evaluate(TurnState(text="yeah totally")) is None


def test_question_rescue_from_topic(monkeypatch) -> None:
    q = MagicMock()
    q.evaluate.return_value = _q(0.7)
    h = _head(enabled=True, question_rescue=q)
    monkeypatch.setattr(h, "predict", lambda t: ("topic", 0.6))
    trig = h.evaluate(TurnState(text="is that the plan then"))
    assert trig is not None
    assert trig.type == TriggerType.QUESTION
    assert trig.metadata["rescued_from"] == "topic"
    assert trig.confidence == 0.7  # max(router 0.6, heuristic 0.7)


def test_no_rescue_when_heuristic_silent(monkeypatch) -> None:
    q = MagicMock()
    q.evaluate.return_value = None
    h = _head(enabled=True, question_rescue=q)
    monkeypatch.setattr(h, "predict", lambda t: ("topic", 0.8))
    trig = h.evaluate(TurnState(text="the vocoder is 258 megabytes"))
    assert trig is not None and trig.type == TriggerType.TOPIC_MATCH


def test_alert_is_not_rescuable(monkeypatch) -> None:
    q = MagicMock()
    q.evaluate.return_value = _q(0.9)  # would fire, but alert must win
    h = _head(enabled=True, question_rescue=q)
    monkeypatch.setattr(h, "predict", lambda t: ("alert", 0.85))
    trig = h.evaluate(TurnState(text="we are out of memory budget"))
    assert trig is not None and trig.type == TriggerType.ALERT
    q.evaluate.assert_not_called()


def test_min_confidence_gate(monkeypatch) -> None:
    h = _head(enabled=True, min_confidence=0.7)
    monkeypatch.setattr(h, "predict", lambda t: ("alert", 0.5))
    assert h.evaluate(TurnState(text="something")) is None


def test_graceful_when_weights_unavailable(monkeypatch) -> None:
    h = _head(enabled=True)
    monkeypatch.setattr(h, "predict", lambda t: None)  # deps/weights missing
    assert h.evaluate(TurnState(text="what is the plan")) is None


def test_empty_text_silent(monkeypatch) -> None:
    h = _head(enabled=True)
    monkeypatch.setattr(h, "predict", lambda t: ("question", 0.9))
    assert h.evaluate(TurnState(text="   ")) is None
