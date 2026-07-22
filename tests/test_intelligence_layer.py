"""Tests for the encoder intelligence layer (F-501).

Covers the new abstractions with fakes only — no model load — so they run in the
fast suite. The parity of the refactored TriggerEngine with prior behavior is
covered transitively by the existing trigger/buffer/session suites.
"""

from __future__ import annotations

from typing import List, Optional


from lib.intelligence import (
    EncoderBackbone,
    EncoderIntelligenceLayer,
    HeuristicHead,
    TurnState,
)
from lib.intelligence.heads.base import Head
from lib.triggers.types import Trigger, TriggerType


def _trigger(t: TriggerType, text: str = "x") -> Trigger:
    return Trigger(type=t, text=text, confidence=0.9)


class _FakeEvaluator:
    """Legacy-shaped evaluator: evaluate(text, context) -> Optional[Trigger]."""

    def __init__(self, result: Optional[Trigger], record: List[tuple]) -> None:
        self._result = result
        self._record = record

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]:
        self._record.append((text, conversation_context))
        return self._result


class _RaisingHead:
    name = "boom"
    trigger_type = TriggerType.QUESTION

    def evaluate(self, state: TurnState) -> Optional[Trigger]:
        raise RuntimeError("head failure")


# ─── TurnState ───────────────────────────────────────────────────────────


class TestTurnState:
    def test_defaults(self) -> None:
        s = TurnState(text="hello")
        assert s.conversation_context == ""
        assert s.timestamp == 0.0
        assert s.embedding is None
        assert s.triggers == []


# ─── HeuristicHead ───────────────────────────────────────────────────────


class TestHeuristicHead:
    def test_passes_text_and_context(self) -> None:
        rec: List[tuple] = []
        head = HeuristicHead(
            "question",
            TriggerType.QUESTION,
            _FakeEvaluator(_trigger(TriggerType.QUESTION), rec),
        )
        state = TurnState(text="what time?", conversation_context="ctx")
        out = head.evaluate(state)
        assert out is not None and out.type is TriggerType.QUESTION
        assert rec == [("what time?", "ctx")]

    def test_satisfies_head_protocol(self) -> None:
        head = HeuristicHead("alert", TriggerType.ALERT, _FakeEvaluator(None, []))
        assert isinstance(head, Head)


# ─── EncoderIntelligenceLayer ────────────────────────────────────────────


class TestEncoderIntelligenceLayer:
    def test_collects_and_priority_sorts(self) -> None:
        # topic (priority 3) registered before alert (priority 1) — output sorts.
        heads = [
            HeuristicHead(
                "topic",
                TriggerType.TOPIC_MATCH,
                _FakeEvaluator(_trigger(TriggerType.TOPIC_MATCH), []),
            ),
            HeuristicHead(
                "alert",
                TriggerType.ALERT,
                _FakeEvaluator(_trigger(TriggerType.ALERT), []),
            ),
        ]
        layer = EncoderIntelligenceLayer(heads)
        state = TurnState(text="hi")
        out = layer.process(state)
        assert [t.type for t in out] == [TriggerType.ALERT, TriggerType.TOPIC_MATCH]
        assert state.triggers == out

    def test_none_results_dropped(self) -> None:
        heads = [
            HeuristicHead("a", TriggerType.ALERT, _FakeEvaluator(None, [])),
            HeuristicHead(
                "q",
                TriggerType.QUESTION,
                _FakeEvaluator(_trigger(TriggerType.QUESTION), []),
            ),
        ]
        out = EncoderIntelligenceLayer(heads).process(TurnState(text="hi"))
        assert [t.type for t in out] == [TriggerType.QUESTION]

    def test_head_exception_isolated(self) -> None:
        heads: List[Head] = [
            _RaisingHead(),
            HeuristicHead(
                "q",
                TriggerType.QUESTION,
                _FakeEvaluator(_trigger(TriggerType.QUESTION), []),
            ),
        ]
        out = EncoderIntelligenceLayer(heads).process(TurnState(text="hi"))
        assert [t.type for t in out] == [TriggerType.QUESTION]

    def test_cold_head_skipped_on_filler_turn(self) -> None:
        rec: List[tuple] = []
        hot = HeuristicHead("question", TriggerType.QUESTION, _FakeEvaluator(None, rec))
        cold_rec: List[tuple] = []
        cold = HeuristicHead(
            "topic",
            TriggerType.TOPIC_MATCH,
            _FakeEvaluator(_trigger(TriggerType.TOPIC_MATCH), cold_rec),
            cold=True,
        )
        layer = EncoderIntelligenceLayer([hot, cold], cold_path_min_words=3)
        state = TurnState(text="yeah totally")  # 2 words → below threshold
        out = layer.process(state)
        assert out == []
        assert cold_rec == []  # RAG-backed head never called
        assert rec == [("yeah totally", "")]  # hot head still ran
        assert state.ran_cold is False

    def test_cold_head_runs_on_substantive_turn(self) -> None:
        cold_rec: List[tuple] = []
        cold = HeuristicHead(
            "topic",
            TriggerType.TOPIC_MATCH,
            _FakeEvaluator(_trigger(TriggerType.TOPIC_MATCH), cold_rec),
            cold=True,
        )
        layer = EncoderIntelligenceLayer([cold], cold_path_min_words=3)
        state = TurnState(text="what is the deployment timeline exactly")
        out = layer.process(state)
        assert [t.type for t in out] == [TriggerType.TOPIC_MATCH]
        assert len(cold_rec) == 1
        assert state.ran_cold is True

    def test_hot_head_always_runs(self) -> None:
        rec: List[tuple] = []
        hot = HeuristicHead(
            "question",
            TriggerType.QUESTION,
            _FakeEvaluator(_trigger(TriggerType.QUESTION), rec),
        )
        layer = EncoderIntelligenceLayer([hot], cold_path_min_words=5)
        out = layer.process(TurnState(text="hi"))  # 1 word
        assert [t.type for t in out] == [TriggerType.QUESTION]

    def test_embedding_off_by_default(self) -> None:
        class _Enc:
            def embed(self, text: str) -> List[float]:
                raise AssertionError("must not be called when disabled")

        layer = EncoderIntelligenceLayer([], encoder=_Enc())  # type: ignore[arg-type]
        state = TurnState(text="hi")
        layer.process(state)
        assert state.embedding is None

    def test_embedding_populated_when_enabled(self) -> None:
        class _Enc:
            def embed(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]

        layer = EncoderIntelligenceLayer(
            [], encoder=_Enc(), compute_embedding=True  # type: ignore[arg-type]
        )
        state = TurnState(text="hi")
        layer.process(state)
        assert state.embedding == [0.1, 0.2, 0.3]

    def test_embedding_degrades_on_failure(self) -> None:
        class _Enc:
            def embed(self, text: str) -> List[float]:
                raise RuntimeError("model missing")

        layer = EncoderIntelligenceLayer(
            [], encoder=_Enc(), compute_embedding=True  # type: ignore[arg-type]
        )
        state = TurnState(text="hi")
        # Must not raise — degrades to no embedding.
        layer.process(state)
        assert state.embedding is None


# ─── EncoderBackbone (no model load) ─────────────────────────────────────


class TestEncoderBackbone:
    def test_resolve_path_points_at_registry(self) -> None:
        eb = EncoderBackbone()
        assert eb.resolve_path().name == "LFM2.5-Encoder-350M"

    def test_dimension(self) -> None:
        assert EncoderBackbone().dimension == 1024

    def test_embed_empty_batch(self) -> None:
        assert EncoderBackbone().embed_batch([]) == []

    def test_is_available_bool(self) -> None:
        assert isinstance(EncoderBackbone().is_available(), bool)


# ─── TriggerEngine still exposes its public API through the layer ────────


class TestTriggerEngineIntegration:
    def test_engine_uses_layer(self) -> None:
        from lib.config import TriggerConfig
        from lib.triggers.engine import TriggerEngine

        class _RAG:
            def query(self, text: str):
                return ("", 0.0, "")

        engine = TriggerEngine(TriggerConfig(), _RAG())
        assert isinstance(engine.layer, EncoderIntelligenceLayer)
        # Three turn-evaluated heads: alert, question, topic.
        assert [h.name for h in engine.layer.heads] == ["alert", "question", "topic"]

    def test_evaluate_returns_list(self) -> None:
        from lib.config import TriggerConfig
        from lib.triggers.engine import TriggerEngine

        class _RAG:
            def query(self, text: str):
                return ("", 0.0, "")

        engine = TriggerEngine(TriggerConfig(), _RAG())
        out = engine.evaluate("just a plain statement", "")
        assert isinstance(out, list)
