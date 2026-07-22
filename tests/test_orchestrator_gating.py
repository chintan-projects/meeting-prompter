"""The listen gate must bite at the orchestrator's trigger choke point (D-02).

`lib/gating.py` is tested in isolation in test_gating.py. What matters here is
the wiring: that `_process_trigger` — the one function both capture pipelines
call — actually consults the gate, and that the explicit user paths bypass it.

The real `MeetingOrchestrator.__init__` loads models, so these tests bind the
real methods onto a minimal stand-in. That keeps the assertions on the actual
production code path rather than a reimplementation of it.
"""

from __future__ import annotations

import time
from types import MethodType, SimpleNamespace
from typing import Any, List, Optional

from lib.gating import ListenGate
from lib.generation.types import GenerationResult
from lib.orchestrator import MeetingOrchestrator
from lib.triggers.types import Trigger, TriggerType


def _result(answer: str = "a borrowable sentence") -> GenerationResult:
    return GenerationResult(
        answer=answer,
        trigger_type=TriggerType.QUESTION,
        confidence=0.9,
        method="retrieval",
        latency_ms=12.0,
        source="doc.md",
    )


class _FakeOrchestrator:
    """Minimal stand-in carrying the real _process_trigger / retrieve_for_text."""

    def __init__(self, gate: ListenGate, retrieval_first: bool = True) -> None:
        self.listen_gate = gate
        self.config = SimpleNamespace(triggers=SimpleNamespace(retrieval_first=retrieval_first))
        self.buffer = SimpleNamespace(get_recent_context=lambda: "")
        self.retrieval_calls: List[str] = []
        self.generated_calls: List[str] = []
        self._process_trigger = MethodType(MeetingOrchestrator._process_trigger, self)
        self.retrieve_for_text = MethodType(MeetingOrchestrator.retrieve_for_text, self)

    def _process_trigger_retrieval(self, trigger: Trigger) -> Optional[GenerationResult]:
        self.retrieval_calls.append(trigger.text)
        return _result()

    def _process_trigger_generated(self, trigger: Trigger) -> Optional[GenerationResult]:
        self.generated_calls.append(trigger.text)
        return _result()


def _trigger(ttype: TriggerType, text: str = "what is the timeline") -> Trigger:
    return Trigger(type=ttype, text=text, confidence=0.8, source_context="", timestamp=time.time())


class TestQuietByDefault:
    def test_question_suppressed_while_disarmed(self) -> None:
        orch: Any = _FakeOrchestrator(ListenGate())
        assert orch._process_trigger(_trigger(TriggerType.QUESTION)) is None
        assert orch.retrieval_calls == []

    def test_suppression_happens_before_retrieval(self) -> None:
        """The gate must short-circuit, not filter after the fact — a suppressed
        trigger should cost nothing, since it fires on most turns."""
        orch: Any = _FakeOrchestrator(ListenGate())
        for ttype in (TriggerType.QUESTION, TriggerType.TOPIC_MATCH, TriggerType.FOLLOW_UP):
            orch._process_trigger(_trigger(ttype))
        assert orch.retrieval_calls == []
        assert orch.generated_calls == []

    def test_alert_passes_while_disarmed(self) -> None:
        orch: Any = _FakeOrchestrator(ListenGate())
        result = orch._process_trigger(_trigger(TriggerType.ALERT, "pricing"))
        assert result is not None
        assert orch.retrieval_calls == ["pricing"]

    def test_armed_admits_questions(self) -> None:
        gate = ListenGate()
        gate.arm()
        orch: Any = _FakeOrchestrator(gate)
        assert orch._process_trigger(_trigger(TriggerType.QUESTION)) is not None

    def test_disarming_stops_the_flow_again(self) -> None:
        gate = ListenGate()
        orch: Any = _FakeOrchestrator(gate)
        gate.arm()
        orch._process_trigger(_trigger(TriggerType.QUESTION, "first"))
        gate.disarm()
        orch._process_trigger(_trigger(TriggerType.QUESTION, "second"))
        assert orch.retrieval_calls == ["first"]


class TestGateDisabled:
    def test_always_on_behaviour_when_gating_disabled(self) -> None:
        """enabled=false restores the pre-D-02 product without touching call sites."""
        orch: Any = _FakeOrchestrator(ListenGate(enabled=False))
        assert orch._process_trigger(_trigger(TriggerType.QUESTION)) is not None


class TestGenerationPathAlsoGated:
    def test_legacy_generative_path_respects_the_gate(self) -> None:
        """retrieval_first=false must not be a hole in the gate."""
        orch: Any = _FakeOrchestrator(ListenGate(), retrieval_first=False)
        assert orch._process_trigger(_trigger(TriggerType.QUESTION)) is None
        assert orch.generated_calls == []

    def test_legacy_path_used_when_armed(self) -> None:
        gate = ListenGate()
        gate.arm()
        orch: Any = _FakeOrchestrator(gate, retrieval_first=False)
        assert orch._process_trigger(_trigger(TriggerType.QUESTION)) is not None
        assert orch.generated_calls and not orch.retrieval_calls


class TestSelectToAnswerBypassesGate:
    """Spatial path: the user asking IS the permission."""

    def test_answers_while_disarmed(self) -> None:
        orch: Any = _FakeOrchestrator(ListenGate())
        result = orch.retrieve_for_text("what about compliance")
        assert result is not None
        assert orch.retrieval_calls == ["what about compliance"]

    def test_uses_retrieval_not_generation(self) -> None:
        """Select-to-answer stays instant — the LLM is a separate opt-in."""
        orch: Any = _FakeOrchestrator(ListenGate(), retrieval_first=False)
        orch.retrieve_for_text("what about compliance")
        assert orch.retrieval_calls and not orch.generated_calls

    def test_unknown_trigger_type_falls_back_to_question(self) -> None:
        orch: Any = _FakeOrchestrator(ListenGate())
        assert orch.retrieve_for_text("text", "nonsense") is not None
