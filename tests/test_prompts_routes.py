"""Route tests for the D-02 user-gated answer surfaces.

Covers the listen-window endpoints and select-to-answer. The gate's own logic is
tested in test_gating.py and its wiring in test_orchestrator_gating.py; what
matters here is the HTTP contract and that arming broadcasts state to clients.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from lib.gating import ListenGate
from lib.generation.types import GenerationResult
from lib.triggers.types import TriggerType
from src.api.routes import prompts as prompts_route


class _FakeOrchestrator:
    def __init__(self, gate: ListenGate, answer: Optional[str] = "a borrowable sentence") -> None:
        self.listen_gate = gate
        self._answer = answer
        self.asked: list[str] = []

    def retrieve_for_text(self, text: str, trigger_type: str = "question") -> Any:
        self.asked.append(text)
        if self._answer is None:
            return None
        return GenerationResult(
            answer=self._answer,
            trigger_type=TriggerType.QUESTION,
            confidence=0.82,
            method="retrieval",
            latency_ms=17.0,
            source="playbook.md",
            heading="Part 1 > 1.3",
            source_text="the full borrowable unit",
        )


class _FakeSession:
    def __init__(self, orch: Any) -> None:
        self._orchestrator = orch
        self._prompt_queue: asyncio.Queue[dict] = asyncio.Queue()


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> Any:
    orch = _FakeOrchestrator(ListenGate())
    session = _FakeSession(orch)
    monkeypatch.setattr(prompts_route, "get_session", lambda: session)
    return SimpleNamespace(orch=orch, session=session, gate=orch.listen_gate)


class TestNoSession:
    def test_listen_state_requires_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompts_route, "get_session", lambda: _FakeSession(None))
        with pytest.raises(HTTPException) as exc:
            prompts_route.listen_state()
        assert exc.value.status_code == 409

    def test_answer_requires_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prompts_route, "get_session", lambda: _FakeSession(None))
        with pytest.raises(HTTPException) as exc:
            prompts_route.answer_selection(prompts_route.AnswerRequest(text="what about it"))
        assert exc.value.status_code == 409


class TestListenEndpoints:
    def test_starts_disarmed(self, wired: Any) -> None:
        """The product default is quiet — a fresh session must not be listening."""
        assert prompts_route.listen_state()["armed"] is False

    def test_toggle_with_no_body_flips(self, wired: Any) -> None:
        first = asyncio.run(prompts_route.set_listen(prompts_route.ListenRequest()))
        assert first["armed"] is True
        second = asyncio.run(prompts_route.set_listen(prompts_route.ListenRequest()))
        assert second["armed"] is False

    def test_explicit_set_is_idempotent(self, wired: Any) -> None:
        for _ in range(3):
            state = asyncio.run(prompts_route.set_listen(prompts_route.ListenRequest(armed=True)))
            assert state["armed"] is True

    def test_arming_broadcasts_listen_state(self, wired: Any) -> None:
        """Every client must agree on the window — the backend is the truth,
        not whichever one received the keypress."""
        asyncio.run(prompts_route.set_listen(prompts_route.ListenRequest(armed=True)))
        msg = wired.session._prompt_queue.get_nowait()
        assert msg["type"] == "listen_state"
        assert msg["armed"] is True

    def test_disarm_broadcasts_too(self, wired: Any) -> None:
        asyncio.run(prompts_route.set_listen(prompts_route.ListenRequest(armed=False)))
        assert wired.session._prompt_queue.get_nowait()["armed"] is False

    def test_state_reflects_the_gate(self, wired: Any) -> None:
        wired.gate.arm()
        assert prompts_route.listen_state()["armed"] is True
        assert prompts_route.listen_state()["always_on"] == ["alert"]


class TestSelectToAnswer:
    def test_rejects_empty_text(self, wired: Any) -> None:
        with pytest.raises(HTTPException) as exc:
            prompts_route.answer_selection(prompts_route.AnswerRequest(text="   "))
        assert exc.value.status_code == 400

    def test_answers_while_disarmed(self, wired: Any) -> None:
        """Asking IS the permission — the spatial path ignores the gate."""
        assert wired.gate.is_armed() is False
        card = prompts_route.answer_selection(
            prompts_route.AnswerRequest(text="what about compliance")
        )
        assert card["answer"] == "a borrowable sentence"
        assert wired.orch.asked == ["what about compliance"]

    def test_card_shape_matches_the_ws_prompt_contract(self, wired: Any) -> None:
        card = prompts_route.answer_selection(prompts_route.AnswerRequest(text="what about it"))
        for key in (
            "type",
            "trigger_type",
            "trigger_text",
            "answer",
            "confidence",
            "method",
            "latency_ms",
            "source",
            "heading",
            "source_text",
        ):
            assert key in card, f"missing {key}"
        assert card["type"] == "prompt"
        assert card["method"] == "retrieval"

    def test_user_requested_cards_never_auto_dismiss(self, wired: Any) -> None:
        card = prompts_route.answer_selection(prompts_route.AnswerRequest(text="what about it"))
        assert card["persistence"] == "persistent"
        assert card["dismiss_ms"] == 0

    def test_no_match_returns_a_note_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silence beats noise, but an explicit request deserves an explicit answer."""
        session = _FakeSession(_FakeOrchestrator(ListenGate(), answer=None))
        monkeypatch.setattr(prompts_route, "get_session", lambda: session)
        result = prompts_route.answer_selection(prompts_route.AnswerRequest(text="unrelated thing"))
        assert result["answer"] == ""
        assert "note" in result

    def test_suppressed_marker_answer_is_not_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeOrchestrator(ListenGate(), answer="[no match]"))
        monkeypatch.setattr(prompts_route, "get_session", lambda: session)
        assert (
            prompts_route.answer_selection(prompts_route.AnswerRequest(text="unrelated thing"))[
                "answer"
            ]
            == ""
        )
