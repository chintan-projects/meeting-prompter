"""Tests for the D-02 listen gate (lib/gating.py)."""

from __future__ import annotations

import threading

from lib.gating import DEFAULT_ALWAYS_ON, ListenGate


class TestDefaultQuiet:
    """The product default: nothing pushes unless the user opened the tap."""

    def test_disarmed_suppresses_question(self) -> None:
        assert ListenGate().allows("question") is False

    def test_disarmed_suppresses_topic_and_followup(self) -> None:
        gate = ListenGate()
        assert gate.allows("topic_match") is False
        assert gate.allows("follow_up") is False

    def test_alert_is_always_on(self) -> None:
        """Watch words are pre-authorised — the user named the terms."""
        assert ListenGate().allows("alert") is True

    def test_armed_admits_everything(self) -> None:
        gate = ListenGate()
        gate.arm()
        assert gate.allows("question") is True
        assert gate.allows("topic_match") is True

    def test_always_on_default_is_alert_only(self) -> None:
        assert DEFAULT_ALWAYS_ON == ("alert",)

    def test_trigger_type_match_is_case_insensitive(self) -> None:
        assert ListenGate(always_on=("ALERT",)).allows("alert") is True
        assert ListenGate().allows("ALERT") is True


class TestDisabledGate:
    """enabled=False restores pre-D-02 always-on behaviour."""

    def test_everything_passes_when_disabled(self) -> None:
        gate = ListenGate(enabled=False)
        assert gate.allows("question") is True
        assert gate.allows("follow_up") is True
        assert gate.is_armed() is False  # still reports the real window state


class TestToggle:
    def test_toggle_opens_then_closes(self) -> None:
        gate = ListenGate()
        assert gate.toggle() is True
        assert gate.is_armed() is True
        assert gate.toggle() is False
        assert gate.is_armed() is False

    def test_disarm_is_idempotent(self) -> None:
        gate = ListenGate()
        assert gate.disarm() is False
        assert gate.disarm() is False

    def test_rearm_restarts_the_clock(self) -> None:
        gate = ListenGate(max_listen_seconds=10)
        gate.arm(now=100.0)
        gate.arm(now=105.0)
        # Without the restart this would have expired at 110.
        assert gate.is_armed(now=114.0) is True


class TestSafetyCap:
    """max_listen_seconds=0 is the chosen default: the window never times out."""

    def test_uncapped_window_stays_open_indefinitely(self) -> None:
        gate = ListenGate()
        gate.arm(now=0.0)
        assert gate.is_armed(now=86_400.0) is True

    def test_capped_window_expires(self) -> None:
        gate = ListenGate(max_listen_seconds=30)
        gate.arm(now=100.0)
        assert gate.is_armed(now=129.0) is True
        assert gate.is_armed(now=130.0) is False

    def test_expiry_suppresses_triggers_again(self) -> None:
        gate = ListenGate(max_listen_seconds=30)
        gate.arm(now=100.0)
        assert gate.allows("question", now=110.0) is True
        assert gate.allows("question", now=140.0) is False

    def test_alert_survives_expiry(self) -> None:
        gate = ListenGate(max_listen_seconds=30)
        gate.arm(now=100.0)
        assert gate.allows("alert", now=999.0) is True

    def test_negative_cap_is_clamped_to_uncapped(self) -> None:
        gate = ListenGate(max_listen_seconds=-5)
        gate.arm(now=0.0)
        assert gate.is_armed(now=1000.0) is True


class TestState:
    def test_disarmed_state(self) -> None:
        state = ListenGate().state()
        assert state["armed"] is False
        assert state["since"] is None
        assert state["expires_at"] is None
        assert state["always_on"] == ["alert"]

    def test_armed_state_reports_since(self) -> None:
        gate = ListenGate()
        gate.arm(now=42.0)
        state = gate.state(now=50.0)
        assert state["armed"] is True
        assert state["since"] == 42.0
        assert state["expires_at"] is None  # uncapped

    def test_capped_state_reports_expiry(self) -> None:
        gate = ListenGate(max_listen_seconds=30)
        gate.arm(now=100.0)
        assert gate.state(now=105.0)["expires_at"] == 130.0

    def test_state_after_expiry_is_disarmed(self) -> None:
        gate = ListenGate(max_listen_seconds=30)
        gate.arm(now=100.0)
        state = gate.state(now=200.0)
        assert state["armed"] is False
        assert state["since"] is None


class TestThreadSafety:
    """Two capture threads call allows() while the API arms from the loop."""

    def test_concurrent_toggle_and_allows_is_consistent(self) -> None:
        gate = ListenGate()
        errors: list[BaseException] = []

        def toggler() -> None:
            try:
                for _ in range(200):
                    gate.toggle()
            except BaseException as e:  # noqa: BLE001 — surface any race to the assert
                errors.append(e)

        def asker() -> None:
            try:
                for _ in range(200):
                    assert gate.allows("question") in (True, False)
                    assert gate.allows("alert") is True
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=toggler), threading.Thread(target=asker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_state_never_reports_armed_without_since(self) -> None:
        """The invariant a naive check-then-read would break under contention."""
        gate = ListenGate()
        errors: list[str] = []

        def churn() -> None:
            for _ in range(300):
                gate.toggle()

        def observe() -> None:
            for _ in range(300):
                s = gate.state()
                if s["armed"] and s["since"] is None:
                    errors.append("armed with no since")

        threads = [threading.Thread(target=churn), threading.Thread(target=observe)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
