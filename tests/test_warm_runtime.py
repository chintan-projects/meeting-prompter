"""Tests for the persistent warm-model runtime (F-508). No model load."""

from __future__ import annotations

from lib.intelligence.encoder import EncoderBackbone
from lib.warm_runtime import WarmModelRuntime


class TestEncoderSingleton:
    def test_encoder_is_shared_instance(self) -> None:
        rt = WarmModelRuntime()
        a = rt.encoder()
        b = rt.encoder()
        assert a is b
        assert isinstance(a, EncoderBackbone)

    def test_encoder_not_loaded_until_used(self) -> None:
        rt = WarmModelRuntime()
        rt.encoder()  # construct, but do not embed
        assert rt.status()["encoder"] is False  # weights not resident yet


class TestRegistry:
    def test_register_and_get(self) -> None:
        rt = WarmModelRuntime()
        sentinel = object()
        rt.register("embedder", sentinel)
        assert rt.get("embedder") is sentinel

    def test_status_reports_registered(self) -> None:
        rt = WarmModelRuntime()
        rt.register("instruct", object())
        rt.register("embedder", None)
        st = rt.status()
        assert st["instruct"] is True
        assert st["embedder"] is False
        assert "encoder" in st

    def test_get_missing_is_none(self) -> None:
        assert WarmModelRuntime().get("nope") is None


class TestLifecycle:
    def test_teardown_clears(self) -> None:
        rt = WarmModelRuntime()
        rt.encoder()
        rt.register("instruct", object())
        rt.teardown()
        st = rt.status()
        assert st == {"encoder": False}  # registry cleared, encoder released

    def test_teardown_idempotent(self) -> None:
        rt = WarmModelRuntime()
        rt.teardown()
        rt.teardown()  # must not raise
        assert rt.status()["encoder"] is False

    def test_warm_does_not_raise_without_model(self, monkeypatch) -> None:
        rt = WarmModelRuntime()

        class _FakeEnc:
            def _load(self) -> None:
                raise FileNotFoundError("no weights in test env")

        monkeypatch.setattr(rt, "_encoder", _FakeEnc())
        # warm() must swallow load failures and degrade to lazy loading.
        rt.warm(encoder=True)
