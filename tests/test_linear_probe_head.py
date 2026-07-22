"""Tests for the frozen-encoder linear-probe head (F-510).

Fast tests use a deterministic fake encoder (perfectly separable one-hot vectors)
so the probe fit/predict/routing logic runs with no model load. The @slow test
runs the real gate (encoder + heuristic) and asserts the recorded decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from lib.intelligence.heads.linear_probe import LinearProbeHead
from lib.intelligence.heads.probe_data import (
    PROBE_LABELS,
    SEED_EXAMPLES,
    load_probe_examples,
    write_seed_dataset,
)
from lib.intelligence.turn_state import TurnState
from lib.triggers.types import TriggerType

_ONEHOT = {"q": 0, "a": 1, "t": 2, "f": 3, "n": 4}


class _FakeEncoder:
    """One-hot embeddings keyed by the first character — perfectly separable."""

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * 5
        vec[_ONEHOT[text.strip()[0]]] = 1.0
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


_FAKE_EXAMPLES = [
    ("q one", "question"),
    ("q two", "question"),
    ("q three", "question"),
    ("a one", "alert"),
    ("a two", "alert"),
    ("a three", "alert"),
    ("t one", "topic"),
    ("t two", "topic"),
    ("f one", "followup"),
    ("f two", "followup"),
    ("n one", "none"),
    ("n two", "none"),
]


def _probe(enabled: bool = True) -> LinearProbeHead:
    return LinearProbeHead(_FakeEncoder(), examples=_FAKE_EXAMPLES, enabled=enabled)


# ─── Probe data ──────────────────────────────────────────────────────────


class TestProbeData:
    def test_seed_is_five_way(self) -> None:
        labels = {y for _, y in SEED_EXAMPLES}
        assert labels == set(PROBE_LABELS)
        assert len(SEED_EXAMPLES) >= 60

    def test_load_returns_seed_without_overlay(self) -> None:
        examples = load_probe_examples(overlay_path=Path("/nonexistent/overlay.jsonl"))
        assert len(examples) == len(SEED_EXAMPLES)

    def test_write_and_merge_overlay(self, tmp_path: Path) -> None:
        overlay = tmp_path / "probe.jsonl"
        n = write_seed_dataset(overlay)
        assert n == len(SEED_EXAMPLES)
        # Re-persisting the seed then loading is idempotent (dedup, no leak).
        assert len(load_probe_examples(overlay_path=overlay)) == len(SEED_EXAMPLES)
        # A hand-added, non-duplicate example merges in.
        with overlay.open("a") as f:
            f.write('{"text": "extra alert about a blocker", "label": "alert"}\n')
        merged = load_probe_examples(overlay_path=overlay)
        assert len(merged) == len(SEED_EXAMPLES) + 1

    def test_overlay_bad_lines_skipped(self, tmp_path: Path) -> None:
        overlay = tmp_path / "probe.jsonl"
        overlay.write_text(
            '{"text": "ok", "label": "topic"}\n'
            "not json\n"
            '{"text": "no label"}\n'
            '{"text": "bad", "label": "notalabel"}\n'
        )
        merged = load_probe_examples(overlay_path=overlay)
        # Only the first line is valid → seed + 1.
        assert len(merged) == len(SEED_EXAMPLES) + 1


# ─── Probe fit / predict ─────────────────────────────────────────────────


class TestLinearProbeFitPredict:
    def test_predicts_separable_classes(self) -> None:
        probe = _probe()
        assert probe.predict([1.0, 0, 0, 0, 0])[0] == "question"
        assert probe.predict([0, 1.0, 0, 0, 0])[0] == "alert"
        assert probe.predict([0, 0, 0, 0, 1.0])[0] == "none"

    def test_prob_is_confidence(self) -> None:
        _, prob = _probe().predict([1.0, 0, 0, 0, 0])
        assert 0.0 <= prob <= 1.0


# ─── Head routing (enabled/disabled) ─────────────────────────────────────


class TestLinearProbeHeadRouting:
    def test_disabled_returns_none(self) -> None:
        probe = _probe(enabled=False)
        assert probe.evaluate(TurnState(text="q new")) is None

    def test_enabled_routes_to_type(self) -> None:
        probe = _probe(enabled=True)
        trig = probe.evaluate(TurnState(text="a new"))
        assert trig is not None and trig.type is TriggerType.ALERT
        assert trig.metadata["label"] == "alert"
        assert trig.metadata["head"] == "linear_probe_router"

    def test_none_label_stays_silent(self) -> None:
        probe = _probe(enabled=True)
        assert probe.evaluate(TurnState(text="n new")) is None

    def test_uses_precomputed_embedding(self) -> None:
        probe = _probe(enabled=True)
        # embedding says "topic" even though text char says "question"
        state = TurnState(text="q new", embedding=[0, 0, 1.0, 0, 0])
        trig = probe.evaluate(state)
        assert trig is not None and trig.type is TriggerType.TOPIC_MATCH

    def test_empty_text_silent(self) -> None:
        assert _probe(enabled=True).evaluate(TurnState(text="   ")) is None

    def test_min_confidence_suppresses(self) -> None:
        probe = LinearProbeHead(
            _FakeEncoder(), examples=_FAKE_EXAMPLES, enabled=True, min_confidence=1.01
        )
        assert probe.evaluate(TurnState(text="a new")) is None


# ─── Engine wiring: probe present but off ────────────────────────────────


class TestEngineProbeWiring:
    def test_probe_wired_off_when_encoder_present(self) -> None:
        from lib.config import TriggerConfig
        from lib.triggers.engine import TriggerEngine

        class _RAG:
            def query(self, text: str):
                return ("", 0.0, "")

        engine = TriggerEngine(TriggerConfig(), _RAG(), encoder=_FakeEncoder())
        assert engine.probe_head is not None
        assert engine.probe_head.enabled is False
        # It is in the layer's head list (wired) but inert (off).
        assert engine.probe_head in engine.layer.heads

    def test_no_probe_without_encoder(self) -> None:
        from lib.config import TriggerConfig
        from lib.triggers.engine import TriggerEngine

        class _RAG:
            def query(self, text: str):
                return ("", 0.0, "")

        engine = TriggerEngine(TriggerConfig(), _RAG())
        assert engine.probe_head is None


# ─── Slow gate (real encoder + heuristic) ────────────────────────────────


@pytest.mark.slow
class TestProbeGate:
    def test_recorded_decision(self) -> None:
        from lib.intelligence.heads.linear_probe import evaluate_probe_gate

        r = evaluate_probe_gate()
        # Probe reproduces the spike's off-the-shelf ceiling on the 5-way task.
        assert r["probe_macro_f1"] >= 0.75
        assert r["probe_beats_heuristic"] is True
        # But it regresses the question class the heuristic owns → stays off.
        assert r["question_regression"] is True
        assert r["wire_as_default"] is False
