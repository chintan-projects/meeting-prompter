"""Unit tests for lib.corpus — text cleaning + the heuristic distiller (F-701).

Covers the deterministic, no-network pieces. The cloud backend is offline/opt-in
(ADR-001) and is exercised via the lab, not here; its guard rails (credential
fail-fast, unknown backend) are tested without any network call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.corpus import distiller, local
from lib.corpus.text import clean_markdown


# --- clean_markdown --------------------------------------------------------
def test_clean_markdown_strips_headers_emphasis_and_inline_code() -> None:
    assert clean_markdown("## Heading\n**bold** and `code` here.") == "Heading bold and code here."


def test_clean_markdown_drops_code_fences_and_tables() -> None:
    src = "Intro line.\n```\nprint('x')\n```\n| a | b |\n|---|---|\n| 1 | 2 |\nOutro line."
    out = clean_markdown(src)
    assert "print" not in out and "|" not in out
    assert "Intro line." in out and "Outro line." in out


def test_clean_markdown_rewrites_links_to_text() -> None:
    assert clean_markdown("see [the docs](https://x.example/y)") == "see the docs"


# --- heuristic distiller ---------------------------------------------------
def test_distill_heuristic_skips_thin_sections() -> None:
    assert distiller._distill_heuristic("Part 1", "too short") == []


def test_distill_consolidated_keeps_whole_section_atomic_truncates() -> None:
    # A long section: atomic caps at MAX_UNIT_WORDS; consolidated keeps it all.
    text = "Fact one is important. " * 40  # ~160 words
    atomic = distiller._distill_heuristic("Topic", text, mode="atomic")[0]
    consolidated = distiller._distill_heuristic("Topic", text, mode="consolidated")[0]
    assert len(consolidated.split()) > len(atomic.split())
    assert len(atomic.split()) <= distiller.MAX_UNIT_WORDS + 20  # cap (plus topic prefix)


def test_distill_heuristic_produces_self_contained_unit() -> None:
    text = (
        "Speculative decoding is provably lossless. The rejection rule makes the "
        "output distribution identical to the target model's, only faster."
    )
    units = distiller._distill_heuristic("5.5 Provably Lossless", text)
    assert len(units) == 1
    assert "lossless" in units[0].lower()


def test_distill_emits_topic_unit_for_multisection_part(tmp_path: Path) -> None:
    # A Part with two sub-sections whose answers are split → a topic unit merges them.
    src = tmp_path / "doc.md"
    src.write_text(
        "# Doc\n\n# Part 1 — Quantization\n\n"
        "## 1.3 INT4 cost\n\nINT4 quantization costs about one to three percent of "
        "model quality for a four times smaller footprint.\n\n"
        "## 1.9 Where it degrades\n\nQuantization degrades most on multi-step math "
        "and reasoning tasks, far less on factual recall.\n",
        encoding="utf-8",
    )
    out = tmp_path / "doc.distilled.md"
    stats = distiller.distill(src, out, backend="heuristic", mode="consolidated")
    assert stats["topic_units"] >= 1
    body = out.read_text(encoding="utf-8")
    # the topic unit for Part 1 should carry BOTH the cost and the degrade content
    topic = [b for b in body.split("## ") if b.startswith("Topic — Part 1")][0]
    assert "percent" in topic and "degrades" in topic


def test_distill_writes_markdown_with_provenance(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text(
        "# Doc\n\n## Section A\n\nThis section explains that quantization trades "
        "precision for a smaller memory footprint on device.\n",
        encoding="utf-8",
    )
    out = tmp_path / "doc.distilled.md"
    stats = distiller.distill(src, out, backend="heuristic")
    assert stats["units"] >= 1
    body = out.read_text(encoding="utf-8")
    assert "_Source: doc.md ›" in body  # provenance pointer present
    assert "## Section A" in body


# --- local backend (F-702 v1) ----------------------------------------------
class _StubGenerator:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, max_tokens: int = 0) -> str:
        self.prompts.append(prompt)
        return self.reply


def _local_with(reply: str) -> "local.LocalDistiller":
    """A LocalDistiller with a stub generator — no model load, no network."""
    d = local.LocalDistiller.__new__(local.LocalDistiller)
    d.model_path = Path("/nonexistent.gguf")
    d._generator = _StubGenerator(reply)  # type: ignore[assignment]
    d.stats = {"model": 0, "rejected": 0, "empty": 0}
    return d


SECTION = (
    "| Level | Strength |\n|---|---|\n| Logit | Strongest |\n| Text | Universal |\n\n"
    "The tokenizer constraint is the practical fork: different vocabularies make "
    "logit-level transfer impossible, so cross-family distillation is text-level."
)


def test_local_distill_returns_model_answer() -> None:
    d = _local_with("There are two levels: logit-level (strongest) and text-level (universal).")
    units = d.distill_section("2.4 Three Levels", SECTION)
    assert units == ["There are two levels: logit-level (strongest) and text-level (universal)."]
    # the model saw the RAW section — table intact
    assert "| Logit | Strongest |" in d._generator.prompts[0]  # type: ignore[attr-defined]


def test_local_distill_skips_thin_sections_without_calling_model() -> None:
    d = _local_with("should never be called")
    assert d.distill_section("Nav", "too short") == []
    assert d._generator.prompts == []  # type: ignore[attr-defined]


def test_local_distill_falls_back_to_heuristic_on_refusal() -> None:
    d = _local_with("NONE")
    units = d.distill_section("2.4 Three Levels", SECTION)
    # section has real prose → heuristic floor, not empty
    assert units and "tokenizer constraint" in units[0]
    assert d.stats["empty"] == 1


# The reasoning model narrates its task instead of answering on a majority of
# sections; shipping that text poisons retrieval. These are real outputs observed
# in the 2026-07-22 run (see tests/eval/corpus_calibration_2026-07-22.md).
NARRATION_SAMPLES = [
    "Okay, I need to write a complete, self-contained answer that captures everything.",
    "I will now carefully analyze this entire SECTION and extract every key fact.",
    "The user wants a complete, self-contained answer from the given SECTION text.",
    "The following is a complete, self-contained documentation section for a speaker.",
    "A speaker-friendly, self-contained explanation of the 'Finding What to Cut' section.",
    "The entire section describes a multi-layered model architecture.",
    "THE SECTION TITLE IS '2.5 Reasoning-Trace Distillation' AND IT CONTAINS A CODE BLOCK.",
]

REAL_ANSWER_SAMPLES = [
    "A weight functions as a signed volume knob that multiplies an input signal.",
    "INT4 quantization costs about one to three percent of quality for a 4x smaller model.",
    "Router collapse happens when a Mixture-of-Experts router sends most tokens to few experts.",
    "The loss is the KL divergence between the teacher and student distributions.",
    "Speculative decoding is provably lossless because the rejection rule preserves the "
    "target model's output distribution exactly.",
]


@pytest.mark.parametrize("sample", NARRATION_SAMPLES)
def test_meta_detector_flags_task_narration(sample: str) -> None:
    assert local.looks_like_meta(sample)


@pytest.mark.parametrize("sample", REAL_ANSWER_SAMPLES)
def test_meta_detector_passes_real_answers(sample: str) -> None:
    assert not local.looks_like_meta(sample)


def test_local_distill_rejects_narration_and_counts_it() -> None:
    d = _local_with("Okay, I need to write a complete, self-contained answer about levels.")
    units = d.distill_section("2.4 Three Levels", SECTION)
    assert units and "tokenizer constraint" in units[0]  # heuristic floor, not the narration
    assert "I need to" not in units[0]
    assert d.stats == {"model": 0, "rejected": 1, "empty": 0}


def test_local_distill_skips_heading_only_section_without_calling_model() -> None:
    # An "# Part 5 — ..." divider clears the raw-word guard on heading length alone;
    # the model then narrates instead of answering. Skip it at source.
    d = _local_with("This section is titled Part 5 but contains only the heading.")
    heading = "Part 5 — Speculative Decoding: Small Model Proposes, Big Model Disposes"
    assert d.distill_section(heading, f"# {heading}") == []
    assert d._generator.prompts == []  # type: ignore[attr-defined]
    assert d.stats == {"model": 0, "rejected": 0, "empty": 0}


def test_local_distill_still_runs_on_table_heavy_section() -> None:
    # Guard must not catch table-heavy sections: they clean down to almost nothing
    # but are exactly what the model is for (prose-ifying the table).
    d = _local_with("Logit-level distillation is strongest but requires a shared tokenizer.")
    table = "## 2.4 Levels\n\n| Level | Strength |\n|---|---|\n| Logit | Strongest |\n"
    assert d.distill_section("2.4 Levels", table) != []
    assert d.stats["model"] == 1


def test_local_distill_counts_accepted_model_output() -> None:
    d = _local_with(
        "Logit-level distillation transfers full dark knowledge but needs one tokenizer."
    )
    d.distill_section("2.4 Three Levels", SECTION)
    assert d.stats == {"model": 1, "rejected": 0, "empty": 0}


def test_local_backend_unavailable_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "_instance", _local_with("x"))
    with pytest.raises(RuntimeError, match="local backend needs the generation model"):
        distiller.distill(Path("/x.md"), Path("/y.md"), backend="local")


# --- cloud output budget (truncation bug) ----------------------------------
def test_output_budget_scales_with_input_for_consolidated() -> None:
    # A whole-Part topic unit (MAX_TOPIC_CHARS) needs far more than the old flat
    # 900 — that truncated the JSON mid-string and killed every topic unit.
    assert distiller.max_output_tokens(distiller.MAX_TOPIC_CHARS, consolidated=True) > 900
    assert distiller.max_output_tokens(600, consolidated=True) == 1500  # floor for small
    assert distiller.max_output_tokens(10**6, consolidated=True) <= 8000  # ceiling
    # A consolidated unit restates its input, so the budget must exceed the
    # input's own token count (~chars/3.5) — the bug was budgeting under it.
    for chars in (2745, 3179, 4154, 6144):  # the Parts that lost their topic unit
        assert distiller.max_output_tokens(chars, consolidated=True) > chars / 3.5


def test_output_budget_flat_for_atomic() -> None:
    # Atomic units are 1-5 short statements regardless of section size.
    assert distiller.max_output_tokens(500, consolidated=False) == 700
    assert distiller.max_output_tokens(distiller.MAX_TOPIC_CHARS, consolidated=False) == 700


def test_cloud_truncation_returns_empty_not_a_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A max_tokens stop must be reported as truncation, not surface as a
    JSONDecodeError on the half-written string."""

    class _Resp:
        stop_reason = "max_tokens"
        content = [type("B", (), {"type": "text", "text": '{"answer": "half a sent'})()]

    class _Messages:
        def create(self, **kwargs: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(
        distiller.cloud, "get_client", lambda: type("C", (), {"messages": _Messages()})()
    )
    text = " ".join(["word"] * 200)
    assert distiller._distill_cloud("Part 1 — Big", text, "consolidated") == []


# --- backend guard rails ---------------------------------------------------
def test_distill_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown distiller backend"):
        distiller.distill(tmp_path / "x.md", tmp_path / "y.md", backend="quantum")


def test_distill_cloud_fails_fast_without_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No credential → distill() must abort before parsing/making any API call.
    monkeypatch.setattr(distiller.cloud, "credential_hint", lambda: "no key")
    with pytest.raises(RuntimeError, match="cloud backend needs a credential"):
        distiller.distill(tmp_path / "x.md", tmp_path / "y.md", backend="cloud")


def test_lab_wrapper_reexports_distiller() -> None:
    # scripts/lab/distiller.py is a thin wrapper — same objects, not copies.
    from scripts.lab import distiller as lab_distiller

    assert lab_distiller.distill is distiller.distill
    assert lab_distiller.MAX_UNIT_WORDS == distiller.MAX_UNIT_WORDS


# --- CLI overwrite guard ---------------------------------------------------
def test_cli_refuses_to_overwrite_a_different_backends_corpus(tmp_path: Path) -> None:
    from scripts.lab.distiller import _guard_overwrite, _meta_path

    out = tmp_path / "c.distilled.md"
    out.write_text("cloud corpus", encoding="utf-8")
    _meta_path(out).write_text('{"backend": "cloud", "units": 74}', encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _guard_overwrite(out, "local", force=False)

    _guard_overwrite(out, "local", force=True)  # --force is the escape hatch
    _guard_overwrite(out, "cloud", force=False)  # same backend = ordinary re-run


def test_cli_overwrite_guard_is_silent_without_provenance(tmp_path: Path) -> None:
    from scripts.lab.distiller import _guard_overwrite

    out = tmp_path / "c.distilled.md"
    out.write_text("corpus with no meta", encoding="utf-8")
    _guard_overwrite(out, "local", force=False)  # unknown provenance → don't block


# Document-referential framing observed in ~52% of cloud units. A unit that says
# "Section 1.3, titled X, explains ..." describes the document instead of being
# the answer, so it is not borrowable — the same contract violation as narration.
FRAMING_SAMPLES = [
    'Section 1.3, titled "Rounding to a Codebook — INT4 First," explains quantization.',
    'Part 9, titled "Multi-Token Prediction," is a section heading with no body.',
    "Section 4.5, titled The Trade Curve, describes an accuracy-versus-tokens curve.",
]


@pytest.mark.parametrize("sample", FRAMING_SAMPLES)
def test_meta_detector_flags_document_referential_framing(sample: str) -> None:
    assert local.looks_like_meta(sample)


def test_all_distiller_prompts_forbid_source_references() -> None:
    # The contract is enforced by a regex at the output; the prompts must state it
    # at the input too, for every backend.
    from lib.corpus.distiller import _CLOUD_SYSTEM, _CLOUD_SYSTEM_CONSOLIDATED
    from lib.corpus.local import _PROMPT

    for prompt in (_CLOUD_SYSTEM, _CLOUD_SYSTEM_CONSOLIDATED, _PROMPT):
        assert "Never refer to the source document" in prompt
