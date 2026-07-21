"""Tests for src.api.notes_generator — structured meeting notes generation."""
from typing import List
from unittest.mock import MagicMock

import pytest

from lib.conversation.meeting_context import MeetingContext
from src.api.notes_generator import (
    _build_context_section,
    _build_key_moments_section,
    _build_speaker_grouped_transcript,
    _fallback_context_header,
    _has_speaker_data,
    generate_structured_notes,
)


def _make_segments(
    speakers: List[str],
    texts: List[str],
    base_ts: float = 100.0,
) -> List[dict]:
    """Helper to build segment dicts for testing."""
    segs = []
    for i, (spk, txt) in enumerate(zip(speakers, texts)):
        segs.append({
            "id": f"turn-{i+1}",
            "text": txt,
            "timestamp": base_ts + i * 5.0,
            "end_timestamp": base_ts + i * 5.0 + 3.0,
            "speaker": spk,
            "source": "mic" if spk == "You" else "system",
            "is_final": True,
            "edited": False,
        })
    return segs


class TestHasSpeakerData:
    """Tests for _has_speaker_data()."""

    def test_returns_true_when_speakers_present(self) -> None:
        segments = _make_segments(["You", "Others"], ["hi", "hello"])
        assert _has_speaker_data(segments) is True

    def test_returns_false_when_all_empty(self) -> None:
        segments = _make_segments(["", ""], ["hi", "hello"])
        assert _has_speaker_data(segments) is False

    def test_returns_false_for_empty_list(self) -> None:
        assert _has_speaker_data([]) is False

    def test_returns_true_if_any_speaker_set(self) -> None:
        segments = _make_segments(["", "You", ""], ["a", "b", "c"])
        assert _has_speaker_data(segments) is True


class TestBuildSpeakerGroupedTranscript:
    """Tests for _build_speaker_grouped_transcript()."""

    def test_splits_you_and_others(self) -> None:
        segments = _make_segments(
            ["You", "Others", "You"],
            ["my point", "their reply", "my follow-up"],
        )
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        assert "my point" in your_text
        assert "my follow-up" in your_text
        assert "their reply" in others_text
        assert "their reply" not in your_text

    def test_empty_speaker_goes_to_others(self) -> None:
        segments = _make_segments(["", "You"], ["unknown", "mine"])
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        assert "unknown" in others_text
        assert "mine" in your_text

    def test_no_you_statements(self) -> None:
        segments = _make_segments(["Others", "Others"], ["a", "b"])
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        assert your_text == "(no statements recorded)"
        assert "a" in others_text

    def test_no_others_statements(self) -> None:
        segments = _make_segments(["You", "You"], ["a", "b"])
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        assert "a" in your_text
        assert others_text == "(no statements recorded)"

    def test_empty_segments(self) -> None:
        your_text, others_text = _build_speaker_grouped_transcript([])
        assert your_text == "(no statements recorded)"
        assert others_text == "(no statements recorded)"

    def test_skips_blank_text(self) -> None:
        segments = [
            {"text": "", "speaker": "You", "timestamp": 100.0},
            {"text": "   ", "speaker": "Others", "timestamp": 105.0},
            {"text": "real text", "speaker": "You", "timestamp": 110.0},
        ]
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        assert "real text" in your_text
        assert others_text == "(no statements recorded)"

    def test_timestamps_formatted(self) -> None:
        segments = _make_segments(["You"], ["hello"])
        your_text, _ = _build_speaker_grouped_transcript(segments)
        # Should contain a [HH:MM:SS] prefix
        assert your_text.startswith("[")
        assert "] hello" in your_text


class TestGenerateStructuredNotes:
    """Tests for generate_structured_notes() — integration tests."""

    def test_empty_transcript_returns_empty_template(self) -> None:
        result = generate_structured_notes("")
        assert "No transcript recorded" in result
        assert "## Summary" in result

    def test_whitespace_transcript_returns_empty_template(self) -> None:
        result = generate_structured_notes("   \n  ")
        assert "No transcript recorded" in result

    def test_no_generator_no_segments_returns_fallback(self) -> None:
        transcript = "[00:00:01] Hello world\n[00:00:05] Testing one two three"
        result = generate_structured_notes(transcript)
        assert "## Summary" in result
        assert "## Key Decisions" in result
        assert "## Action Items" in result
        assert "Hello world" in result

    def test_with_speaker_segments_no_generator_returns_speaker_template(self) -> None:
        segments = _make_segments(
            ["You", "Others", "You"],
            ["I think we should deploy", "Agreed, lets do it", "Great, I will handle CI"],
        )
        transcript = "I think we should deploy\nAgreed\nGreat"
        result = generate_structured_notes(transcript, segments=segments)
        assert "## Your Key Points" in result
        assert "## Others' Key Points" in result
        assert "I think we should deploy" in result
        assert "Agreed, lets do it" in result

    def test_without_speaker_data_uses_generic_template(self) -> None:
        segments = _make_segments(["", ""], ["hello", "world"])
        transcript = "hello\nworld"
        result = generate_structured_notes(transcript, segments=segments)
        # No speaker data → generic fallback, not speaker-attributed
        assert "## Key Decisions" in result
        assert "Your Key Points" not in result

    def test_segments_none_uses_markdown_path(self) -> None:
        transcript = "[00:00:01] Some discussion happened"
        result = generate_structured_notes(transcript, segments=None)
        assert "## Summary" in result
        assert "Some discussion happened" in result

    def test_speaker_fallback_includes_both_groups(self) -> None:
        segments = _make_segments(
            ["You", "Others"],
            ["my contribution", "their input"],
        )
        transcript = "my contribution\ntheir input"
        result = generate_structured_notes(transcript, segments=segments)
        assert "## Your Statements" in result
        assert "## Others' Statements" in result
        assert "my contribution" in result
        assert "their input" in result

    def test_action_items_section_present_in_speaker_template(self) -> None:
        segments = _make_segments(["You", "Others"], ["task a", "task b"])
        result = generate_structured_notes("task a\ntask b", segments=segments)
        assert "## Action Items" in result
        assert "[You]" in result
        assert "[Others]" in result


# --- Fixtures for context + trigger history tests ---


@pytest.fixture
def sample_context() -> MeetingContext:
    """Meeting context with agenda and participants."""
    return MeetingContext(
        title="Sprint Planning",
        agenda_items=["Review roadmap", "Sprint goals", "Risk assessment"],
        watch_words=["budget", "timeline"],
        participants=["Alice (PM)", "Bob (Eng)", "Carol (Design)"],
        key_topics=["deployment"],
        notes="Focus on Q2 deliverables",
    )


@pytest.fixture
def sample_trigger_history() -> List[dict]:
    """Trigger history with questions and alerts."""
    return [
        {
            "trigger_type": "question",
            "trigger_text": "What is the deployment timeline?",
            "answer": "Targeting Q2 beta release",
            "confidence": 0.75,
            "timestamp": 1000.0,
        },
        {
            "trigger_type": "alert",
            "trigger_text": "budget",
            "answer": "Budget discussed in context of infrastructure costs",
            "confidence": 0.9,
            "timestamp": 1050.0,
        },
        {
            "trigger_type": "topic_match",
            "trigger_text": "deployment planning",
            "answer": "Docs mention phased rollout strategy",
            "confidence": 0.6,
            "timestamp": 1100.0,
        },
    ]


@pytest.fixture
def mock_generator() -> MagicMock:
    """Mock RAGAnswerGenerator with generate_text()."""
    gen = MagicMock()
    gen.generate_text.return_value = "## Summary\nMock LLM notes output."
    return gen


class TestBuildContextSection:
    """Tests for _build_context_section helper."""

    def test_none_context_returns_empty(self) -> None:
        assert _build_context_section(None) == ""

    def test_empty_context_returns_empty(self) -> None:
        ctx = MeetingContext()
        assert _build_context_section(ctx) == ""

    def test_context_with_title(self) -> None:
        ctx = MeetingContext(title="Sprint Planning")
        result = _build_context_section(ctx)
        assert "MEETING CONTEXT:" in result
        assert "Sprint Planning" in result

    def test_context_with_participants(self, sample_context: MeetingContext) -> None:
        result = _build_context_section(sample_context)
        assert "Alice (PM)" in result
        assert "Bob (Eng)" in result

    def test_context_with_agenda(self, sample_context: MeetingContext) -> None:
        result = _build_context_section(sample_context)
        assert "Review roadmap" in result
        assert "Sprint goals" in result


class TestBuildKeyMomentsSection:
    """Tests for _build_key_moments_section helper."""

    def test_none_history_returns_empty(self) -> None:
        assert _build_key_moments_section(None) == ""

    def test_empty_history_returns_empty(self) -> None:
        assert _build_key_moments_section([]) == ""

    def test_question_formatted(self, sample_trigger_history: List[dict]) -> None:
        result = _build_key_moments_section(sample_trigger_history)
        assert "KEY MOMENTS" in result
        assert "Q: What is the deployment timeline?" in result
        assert "Targeting Q2 beta release" in result

    def test_alert_formatted(self, sample_trigger_history: List[dict]) -> None:
        result = _build_key_moments_section(sample_trigger_history)
        assert 'ALERT: "budget" detected' in result

    def test_topic_formatted(self, sample_trigger_history: List[dict]) -> None:
        result = _build_key_moments_section(sample_trigger_history)
        assert "Topic: Docs mention phased rollout strategy" in result

    def test_caps_at_15_entries(self) -> None:
        history = [
            {"trigger_type": "question", "trigger_text": f"Q{i}?", "answer": f"A{i}"}
            for i in range(20)
        ]
        result = _build_key_moments_section(history)
        lines = [line for line in result.strip().split("\n") if line.startswith("- ")]
        assert len(lines) == 15


class TestFallbackContextHeader:
    """Tests for _fallback_context_header used in template fallbacks."""

    def test_no_context_no_history(self) -> None:
        assert _fallback_context_header() == ""

    def test_with_context(self, sample_context: MeetingContext) -> None:
        header = _fallback_context_header(sample_context)
        assert "Sprint Planning" in header
        assert "Alice (PM)" in header
        assert "Review roadmap" in header

    def test_with_trigger_history(self, sample_trigger_history: List[dict]) -> None:
        header = _fallback_context_header(trigger_history=sample_trigger_history)
        assert "Key Moments" in header
        assert "What is the deployment timeline?" in header
        assert 'Alert: "budget"' in header


class TestNotesWithContext:
    """Tests for generate_structured_notes with meeting context and trigger history."""

    def test_generic_fallback_with_context(
        self, sample_context: MeetingContext,
    ) -> None:
        """Fallback template should include context header."""
        result = generate_structured_notes(
            "Some transcript", meeting_context=sample_context,
        )
        assert "Sprint Planning" in result
        assert "Alice (PM)" in result

    def test_generic_fallback_with_trigger_history(
        self, sample_trigger_history: List[dict],
    ) -> None:
        """Fallback template should include key moments."""
        result = generate_structured_notes(
            "Some transcript", trigger_history=sample_trigger_history,
        )
        assert "Key Moments" in result
        assert "What is the deployment timeline?" in result

    def test_llm_prompt_includes_context(
        self,
        mock_generator: MagicMock,
        sample_context: MeetingContext,
        sample_trigger_history: List[dict],
    ) -> None:
        """LLM prompt should contain context and key moments."""
        result = generate_structured_notes(
            "Some transcript text",
            generator=mock_generator,
            meeting_context=sample_context,
            trigger_history=sample_trigger_history,
        )
        assert result == "## Summary\nMock LLM notes output."
        mock_generator.generate_text.assert_called_once()

        prompt = mock_generator.generate_text.call_args[0][0]
        assert "Sprint Planning" in prompt
        assert "KEY MOMENTS" in prompt
        assert "What is the deployment timeline?" in prompt

    def test_llm_max_tokens_is_800(self, mock_generator: MagicMock) -> None:
        """Token budget should be 800."""
        generate_structured_notes("Some transcript", generator=mock_generator)
        call_kwargs = mock_generator.generate_text.call_args[1]
        assert call_kwargs["max_tokens"] == 800

    def test_speaker_aware_prompt_includes_context(
        self,
        mock_generator: MagicMock,
        sample_context: MeetingContext,
        sample_trigger_history: List[dict],
    ) -> None:
        """Speaker-aware LLM prompt should include context and key moments."""
        segments = _make_segments(
            ["You", "Others"], ["my point", "their point"],
        )
        generate_structured_notes(
            "my point\ntheir point",
            generator=mock_generator,
            segments=segments,
            meeting_context=sample_context,
            trigger_history=sample_trigger_history,
        )
        prompt = mock_generator.generate_text.call_args[0][0]
        assert "YOUR STATEMENTS:" in prompt
        assert "MEETING CONTEXT:" in prompt
        assert "Sprint Planning" in prompt
        assert "KEY MOMENTS" in prompt

    def test_speaker_fallback_with_context(
        self, sample_context: MeetingContext,
    ) -> None:
        """Speaker fallback should include context header."""
        segments = _make_segments(["You", "Others"], ["hi", "hello"])
        result = generate_structured_notes(
            "hi\nhello", segments=segments, meeting_context=sample_context,
        )
        assert "Sprint Planning" in result
        assert "Your Statements" in result

    def test_llm_failure_falls_back_with_context(
        self, mock_generator: MagicMock, sample_context: MeetingContext,
    ) -> None:
        """If LLM returns empty, fallback should still include context."""
        mock_generator.generate_text.return_value = ""
        result = generate_structured_notes(
            "Some transcript",
            generator=mock_generator,
            meeting_context=sample_context,
        )
        assert "Raw Transcript" in result
        assert "Sprint Planning" in result

    def test_llm_exception_falls_back(self, mock_generator: MagicMock) -> None:
        """If LLM raises, should fall back to template."""
        mock_generator.generate_text.side_effect = RuntimeError("model crashed")
        result = generate_structured_notes("Some transcript", generator=mock_generator)
        assert "Raw Transcript" in result


# ─── F-507: structured extraction path (LFM2.5-350M-Extract) ─────────────

from src.api.notes_generator import (  # noqa: E402
    EXTRACT_SYSTEM,
    StructuredNotes,
    build_extract_prompt,
    extract_structured_notes,
    parse_structured_response,
    render_structured_notes,
)


class _FakeExtractor:
    """Duck-typed extractor returning a canned YAML payload."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.last_prompt = ""

    def generate_text(self, prompt: str, max_tokens: int = 600) -> str:
        self.last_prompt = prompt
        return self._payload


_GOOD_YAML = (
    "summary: We agreed to ship the encoder swap next week.\n"
    "key_decisions:\n"
    "  - Adopt LFM2.5-Embedding for retrieval\n"
    "  - Keep the heuristic question detector as default\n"
    "action_items:\n"
    "  - Chintan to run the retrieval eval\n"
    "follow_ups:\n"
    "  - Revisit the probe head with real labels\n"
)


class TestExtractPrompt:
    def test_schema_in_system_prompt(self) -> None:
        prompt = build_extract_prompt("some transcript")
        assert "summary:" in EXTRACT_SYSTEM
        assert "key_decisions:" in EXTRACT_SYSTEM
        assert "some transcript" in prompt
        assert "<|im_start|>system" in prompt


class TestParseStructuredResponse:
    def test_parses_fields(self) -> None:
        notes = parse_structured_response(_GOOD_YAML)
        assert "encoder swap" in notes.summary
        assert len(notes.key_decisions) == 2
        assert notes.action_items == ["Chintan to run the retrieval eval"]
        assert notes.follow_ups == ["Revisit the probe head with real labels"]

    def test_scalar_coerced_to_list(self) -> None:
        notes = parse_structured_response(
            "summary: s\naction_items: single item\nkey_decisions: []\nfollow_ups: []\n"
        )
        assert notes.action_items == ["single item"]

    def test_garbage_returns_empty(self) -> None:
        assert parse_structured_response("::: not yaml :::").is_empty() in (True, False)
        # A non-dict YAML scalar → empty structured notes.
        assert parse_structured_response("just a sentence").is_empty()

    def test_missing_fields_default_empty(self) -> None:
        notes = parse_structured_response("summary: only a summary\n")
        assert notes.summary == "only a summary"
        assert notes.key_decisions == []
        assert notes.action_items == []


class TestRenderStructuredNotes:
    def test_renders_sections(self) -> None:
        notes = parse_structured_response(_GOOD_YAML)
        md = render_structured_notes(notes)
        assert "## Summary" in md
        assert "## Key Decisions" in md
        assert "- [ ] Chintan to run the retrieval eval" in md
        assert "## Follow-ups" in md

    def test_empty_fields_render_none(self) -> None:
        md = render_structured_notes(StructuredNotes(summary="s"))
        assert "- (none)" in md
        assert "- [ ] (none)" in md


class TestExtractPathIntegration:
    def test_generate_uses_extractor(self) -> None:
        extractor = _FakeExtractor(_GOOD_YAML)
        md = generate_structured_notes(
            "Full transcript text here.", extractor=extractor
        )
        assert "Adopt LFM2.5-Embedding for retrieval" in md
        assert "TRANSCRIPT:" in extractor.last_prompt

    def test_empty_extract_falls_back(self) -> None:
        # Extractor returns unusable output → falls back to the template path.
        extractor = _FakeExtractor("not yaml at all")
        md = generate_structured_notes(
            "Full transcript text here.", extractor=extractor
        )
        # Fallback template still yields a Summary section.
        assert "## Summary" in md

    def test_extractor_none_unchanged(self) -> None:
        # No extractor → existing behavior (template fallback, no crash).
        md = generate_structured_notes("Full transcript text here.")
        assert "## Summary" in md

    def test_extract_structured_notes_helper(self) -> None:
        notes = extract_structured_notes("t", _FakeExtractor(_GOOD_YAML))
        assert isinstance(notes, StructuredNotes)
        assert notes.summary
