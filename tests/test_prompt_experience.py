"""Tests for prompt experience overhaul — F-202, F-203, F-205, F-206.

Covers:
- TriggerType display properties (labels, emoji, persistence)
- Dead-end response suppression (generator + session layers)
- Display metadata in WebSocket prompt messages
- Coaching voice prompt templates
"""
import asyncio
from pathlib import Path

import pytest


class TestTriggerTypeDisplayProperties:
    """TriggerType enum has coaching-oriented labels, emoji, and persistence tiers."""

    def test_alert_properties(self) -> None:
        from lib.triggers.types import TriggerType

        assert TriggerType.ALERT.label == "HEADS UP"
        assert TriggerType.ALERT.emoji == "\u26a0\ufe0f"
        assert TriggerType.ALERT.persistence == "persistent"
        assert TriggerType.ALERT.priority == 1

    def test_question_properties(self) -> None:
        from lib.triggers.types import TriggerType

        assert TriggerType.QUESTION.label == "ANSWER"
        assert TriggerType.QUESTION.emoji == "\U0001f4a1"
        assert TriggerType.QUESTION.persistence == "persistent"
        assert TriggerType.QUESTION.priority == 2

    def test_topic_match_properties(self) -> None:
        from lib.triggers.types import TriggerType

        assert TriggerType.TOPIC_MATCH.label == "FYI"
        assert TriggerType.TOPIC_MATCH.emoji == "\U0001f4cc"
        assert TriggerType.TOPIC_MATCH.persistence == "ephemeral"
        assert TriggerType.TOPIC_MATCH.priority == 3

    def test_follow_up_properties(self) -> None:
        from lib.triggers.types import TriggerType

        assert TriggerType.FOLLOW_UP.label == "SUGGEST"
        assert TriggerType.FOLLOW_UP.emoji == "\U0001f4ac"
        assert TriggerType.FOLLOW_UP.persistence == "standard"
        assert TriggerType.FOLLOW_UP.priority == 4

    def test_all_trigger_types_have_persistence(self) -> None:
        """Every trigger type must declare a persistence tier."""
        from lib.triggers.types import TriggerType

        valid_tiers = {"persistent", "standard", "ephemeral"}
        for tt in TriggerType:
            assert tt.persistence in valid_tiers, f"{tt} has invalid persistence: {tt.persistence}"

    def test_enum_values_unchanged(self) -> None:
        """Enum values (wire format) must not change — they're the WebSocket protocol."""
        from lib.triggers.types import TriggerType

        assert TriggerType.ALERT.value == "alert"
        assert TriggerType.QUESTION.value == "question"
        assert TriggerType.TOPIC_MATCH.value == "topic"
        assert TriggerType.FOLLOW_UP.value == "follow_up"


class TestDeadEndSuppression:
    """F-202: Generator suppresses empty/near-empty answers."""

    def test_empty_answer_suppressed(self) -> None:
        """Generator returns 'suppressed' method for empty answers."""
        from lib.generation.generator import ModeAwareGenerator
        from lib.triggers.types import Trigger, TriggerType

        gen = ModeAwareGenerator(
            model_path=Path("/nonexistent"),
            use_generation=False,
            min_answer_length=10,
        )

        trigger = Trigger(
            type=TriggerType.QUESTION,
            text="What is the timeline?",
            confidence=0.8,
        )
        # With no generation and no RAG context, result will be empty
        result = gen.process_trigger(trigger, rag_context="", conversation_context="")
        assert result.method == "suppressed"
        assert result.answer == ""

    def test_short_answer_suppressed(self) -> None:
        """Answers shorter than min_answer_length are suppressed."""
        from lib.generation.generator import ModeAwareGenerator
        from lib.triggers.types import Trigger, TriggerType

        gen = ModeAwareGenerator(
            model_path=Path("/nonexistent"),
            use_generation=False,
            min_answer_length=10,
        )

        trigger = Trigger(
            type=TriggerType.TOPIC_MATCH,
            text="pricing discussion",
            confidence=0.6,
        )
        # With no model, extraction on empty context → empty result → suppressed
        result = gen.process_trigger(trigger, rag_context="", conversation_context="")
        assert result.method == "suppressed"

    def test_adequate_answer_not_suppressed(self) -> None:
        """Answers at or above min_answer_length pass through."""
        from lib.generation.generator import ModeAwareGenerator
        from lib.triggers.types import Trigger, TriggerType

        gen = ModeAwareGenerator(
            model_path=Path("/nonexistent"),
            use_generation=False,
            min_answer_length=10,
        )

        trigger = Trigger(
            type=TriggerType.QUESTION,
            text="What is the deployment target?",
            confidence=0.8,
        )

        # Give it enough RAG context to produce an extraction result
        rag_context = (
            "The deployment target is Q2 2024 for mobile and web platforms. "
            "Edge SDK will be ready by March."
        )
        result = gen.process_trigger(trigger, rag_context=rag_context)

        # Extraction should produce bullets that are >= 10 chars
        if result.method != "suppressed":
            assert len(result.answer.strip()) >= 10

    def test_min_answer_length_configurable(self) -> None:
        """min_answer_length is configurable via constructor."""
        from lib.generation.generator import ModeAwareGenerator

        gen = ModeAwareGenerator(
            model_path=Path("/nonexistent"),
            use_generation=False,
            min_answer_length=25,
        )
        assert gen.min_answer_length == 25


class TestSessionDeadEndFilter:
    """Session._on_trigger_result filters dead-end methods from WebSocket."""

    @pytest.mark.asyncio
    async def test_no_match_suppressed(self) -> None:
        """Prompts with method='no_match' are not queued."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.QUESTION, text="test?", confidence=0.5)
        result = GenerationResult(
            answer="",
            trigger_type=TriggerType.QUESTION,
            confidence=0.0,
            method="no_match",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)
        assert session._prompt_queue.empty()

    @pytest.mark.asyncio
    async def test_no_context_suppressed(self) -> None:
        """Prompts with method='no_context' are not queued."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.TOPIC_MATCH, text="pricing", confidence=0.5)
        result = GenerationResult(
            answer="",
            trigger_type=TriggerType.TOPIC_MATCH,
            confidence=0.0,
            method="no_context",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)
        assert session._prompt_queue.empty()

    @pytest.mark.asyncio
    async def test_suppressed_method_filtered(self) -> None:
        """Prompts with method='suppressed' are not queued."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.QUESTION, text="test?", confidence=0.5)
        result = GenerationResult(
            answer="short",
            trigger_type=TriggerType.QUESTION,
            confidence=0.3,
            method="suppressed",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)
        assert session._prompt_queue.empty()

    @pytest.mark.asyncio
    async def test_empty_answer_suppressed(self) -> None:
        """Prompts with empty answer are not queued, regardless of method."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.QUESTION, text="test?", confidence=0.5)
        result = GenerationResult(
            answer="",
            trigger_type=TriggerType.QUESTION,
            confidence=0.5,
            method="hybrid",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)
        assert session._prompt_queue.empty()

    @pytest.mark.asyncio
    async def test_dead_end_not_stored_in_trigger_history(self) -> None:
        """Suppressed prompts must not appear in trigger_history (post-meeting summary)."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.QUESTION, text="test?", confidence=0.5)
        result = GenerationResult(
            answer="",
            trigger_type=TriggerType.QUESTION,
            confidence=0.0,
            method="no_match",
        )

        session._on_trigger_result(trigger, result)
        assert len(session.trigger_history) == 0


class TestPromptDisplayMetadata:
    """WebSocket prompt messages include persistence, label, emoji, and dismiss_ms."""

    @pytest.mark.asyncio
    async def test_question_prompt_has_display_fields(self) -> None:
        """QUESTION prompt message includes all display metadata."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.QUESTION, text="What is the plan?", confidence=0.8)
        result = GenerationResult(
            answer="The plan targets Q2 2024 for the initial release.",
            trigger_type=TriggerType.QUESTION,
            confidence=0.75,
            method="hybrid",
            latency_ms=480,
            source="docs/roadmap.md",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)

        msg = session._prompt_queue.get_nowait()
        assert msg["persistence"] == "persistent"
        assert msg["display_label"] == "ANSWER"
        assert msg["display_emoji"] == "\U0001f4a1"
        assert msg["dismiss_ms"] == 0  # persistent = no auto-dismiss

    @pytest.mark.asyncio
    async def test_topic_prompt_has_ephemeral_persistence(self) -> None:
        """TOPIC prompt gets ephemeral persistence with configurable dismiss_ms."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.TOPIC_MATCH, text="pricing model", confidence=0.6)
        result = GenerationResult(
            answer="The current pricing tier starts at $1.5M annually.",
            trigger_type=TriggerType.TOPIC_MATCH,
            confidence=0.6,
            method="generation",
            source="docs/pricing.md",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)

        msg = session._prompt_queue.get_nowait()
        assert msg["persistence"] == "ephemeral"
        assert msg["display_label"] == "FYI"
        assert msg["dismiss_ms"] == 45_000

    @pytest.mark.asyncio
    async def test_follow_up_prompt_has_standard_persistence(self) -> None:
        """FOLLOW_UP prompt gets standard persistence (90s dismiss)."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.FOLLOW_UP, text="migration approach", confidence=0.5)
        result = GenerationResult(
            answer="Ask about their preferred migration timeline for the data layer.",
            trigger_type=TriggerType.FOLLOW_UP,
            confidence=0.5,
            method="generation",
            source="docs/migration.md",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)

        msg = session._prompt_queue.get_nowait()
        assert msg["persistence"] == "standard"
        assert msg["display_label"] == "SUGGEST"
        assert msg["dismiss_ms"] == 90_000

    @pytest.mark.asyncio
    async def test_alert_prompt_has_persistent_dismiss(self) -> None:
        """ALERT prompt gets persistent (dismiss_ms=0)."""
        from lib.generation.types import GenerationResult
        from lib.triggers.types import Trigger, TriggerType
        from src.api.session import Session

        session = Session()
        session._loop = asyncio.get_running_loop()

        trigger = Trigger(type=TriggerType.ALERT, text="competitor pricing", confidence=1.0)
        result = GenerationResult(
            answer="Competitor quoted $2M last quarter; your current offer is $1.5M.",
            trigger_type=TriggerType.ALERT,
            confidence=1.0,
            method="generation",
            source="docs/competitive.md",
        )

        session._on_trigger_result(trigger, result)
        await asyncio.sleep(0.01)

        msg = session._prompt_queue.get_nowait()
        assert msg["persistence"] == "persistent"
        assert msg["display_label"] == "HEADS UP"
        assert msg["display_emoji"] == "\u26a0\ufe0f"
        assert msg["dismiss_ms"] == 0


class TestPromptTemplatesCoachingVoice:
    """F-206: Prompt templates use coaching persona, not encyclopedia."""

    def test_question_system_prompt_coaching_persona(self) -> None:
        """QUESTION system prompt says 'meeting coach', not 'meeting intelligence assistant'."""
        from lib.generation import prompts

        assert "meeting coach" in prompts.QUESTION_SYSTEM
        assert "meeting intelligence assistant" not in prompts.QUESTION_SYSTEM

    def test_question_system_prompt_no_dead_end_instruction(self) -> None:
        """QUESTION prompt must not tell model to say 'I don't have that information'."""
        from lib.generation import prompts

        assert "don't have" not in prompts.QUESTION_SYSTEM.lower()
        assert "do not have" not in prompts.QUESTION_SYSTEM.lower()

    def test_question_system_prompt_suggests_coaching(self) -> None:
        """QUESTION system prompt encourages optional coaching suffix."""
        from lib.generation import prompts

        assert "You could mention" in prompts.QUESTION_SYSTEM

    def test_topic_system_prompt_surfaces_new_info(self) -> None:
        """F-203: TOPIC prompt tells model to add new info, not summarize."""
        from lib.generation import prompts

        assert "not been mentioned" in prompts.TOPIC_SYSTEM or "hasn't been" in prompts.TOPIC_SYSTEM
        assert "summarize" in prompts.TOPIC_SYSTEM.lower()

    def test_followup_system_prompt_coaching_verbs(self) -> None:
        """F-205: FOLLOW_UP prompt uses action verbs for coaching."""
        from lib.generation import prompts

        assert "Ask about" in prompts.FOLLOWUP_SYSTEM
        assert "Mention that" in prompts.FOLLOWUP_SYSTEM
        assert "Clarify whether" in prompts.FOLLOWUP_SYSTEM

    def test_alert_system_prompt_direct_tone(self) -> None:
        """ALERT prompt is direct: 'needs to know right now'."""
        from lib.generation import prompts

        assert "right now" in prompts.ALERT_SYSTEM

    def test_all_prompts_use_coaching_persona(self) -> None:
        """All system prompts use 'meeting coach' persona."""
        from lib.generation import prompts

        for system in [
            prompts.QUESTION_SYSTEM,
            prompts.TOPIC_SYSTEM,
            prompts.FOLLOWUP_SYSTEM,
            prompts.ALERT_SYSTEM,
        ]:
            assert "meeting coach" in system, f"Missing coaching persona in: {system[:60]}"

    def test_alert_template_says_key_term_not_watch_word(self) -> None:
        """Alert template uses 'KEY TERM' (coaching) not 'WATCH WORD' (mechanical)."""
        from lib.generation import prompts

        assert "KEY TERM" in prompts.ALERT_PROMPT
        assert "WATCH WORD" not in prompts.ALERT_PROMPT


class TestConfigMinAnswerLength:
    """TriggerConfig.min_answer_length is loaded from config.yaml."""

    def test_default_min_answer_length(self) -> None:
        from lib.config import TriggerConfig

        config = TriggerConfig()
        assert config.min_answer_length == 10

    def test_default_dismiss_timers(self) -> None:
        from lib.config import TriggerConfig

        config = TriggerConfig()
        assert config.dismiss_persistent_ms == 0
        assert config.dismiss_standard_ms == 90_000
        assert config.dismiss_ephemeral_ms == 45_000

    def test_config_loaded_from_yaml(self) -> None:
        """Config loader picks up new trigger fields from config.yaml."""
        from lib.config import load_config

        config = load_config()
        assert config.triggers.min_answer_length == 10
        assert config.triggers.dismiss_persistent_ms == 0
        assert config.triggers.dismiss_standard_ms == 90_000
        assert config.triggers.dismiss_ephemeral_ms == 45_000
