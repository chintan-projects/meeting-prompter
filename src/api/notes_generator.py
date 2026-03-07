"""Post-meeting structured notes generator.

After a session ends, runs LFM2.5-Instruct over the full transcript
to produce structured meeting notes. When speaker data is available
(dual-stream: You/Others), generates speaker-attributed notes with
perspective-grouped sections and attributed action items.

Accepts optional meeting context (agenda, participants) and trigger
history (questions asked, alerts fired) to enrich the output.
"""
import logging
import time
from typing import List, Optional, Tuple

from lib.conversation.meeting_context import MeetingContext
from lib.rag_generator import RAGAnswerGenerator

logger = logging.getLogger(__name__)

# --- Generic (no speaker data) prompts ---

NOTES_SYSTEM = (
    "You are a meeting notes assistant. Generate structured meeting notes from "
    "the transcript. Use ONLY information present in the transcript. "
    "Be concise and actionable."
)

NOTES_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Generate structured meeting notes from this transcript.
{context_section}
{key_moments_section}
TRANSCRIPT:
{transcript}

Format your response as:
## Summary
(2-3 sentence overview)

## Key Decisions
- (bullet points of decisions made)

## Action Items
- [ ] (actionable tasks with owners if mentioned)

## Follow-ups
- (topics that need further discussion)<|im_end|>
<|im_start|>assistant
"""

# --- Speaker-aware prompts ---

SPEAKER_NOTES_SYSTEM = (
    "You are a meeting notes assistant. Generate speaker-attributed notes. "
    "Attribute action items and decisions to speakers. Be concise."
)

SPEAKER_NOTES_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Generate structured meeting notes from this meeting.
{context_section}
{key_moments_section}
YOUR STATEMENTS:
{your_statements}

OTHERS' STATEMENTS:
{others_statements}

Format your response as:
## Summary
(2-3 sentence overview mentioning both sides)

## Your Key Points
- (your main contributions)

## Others' Key Points
- (what others discussed)

## Action Items
- [ ] [You] (your commitments)
- [ ] [Others] (their commitments)

## Follow-ups
- (topics needing follow-up)<|im_end|>
<|im_start|>assistant
"""

# --- Max transcript chars to fit in context window ---
_MAX_TRANSCRIPT_CHARS = 12000


def _build_context_section(meeting_context: Optional[MeetingContext]) -> str:
    """Build a context preamble from meeting context, or empty string."""
    if not meeting_context:
        return ""
    ctx = meeting_context.as_prompt_context()
    if not ctx.strip():
        return ""
    return f"\nMEETING CONTEXT:\n{ctx}\n"


def _build_key_moments_section(trigger_history: Optional[List[dict]]) -> str:
    """Build a key moments section from trigger history, or empty string."""
    if not trigger_history:
        return ""
    lines: List[str] = []
    for entry in trigger_history:
        ttype = entry.get("trigger_type", "")
        ttext = entry.get("trigger_text", "")
        answer = entry.get("answer", "")
        if ttype == "question" and ttext:
            line = f"- Q: {ttext[:120]}"
            if answer:
                line += f" -> {answer[:120]}"
            lines.append(line)
        elif ttype == "alert" and ttext:
            lines.append(f"- ALERT: \"{ttext[:80]}\" detected")
        elif ttype == "topic_match" and answer:
            lines.append(f"- Topic: {answer[:120]}")
    if not lines:
        return ""
    moments = "\n".join(lines[:15])  # Cap at 15 entries
    return f"\nKEY MOMENTS (questions asked, alerts triggered):\n{moments}\n"


def generate_structured_notes(
    transcript_markdown: str,
    generator: Optional[RAGAnswerGenerator] = None,
    segments: Optional[List[dict]] = None,
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Generate structured meeting notes from transcript.

    Args:
        transcript_markdown: Full merged transcript as markdown.
        generator: LLM instance (optional, for LLM generation).
        segments: Structured segment dicts from export_json() (optional).
            When provided with speaker data, generates speaker-attributed notes.
        meeting_context: Pre-meeting config (agenda, participants, watch words).
        trigger_history: Accumulated trigger results from the session.

    Returns:
        Structured notes as markdown. Falls back to template if no LLM.
    """
    if not transcript_markdown.strip():
        return _empty_template()

    context_section = _build_context_section(meeting_context)
    key_moments_section = _build_key_moments_section(trigger_history)

    # Speaker-aware path: use structured segments when speaker data exists
    if segments and _has_speaker_data(segments):
        your_text, others_text = _build_speaker_grouped_transcript(segments)
        if generator is not None:
            try:
                return _generate_speaker_aware(
                    your_text, others_text, generator,
                    context_section, key_moments_section,
                    meeting_context, trigger_history,
                )
            except Exception as e:
                logger.error("Speaker-aware LLM generation failed: %s", e)
        return _speaker_fallback_template(
            your_text, others_text, meeting_context, trigger_history,
        )

    # Generic path: no speaker data
    if generator is not None:
        try:
            return _generate_with_llm(
                transcript_markdown, generator,
                context_section, key_moments_section,
                meeting_context, trigger_history,
            )
        except Exception as e:
            logger.error("LLM notes generation failed: %s", e)

    return _fallback_template(transcript_markdown, meeting_context, trigger_history)


# --- Speaker-aware helpers ---


def _has_speaker_data(segments: List[dict]) -> bool:
    """Check if segments contain meaningful speaker attribution."""
    return any(seg.get("speaker", "") != "" for seg in segments)


def _build_speaker_grouped_transcript(
    segments: List[dict],
) -> Tuple[str, str]:
    """Split segments into 'You' and 'Others' transcript strings.

    Returns (your_statements, others_statements). Each line is formatted
    as [HH:MM:SS] text. Segments without a speaker go into 'Others'.
    """
    your_lines: List[str] = []
    others_lines: List[str] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        ts = time.strftime("%H:%M:%S", time.localtime(seg.get("timestamp", 0)))
        line = f"[{ts}] {text}"

        if seg.get("speaker") == "You":
            your_lines.append(line)
        else:
            others_lines.append(line)

    your_text = "\n".join(your_lines) or "(no statements recorded)"
    others_text = "\n".join(others_lines) or "(no statements recorded)"
    return your_text, others_text


def _generate_speaker_aware(
    your_statements: str,
    others_statements: str,
    generator: RAGAnswerGenerator,
    context_section: str = "",
    key_moments_section: str = "",
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Generate speaker-attributed notes via LLM (thread-safe)."""
    half_budget = _MAX_TRANSCRIPT_CHARS // 2
    your_trunc = your_statements[:half_budget]
    others_trunc = others_statements[:half_budget]

    prompt = SPEAKER_NOTES_PROMPT.format(
        system=SPEAKER_NOTES_SYSTEM,
        your_statements=your_trunc,
        others_statements=others_trunc,
        context_section=context_section,
        key_moments_section=key_moments_section,
    )

    result = generator.generate_text(prompt, max_tokens=800)
    if not result:
        return _speaker_fallback_template(
            your_statements, others_statements, meeting_context, trigger_history,
        )
    return result


# --- Generic helpers ---


def _generate_with_llm(
    transcript: str,
    generator: RAGAnswerGenerator,
    context_section: str = "",
    key_moments_section: str = "",
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Use LFM2.5-Instruct to generate structured notes (thread-safe)."""
    truncated = transcript[:_MAX_TRANSCRIPT_CHARS]
    prompt = NOTES_PROMPT.format(
        system=NOTES_SYSTEM,
        transcript=truncated,
        context_section=context_section,
        key_moments_section=key_moments_section,
    )

    result = generator.generate_text(prompt, max_tokens=800)
    if not result:
        return _fallback_template(transcript, meeting_context, trigger_history)
    return result


# --- Fallback templates ---


def _fallback_context_header(
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Build optional header for fallback templates with context info."""
    parts: List[str] = []
    if meeting_context:
        if meeting_context.title:
            parts.append(f"**Meeting:** {meeting_context.title}")
        if meeting_context.participants:
            parts.append(f"**Participants:** {', '.join(meeting_context.participants)}")
        if meeting_context.agenda_items:
            parts.append("**Agenda:**")
            for item in meeting_context.agenda_items:
                parts.append(f"- {item}")
    if trigger_history:
        questions = [
            e for e in trigger_history if e.get("trigger_type") == "question"
        ]
        alerts = [
            e for e in trigger_history if e.get("trigger_type") == "alert"
        ]
        if questions or alerts:
            parts.append("")
            parts.append("**Key Moments:**")
            for q in questions[:5]:
                parts.append(f"- Q: {q.get('trigger_text', '')[:80]}")
            for a in alerts[:5]:
                parts.append(f"- Alert: \"{a.get('trigger_text', '')[:60]}\"")
    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


def _speaker_fallback_template(
    your_text: str,
    others_text: str,
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Speaker-attributed template when LLM is not available."""
    header = _fallback_context_header(meeting_context, trigger_history)
    return f"""{header}## Summary
(Edit this section with meeting highlights)

## Your Key Points
- (Add your main contributions)

## Others' Key Points
- (Add what others discussed)

## Action Items
- [ ] [You] (Add your action items)
- [ ] [Others] (Add their action items)

## Follow-ups
- (Add topics for follow-up)

---

## Your Statements

{your_text}

## Others' Statements

{others_text}
"""


def _fallback_template(
    transcript: str,
    meeting_context: Optional[MeetingContext] = None,
    trigger_history: Optional[List[dict]] = None,
) -> str:
    """Structured template when LLM is not available."""
    header = _fallback_context_header(meeting_context, trigger_history)
    return f"""{header}## Summary
(Edit this section with meeting highlights)

## Key Decisions
- (Add decisions made during the meeting)

## Action Items
- [ ] (Add action items)

## Follow-ups
- (Add topics for follow-up)

---

## Raw Transcript

{transcript}
"""


def _empty_template() -> str:
    """Template for empty sessions."""
    return """## Summary
No transcript recorded.

## Key Decisions
(none)

## Action Items
(none)

## Follow-ups
(none)
"""
