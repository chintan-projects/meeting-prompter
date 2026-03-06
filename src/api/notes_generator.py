"""Post-meeting structured notes generator.

After a session ends, runs LFM2.5-Instruct over the full transcript
to produce: Summary, Key Decisions, Action Items, Follow-ups.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

NOTES_SYSTEM = (
    "You are a meeting notes assistant. Generate structured meeting notes from "
    "the transcript. Use ONLY information present in the transcript. "
    "Be concise and actionable."
)

NOTES_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Generate structured meeting notes from this transcript.

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


def generate_structured_notes(
    transcript_markdown: str,
    generator: Optional[object] = None,
) -> str:
    """Generate structured meeting notes from transcript.

    Args:
        transcript_markdown: Full merged transcript as markdown.
        generator: RAGAnswerGenerator instance (optional, for LLM generation).

    Returns:
        Structured notes as markdown. Falls back to template if no LLM.
    """
    if not transcript_markdown.strip():
        return _empty_template()

    # If we have a generator, use LLM
    if generator is not None:
        try:
            return _generate_with_llm(transcript_markdown, generator)
        except Exception as e:
            logger.error("LLM notes generation failed: %s", e)

    # Fallback: return template with raw transcript
    return _fallback_template(transcript_markdown)


def _generate_with_llm(transcript: str, generator: object) -> str:
    """Use LFM2.5-Instruct to generate structured notes."""
    # Truncate transcript to fit in context window
    max_chars = 12000  # Leave room for prompt + output
    truncated = transcript[:max_chars]

    prompt = NOTES_PROMPT.format(system=NOTES_SYSTEM, transcript=truncated)

    generator.load()  # type: ignore[attr-defined]
    generator._reset_state()  # type: ignore[attr-defined]

    response = generator.llm(  # type: ignore[attr-defined]
        prompt,
        max_tokens=500,
        stop=["<|im_end|>", "<|im_start|>"],
        temperature=0,
        top_p=1.0,
    )

    result = response["choices"][0]["text"].strip()
    if not result:
        return _fallback_template(transcript)

    return result


def _fallback_template(transcript: str) -> str:
    """Structured template when LLM is not available."""
    return f"""## Summary
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
