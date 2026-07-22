"""Generation result types."""

from dataclasses import dataclass

from lib.triggers.types import TriggerType


@dataclass
class GenerationResult:
    """Result from mode-aware generation.

    Attributes:
        answer: Generated answer text.
        trigger_type: Which trigger type produced this result.
        confidence: Confidence score from extraction/retrieval (0.0-1.0).
        method: How the answer was produced ("retrieval", "hybrid", "extraction",
            "no_match"). "retrieval" = borrowable unit, no LLM (F-705/D-08).
        latency_ms: Generation time in milliseconds.
        source: Brief description of the source (e.g. "deployment.md").
        heading: Section heading path of the borrowable unit (provenance).
        source_text: Full borrowable unit for expand-to-source display.
    """

    answer: str
    trigger_type: TriggerType
    confidence: float
    method: str
    latency_ms: float = 0.0
    source: str = ""
    heading: str = ""
    source_text: str = ""
