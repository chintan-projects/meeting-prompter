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
        method: How the answer was produced ("hybrid", "extraction", "no_match").
        latency_ms: Generation time in milliseconds.
        source: Brief description of the source (e.g. "ColBERT top-3").
    """

    answer: str
    trigger_type: TriggerType
    confidence: float
    method: str
    latency_ms: float = 0.0
    source: str = ""
