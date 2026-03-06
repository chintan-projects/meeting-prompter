"""Mode-aware generation — prompt templates and trigger-routed generation."""
from .generator import ModeAwareGenerator
from .types import GenerationResult

__all__ = ["ModeAwareGenerator", "GenerationResult"]
