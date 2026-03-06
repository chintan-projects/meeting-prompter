"""Text filters — hallucination detection, noise filtering, normalization.

Extracted from coach.py. Used by the orchestrator to clean transcribed text
before feeding it into the trigger engine and generation pipeline.
"""
import re
from typing import List

# Common hallucination starters from LFM2-Audio on noise/silence
_HALLUCINATION_STARTERS: List[str] = [
    "i don't know what",
    "i'm not sure what",
    "she chose",
    "he chose",
    "they chose",
    "it's just the",
    "it's going to be",
    "that's going to be",
    "you're going to",
    "we're going to",
    "the one that was",
    "the reason why",
    "i think it's",
    "i think that",
    "i guess",
    "i suppose",
    "maybe it's",
    "perhaps it's",
    "it seems like",
    "it looks like",
    "sort of",
    "kind of like",
]

_THIRD_PERSON_STARTERS: List[str] = ["she ", "he ", "they ", "it was ", "there was "]

_VAGUE_QUESTIONS: List[str] = [
    "can you explain to me",
    "can you tell me",
    "can you help me",
    "tell me about",
    "explain to me",
    "what do you mean",
    "what does that mean",
]

_NOISE_PHRASES: List[str] = [
    "yeah", "yeah yeah", "yeah yeah yeah",
    "um", "uh", "uh huh", "um um",
    "okay", "ok", "oh", "oh well",
    "i don't know", "i dunno",
    "and then", "to that one", "so",
    "a", "the", "is it", "it is",
    "hmm", "hm", "ah", "eh",
    "right", "right right",
    "sure", "sure sure",
    "well", "well well",
    "you know", "like",
]

_FILLER_WORDS = frozenset({
    "yeah", "um", "uh", "okay", "ok", "oh", "well", "so", "like",
    "right", "hmm", "hm", "ah", "eh", "a", "the", "and", "then",
    "i", "don't", "know", "just", "to", "that", "one", "it", "is",
})

_QUESTION_STARTERS: List[str] = [
    "how", "what", "why", "when", "where", "who", "which",
    "can", "could", "would", "help", "tell", "explain", "does", "do", "is", "are",
]

_MISHEARING_REPLACEMENTS = [
    (r"\bL\s+Those\b", "Liquid"),
    (r"\bLiquid\s+AI\s+Liquid\s+AI\b", "Liquid AI"),
    (r"\bliquid\s+liquid\b", "Liquid"),
]


def is_hallucination(text: str) -> bool:
    """Detect LFM2-Audio hallucination patterns.

    The model produces predictable starters when given noise/silence.
    Also detects third-person narration and vague standalone questions.
    """
    text_lower = text.lower().strip()

    for starter in _HALLUCINATION_STARTERS:
        if text_lower.startswith(starter):
            return True

    for starter in _THIRD_PERSON_STARTERS:
        if text_lower.startswith(starter):
            return True

    text_clean = text_lower.rstrip("?.,!")
    if text_clean in _VAGUE_QUESTIONS:
        return True

    # Repetitive/circular phrases (hallucination symptom)
    words = text_lower.split()
    if len(words) >= 6:
        for i in range(len(words) - 5):
            seq = " ".join(words[i : i + 3])
            rest = " ".join(words[i + 3 :])
            if seq in rest:
                return True

    return False


def is_noise(text: str) -> bool:
    """Check if text is filler words, noise, or hallucination."""
    text_lower = text.lower().strip()
    words = text_lower.split()

    if len(words) < 3:
        return True

    if is_hallucination(text):
        return True

    text_clean = text_lower.rstrip(".,?!")
    if text_clean in _NOISE_PHRASES:
        return True

    meaningful = [w for w in words if w.rstrip(".,?!") not in _FILLER_WORDS]
    if len(meaningful) < 2:
        return True

    return False


def normalize_text(text: str) -> str:
    """Light normalization — fix duplicates, mishearings, punctuation.

    Preserves content, only fixes obvious issues.
    """
    result = text

    # Fix consecutive duplicate words
    result = re.sub(r"\b(\w+)\s+\1\b", r"\1", result, flags=re.IGNORECASE)

    # Fix known mishearings
    for pattern, replacement in _MISHEARING_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean whitespace
    result = re.sub(r"\s+", " ", result).strip()

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

    # Add ? if clearly a question
    if result and not result.endswith("?") and not result.endswith("."):
        if any(result.lower().startswith(w) for w in _QUESTION_STARTERS):
            result = result.rstrip(".,!") + "?"

    return result
