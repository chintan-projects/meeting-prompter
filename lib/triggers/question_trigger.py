"""Question trigger — detects direct questions in transcribed speech.

Ports the scoring logic from the original question_detector.py with the same
24 patterns, 36 keywords, and multi-factor scoring algorithm. Includes
rhetorical question suppression (F-201) to filter confirmations, self-answers,
tag questions, and rhetorical forms.
"""
import re
import time
from typing import Optional

from lib.config import TriggerConfig
from .types import Trigger, TriggerType

# Question patterns (ported from question_detector.py lines 7-24)
QUESTION_PATTERNS = [
    r'\b(what|how|why|when|where|who|which|whose|whom)\b.*\?',
    r'\b(what|how|why|when|where|who|which|whose|whom)\s+'
    r'(?:is|are|do|does|did|can|could|would|will|should|has|have|had)\b',
    r'\b(is|are|do|does|did|can|could|would|will|should|has|have|had)\s+'
    r'(?:it|this|that|there|you|we|they|the)\b',
    r'\b(can you|could you|would you|tell me|explain|describe)\b',
    r'\b(what about|how about|what if)\b',
    r'\b(how does|how do|how can|how would)\b',
    r"\b(what's the|what is the|what are the)\b",
    r'\b(does it|can it|will it|is it)\b',
    r'\b(do you|can you|could you)\s+(?:support|offer|provide|have|integrate)\b',
]

# Domain keywords (ported from question_detector.py lines 27-36)
QUESTION_KEYWORDS = [
    'pricing', 'cost', 'price', 'license', 'subscription',
    'integrate', 'integration', 'api', 'sdk', 'compatibility',
    'security', 'privacy', 'compliance', 'gdpr', 'hipaa', 'soc2',
    'performance', 'latency', 'speed', 'benchmark',
    'support', 'documentation', 'training', 'onboarding',
    'difference', 'compare', 'versus', 'vs', 'better',
    'feature', 'capability', 'limitation', 'roadmap',
    'example', 'demo', 'proof', 'case study',
]

# Fragment patterns that should be rejected
_FRAGMENT_PATTERNS = [
    r'^can you tell me\??$',
    r'^tell me\??$',
    r'^can you explain\??$',
    r'^what about\??$',
    r'^how about\??$',
    r'^okay,?\s*(can you|tell me)?\??$',
    r'^so,?\s*(tell me|can you)?\??$',
]

_INCOMPLETE_ENDINGS = frozenset([
    'the', 'a', 'an', 'to', 'of', 'for', 'with', 'about',
    'how', 'what', 'is', 'are', 'does', 'do', 'can', 'could', 'would',
])

# ─── Rhetorical / confirmation suppression (F-201) ──────────────────────

# Tag questions appended to statements: "...right?", "...isn't it?"
_TAG_PATTERNS = [
    re.compile(r',\s*(?:right|okay|ok|yeah|no|huh|eh)\s*\??\s*$'),
    re.compile(r',\s*(?:isn\'?t it|aren\'?t they|don\'?t you think)\s*\??\s*$'),
    re.compile(r',\s*(?:isn\'?t that right|don\'?t you agree|you know)\s*\??\s*$'),
]

# Rhetorical form patterns — answers implied by structure
_RHETORICAL_PATTERNS = [
    re.compile(
        r'^(?:don\'?t|doesn\'?t|didn\'?t|won\'?t|wouldn\'?t|isn\'?t|aren\'?t'
        r'|can\'?t|couldn\'?t)\s+\w+\s+already\b'
    ),
    re.compile(r'^isn\'?t\s+it\s+obvious\b'),
    re.compile(r'^why\s+would\s+anyone\b'),
    re.compile(r'^who\s+(?:cares|needs|wants)\b.*\?'),
    re.compile(r'^what\'?s\s+the\s+point\s+of\b'),
    re.compile(r'^how\s+hard\s+can\s+it\s+be\b'),
    re.compile(r'^do\s+we\s+really\s+need\b'),
]

# Phrases that signal self-answering after a question mark
_SELF_ANSWER_STARTS = (
    'yeah', 'yes', 'no', 'nah', 'nope',
    'i think so', 'i think', 'i believe', 'i guess', 'i mean',
    'probably', 'definitely', 'absolutely', 'of course', 'sure',
    'right', 'exactly', 'correct', 'well',
)


def _is_tag_question(text: str) -> bool:
    """Detect tag questions appended to statements (', right?', ', isn't it?')."""
    for pattern in _TAG_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _is_self_answered(text: str) -> bool:
    """Detect self-answering: 'Can we X? Yeah.' / 'Is that Y? I think so.'"""
    q_idx = text.find('?')
    if q_idx < 0:
        return False
    after = text[q_idx + 1:].strip().lower()
    if not after:
        return False
    for phrase in _SELF_ANSWER_STARTS:
        if after.startswith(phrase):
            return True
    return False


def _is_rhetorical(text: str) -> bool:
    """Detect rhetorical forms where the answer is implied by structure."""
    for pattern in _RHETORICAL_PATTERNS:
        if pattern.search(text):
            return True
    return False


class QuestionTrigger:
    """Detects direct questions using pattern matching and keyword scoring."""

    def __init__(self, config: TriggerConfig) -> None:
        self.min_confidence = config.question_score_threshold

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]:
        """Score text as a question. Returns Trigger if above threshold."""
        score = score_question(text)
        if score >= self.min_confidence:
            return Trigger(
                type=TriggerType.QUESTION,
                text=text,
                confidence=score,
                source_context=conversation_context,
                timestamp=time.time(),
            )
        return None


def score_question(sentence: str) -> float:
    """Score how likely a sentence is a complete question (0.0-1.0).

    Multi-factor scoring:
    - Question mark: +0.5
    - Pattern match: +0.3
    - Domain keywords: +0.1 per keyword (max 0.3)
    - Question word at start: +0.1 to +0.2
    - Length bonus (7+ words): +0.1
    """
    sentence_lower = sentence.lower().strip()
    words = sentence_lower.split()
    score = 0.0

    if len(words) < 5:
        return 0.0

    # Reject fragments with incomplete endings
    last_word = words[-1].rstrip('?.,!') if words else ""
    if last_word in _INCOMPLETE_ENDINGS:
        return 0.0

    # Reject bare fragment patterns
    for pattern in _FRAGMENT_PATTERNS:
        if re.match(pattern, sentence_lower):
            return 0.0

    # Rhetorical / confirmation suppression (F-201)
    if _is_tag_question(sentence_lower):
        return 0.0
    if _is_self_answered(sentence_lower):
        return 0.0
    if _is_rhetorical(sentence_lower):
        return 0.0

    if '?' in sentence:
        score += 0.5

    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, sentence_lower):
            score += 0.3
            break

    keyword_count = sum(1 for kw in QUESTION_KEYWORDS if kw in sentence_lower)
    if keyword_count > 0:
        score += min(0.3, keyword_count * 0.1)

    first_word = words[0] if words else ""
    if first_word in ('what', 'how', 'why', 'when', 'where', 'who', 'which'):
        score += 0.2
    elif first_word in ('can', 'could', 'would', 'does', 'do', 'is', 'are'):
        score += 0.1

    if len(words) >= 7:
        score += 0.1

    return min(1.0, score)
