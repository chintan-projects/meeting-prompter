"""Topic trigger — detects when discussion matches document content.

Extracts key terms from speech and checks ColBERT for matching content.
Fires when a new topic is detected that has relevant documents, enabling
proactive information surfacing before a question is asked.
"""
import logging
import re
import time
from typing import Any, Dict, Optional, Protocol

from lib.config import TriggerConfig
from .types import Trigger, TriggerType

logger = logging.getLogger(__name__)

# Common stop words to filter out when extracting topics
_STOP_WORDS = frozenset([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
    'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'going', 'think',
    'know', 'like', 'just', 'get', 'got', 'make', 'made', 'really',
    'right', 'well', 'yeah', 'okay', 'sure', 'thing', 'things',
])


class RAGQueryable(Protocol):
    """Protocol for RAG engines that support querying."""

    def query(self, text: str) -> tuple:
        """Query RAG and return (context, confidence, source)."""
        ...


class TopicTrigger:
    """Detects when speech topics match document content.

    Uses lightweight keyword extraction + a quick ColBERT query to check
    if the current discussion matches indexed documents. Includes cooldown
    to avoid re-triggering the same topic.
    """

    def __init__(self, config: TriggerConfig, rag_engine: RAGQueryable) -> None:
        self.rag = rag_engine
        self.min_confidence = config.topic_match_threshold
        self.cooldown_seconds = config.topic_cooldown_seconds
        self._recent_topics: Dict[str, float] = {}

    def evaluate(self, text: str, conversation_context: str = "") -> Optional[Trigger]:
        """Extract topics from text and check against RAG index."""
        terms = _extract_key_terms(text)
        if not terms:
            return None

        now = time.time()

        # Check each term against the index
        for term in terms:
            # Cooldown
            last = self._recent_topics.get(term, 0.0)
            if now - last < self.cooldown_seconds:
                continue

            try:
                _, confidence, source = self.rag.query(term)
            except Exception:
                logger.debug("Topic query failed for term: %s", term)
                continue

            if confidence >= self.min_confidence:
                self._recent_topics[term] = now
                return Trigger(
                    type=TriggerType.TOPIC_MATCH,
                    text=term,
                    confidence=confidence,
                    source_context=conversation_context,
                    metadata={"topic": term, "source": source},
                    timestamp=now,
                )

        return None


def _extract_key_terms(text: str) -> list[str]:
    """Extract meaningful noun phrases / key terms from text.

    Returns terms sorted by length (longer = more specific = better).
    """
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)

    # Filter stop words, keep meaningful terms
    meaningful = [w for w in words if w not in _STOP_WORDS]

    # Also look for multi-word terms (2-3 word sequences without stop words)
    bigrams: list[str] = []
    for i in range(len(words) - 1):
        if words[i] not in _STOP_WORDS and words[i + 1] not in _STOP_WORDS:
            bigrams.append(f"{words[i]} {words[i + 1]}")

    # Combine and deduplicate, prioritize longer terms
    all_terms = list(set(bigrams + meaningful))
    all_terms.sort(key=len, reverse=True)

    return all_terms[:5]  # Top 5 terms to keep queries fast
