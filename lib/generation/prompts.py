"""Prompt templates per trigger type.

Each trigger type gets a purpose-built prompt that:
- Includes conversation context for relevance
- Limits output length appropriately
- Instructs the model on the expected response style

All prompts use ChatML format for LFM2.5-Instruct.
"""

# --- Question: direct answer with grounded context ---
QUESTION_SYSTEM = (
    "You are a meeting intelligence assistant. Answer the question using ONLY "
    "the provided context. Be concise and direct (2-3 sentences max). "
    "If the context does not contain the answer, say "
    '"I don\'t have that information in my documents."'
)

QUESTION_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
RECENT CONVERSATION:
{conversation}

CONTEXT:
{context}

QUESTION: {text}<|im_end|>
<|im_start|>assistant
"""

QUESTION_MAX_TOKENS = 200


# --- Topic brief: short key-fact surfacing ---
TOPIC_SYSTEM = (
    "You are a meeting intelligence assistant. The discussion has touched on a "
    "topic that matches your documents. Provide a brief, relevant fact from the "
    "context that would be useful right now. One to two sentences max."
)

TOPIC_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
RECENT CONVERSATION:
{conversation}

RELEVANT CONTEXT:
{context}

TOPIC DETECTED: {text}<|im_end|>
<|im_start|>assistant
"""

TOPIC_MAX_TOKENS = 100


# --- Follow-up: suggest one follow-up point ---
FOLLOWUP_SYSTEM = (
    "You are a meeting intelligence assistant. Based on the recent conversation "
    "and your documents, suggest ONE specific follow-up point or question that "
    "would be valuable to raise. Keep it to one sentence."
)

FOLLOWUP_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
RECENT CONVERSATION:
{conversation}

RELATED CONTEXT:
{context}

DISCUSSION POINT: {text}<|im_end|>
<|im_start|>assistant
"""

FOLLOWUP_MAX_TOKENS = 75


# --- Alert: relevant context for watch word ---
ALERT_SYSTEM = (
    "You are a meeting intelligence assistant. A watch word was detected in the "
    "conversation. Provide the most relevant information from your documents "
    "about this topic. Be concise (1-2 sentences)."
)

ALERT_PROMPT = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
RECENT CONVERSATION:
{conversation}

CONTEXT:
{context}

WATCH WORD DETECTED: {text}<|im_end|>
<|im_start|>assistant
"""

ALERT_MAX_TOKENS = 100


# ChatML stop tokens shared by all prompt types
STOP_TOKENS = [
    "<|im_end|>",
    "<|im_start|>",
    "\n\nQUESTION:",
    "\n\nCONTEXT:",
    "---",
]
