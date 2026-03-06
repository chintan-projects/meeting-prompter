# Meeting Intelligence CLI - Architectural Decisions & Learnings

A comprehensive record of design decisions, trade-offs, and lessons learned building a real-time audio Q&A system with Liquid AI models.

---

## Executive Summary

This project evolved through several key architectural pivots:

1. **Keyword → Semantic Search**: Jaccard similarity couldn't handle semantic queries → ColBERT late-interaction model
2. **Generation → Extraction → Hybrid**: Small LLMs hallucinated → extraction-only → hybrid (extract + generate)
3. **Chunk-based → Time-based Buffering**: Counting audio chunks was unreliable → real timestamp-based pause detection
4. **Dense Embedding → Late Interaction**: Single-vector embeddings lost token semantics → ColBERT MaxSim scoring

---

## 1. The Hallucination Problem & Solution Journey

### Problem: Small LLMs Don't Follow Context Instructions

**Discovery**: When we initially tried using LLMs (1-3B parameters) for direct answer generation from retrieved context, they would frequently:
- Add information not in the context
- Contradict the source material
- Generate plausible-sounding but incorrect answers

**Root Cause**: Small models are weak at following "only use this context" instructions. They're trained on broad knowledge and tend to draw from it even when told not to.

### Solution 1: Extraction-Only (No Generation)

**Approach** (`lib/answer_extractor.py`):
- Score each sentence for relevance to the question
- Return top-3 sentences as bullet points
- No LLM involved in answer creation

**Benefits**:
- Zero hallucination (answers are direct quotes)
- Traceable to source text
- Fast (~10ms)

**Drawbacks**:
- Choppy, unnatural bullet points
- No synthesis across sentences
- Sounds robotic

### Solution 2: Hybrid RAG (Final Architecture)

**Discovery**: LFM2-1.2B-RAG is specifically trained on 1M+ multi-document RAG samples to respect context.

**Architecture** (`lib/hybrid_answerer.py`):
```
Question → ColBERT (top-3) → Sentence Extraction → LFM2-1.2B-RAG → Fluent Answer
           Stage 1           Stage 2                Stage 3
           RETRIEVAL         GROUNDING              GENERATION
```

**Why This Works**:
1. **Extraction stage** filters irrelevant text before the LLM sees it
2. **LFM2-1.2B-RAG** is specifically trained to follow context
3. **Fallback** to extraction-only if generation fails

**Key Insight**: The extraction stage isn't just for fallback—it's **grounding**. By pre-filtering context, we structurally reduce hallucination risk.

---

## 2. Retrieval: Why ColBERT Over Dense Embeddings

### Problem: Keyword Search Misses Semantic Matches

| Query | Keyword (Jaccard) | ColBERT |
|-------|-------------------|---------|
| "What is Liquid AI?" | Found | Found (77%) |
| "neural network alternatives" | **MISS** | **Found (74%)** |
| "compete with OpenAI" | **MISS** | **Found (76%)** |

**Why Keyword Fails**: No word overlap between query and relevant content.

### Decision: ColBERT Late Interaction

**Traditional Dense Embeddings**:
```
Document → [single 768-dim vector]
Query    → [single 768-dim vector]
Score    = cosine_similarity
```
Problem: Compressing entire documents into single vectors loses token-level semantics.

**ColBERT (Our Choice)**:
```
Document → [[vec1], [vec2], [vec3], ...] (128-dim per token)
Query    → [[vec1], [vec2], [vec3], ...]
Score    = MaxSim (find best match for EACH query token)
```

**Why MaxSim Works**: "Neural" can match "model", "alternatives" can match "architecture"—even without exact keyword overlap.

### Implementation Details (`lib/colbert/`)

| Component | Purpose |
|-----------|---------|
| `retriever.py` | LFM2-ColBERT-350M model + PLAID index |
| `chunker.py` | Section-aware Markdown chunking |
| `index_manager.py` | Persistent index caching |
| `normalizer.py` | Sigmoid normalization for MaxSim scores |

**Fallback Strategy**: If ColBERT fails to load (memory pressure, missing dependencies), system automatically falls back to Jaccard similarity.

---

## 3. Section-Aware Chunking

### Problem: Random Chunking Loses Document Structure

When documents are chunked at arbitrary 400-character boundaries:
- Chunks lose section context
- "What is LEAP?" might retrieve text from adjacent sections
- Answer quality degrades

### Solution: Markdown Header Awareness

**Implementation** (`lib/colbert/chunker.py`):
```python
1. Split document on ## and ### headers first
2. Chunk each section separately (400 tokens max)
3. Prepend section header to each chunk
4. 50-token overlap preserves context at boundaries
```

**Result**: When you ask "What is LEAP?", you get chunks from the LEAP section, not random nearby text.

### Configuration Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `target_tokens` | 400 | ColBERT max is 512; leave room for special tokens |
| `overlap_tokens` | 50 | ~5% overlap preserves context at boundaries |
| `min_tokens` | 30 | Filter noise (fewer than ~6 words) |

---

## 4. Multi-Chunk Retrieval

### Problem: Single Best Match Misses Information

Answers often span multiple chunks. Returning only the top-1 result misses context.

### Solution: Top-3 with Confidence Filtering

**Implementation** (`lib/rag_engine.py`):
```python
results = colbert.query(text, k=3)  # Get top 3

# Combine chunks within 80% of top confidence
for chunk_text, confidence, source in results:
    if confidence >= top_confidence * 0.8:
        combined_chunks.append(chunk_text)
```

**Why 80% Threshold**: Include supplementary context only if it's nearly as relevant as the best match.

---

## 5. Audio Transcription Hallucinations

### Problem: LFM2-Audio Hallucinates on Noise

When given silence, background noise, or unclear audio, LFM2-Audio produces predictable patterns:
- "I don't know what's going to..."
- "She chose the one that was..."
- "Can you explain to me?"

These get processed as real questions and trigger RAG queries.

### Solution: Pattern-Based Hallucination Filter

**Implementation** (`coach.py:_is_hallucination`):

**3 Pattern Categories**:
1. **Vague Starters** (32 patterns): "i don't know what", "i think it's"
2. **Third-Person Statements** (5 patterns): "she ", "he ", "there was "
3. **Repetitive Phrases**: Detected via 3-word sequence repetition

**Key Insight**: Hallucinations are content-agnostic and follow predictable patterns. Pattern matching is more reliable than trying to detect "invalid" content semantically.

### Noise vs. Hallucination Pipeline

```
Transcription → Hallucination Filter → Noise Filter → Question Detector
                (pattern match)        (filler words)   (confidence score)
```

**Why Two Stages**:
- Hallucinations are dangerous (trigger false answers)
- Noise is benign (just ignored)

---

## 6. Time-Based Question Buffering

### Problem: "Second Question Triggers First"

**Naive Approach**: Count audio chunks, flush after N chunks of silence.

**Why It Failed**: Audio chunks don't correspond to real pauses. A 4-second chunk might contain silence at the start, middle, or end unpredictably.

### Solution: Real Timestamp-Based Pause Detection

**Implementation** (`lib/question_buffer.py`):
```python
class BufferConfig:
    pause_threshold: float = 1.5   # Seconds of silence to flush
    max_buffer_time: float = 8.0   # Maximum before forced flush
    min_words: int = 4             # Minimum for valid question
    confidence_threshold: float = 0.3
```

**Key Behaviors**:
1. **Pause Detection**: Flush when `time_since_last_speech >= 1.5s`
2. **Max Buffer**: Force flush at 8 seconds (prevents indefinite buffering)
3. **Peek-Ahead**: When flushing, check if next chunk is a question start

### Thread Safety

The buffer uses `threading.Lock()` because audio capture runs in a separate thread from processing.

---

## 7. Question Detection Without Punctuation

### Problem: Transcription Omits Question Marks

ASR models often don't produce punctuation. Relying on `?` detection misses most questions.

### Solution: Multi-Factor Scoring

**Implementation** (`lib/question_detector.py`):
```python
score = 0.0
+ 0.5  (if has '?')
+ 0.3  (if matches question patterns like "what is", "how does")
+ 0.3  (keyword overlap: pricing, api, integrate, etc.)
+ 0.2  (question word at sentence start)
+ 0.1  (if >= 7 words)
```

**7 Pattern Categories**:
1. Direct question words: "what", "how", "why"
2. Verb patterns: "what is", "how does"
3. Yes/No starters: "is this", "can you"
4. Common phrases: "tell me", "explain"
5. Exploration: "what about", "how about"
6. Technical: "how does X work"
7. Definition: "what's the", "what is the"

**Key Insight**: Single signals are unreliable. Additive multi-factor scoring combines weak signals into strong confidence.

---

## 8. KV Cache Corruption Fix

### Problem: `llama_decode returned -1` Error

On subsequent generation calls, llama.cpp threw errors indicating KV cache corruption.

### Root Cause

The KV cache (key-value cache for attention) becomes corrupted between calls when the model state isn't properly reset.

### Solution: Reset Before Each Generation

**Before** (broken):
```python
def _reset_if_needed(self):
    self._call_count += 1
    if self._call_count >= 10:  # Reset every 10 calls
        self.llm.reset()
```

**After** (working):
```python
def _reset_state(self):
    if self.llm is not None:
        try:
            self.llm.reset()
        except Exception:
            self.llm = None  # Reload on failure
            self.load()
```

**Trade-off**: Per-call reset is slightly slower but prevents all KV cache issues.

---

## 9. ChatML Format for RAG Generation

### Why ChatML

LFM2-1.2B-RAG was trained on 1M+ samples using ChatML format:
```
<|im_start|>user
[context and question]
<|im_end|>
<|im_start|>assistant
```

### Prompt Template

```python
RAG_PROMPT_TEMPLATE = """<|im_start|>user
Use the following context to answer the question. Be concise and direct.
Only use information from the provided context.

CONTEXT:
{context}

QUESTION: {question}<|im_end|>
<|im_start|>assistant
"""
```

### Generation Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0 | Greedy decoding for factual responses |
| `max_tokens` | 200 | Concise answers |
| `n_ctx` | 2048 | Context window |
| `max_context_chars` | 1500 | Leave room for prompt + generation |

### Stop Tokens

```python
stop=["<|im_end|>", "<|im_start|>", "\n\nQUESTION:", "\n\nCONTEXT:", "---"]
```

These prevent the model from trying to continue the conversation or generate new context.

---

## 10. Confidence Score Normalization

### Problem: Raw ColBERT Scores Are Unbounded

ColBERT MaxSim scores range from ~10 to ~50 depending on query/document. Hard to interpret.

### Solution: Sigmoid Normalization

**Implementation** (`lib/colbert/normalizer.py`):
```python
normalized = 1 / (1 + exp(-(raw_score - center) / scale))
center = 25.0  # Moderate match = 0.5
scale = 5.0    # Controls steepness
```

**Score Interpretation**:
| Raw Score | Normalized | Meaning |
|-----------|------------|---------|
| < 15 | < 0.15 | Poor match |
| 15-25 | 0.15-0.50 | Moderate match |
| 25-35 | 0.50-0.85 | Good match |
| > 35 | > 0.85 | Excellent match |

### Confidence Thresholds

| Threshold | Value | Used For |
|-----------|-------|----------|
| `MIN_CONFIDENCE` | 0.30 | Below this, return "no match" |
| `HIGH_CONFIDENCE` | 0.70 | UI indicator for strong matches |
| `EXTRACTION_MIN` | 0.25 | Minimum to proceed to generation |

---

## 11. Audio Capture Architecture

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_duration` | 4.0s | Long enough for complete phrases |
| `overlap` | 0.5s | Preserves context between chunks |
| `sample_rate` | 16000 | Required by LFM2-Audio |
| `blocksize` | 100ms | Balance between latency and efficiency |

### Dual-Threshold Silence Detection

```python
is_speech = (rms > 0.005) and (peak > 0.02)
```

**Why Both Thresholds**:
- RMS alone: Misses short noise spikes
- Peak alone: Triggers on low background
- Both: Robust silence detection

### Silence Callback

When silence is detected, the audio capture notifies the question buffer rather than silently discarding. This enables the buffer's pause-based flush logic.

---

## 12. Extraction Scoring Algorithm

### Multi-Factor Sentence Scoring

**Implementation** (`lib/answer_extractor.py`):
```python
score = 0.0
+ (keyword_overlap / question_words) * 0.5  # Primary signal
+ 0.2 (if definition pattern found)         # "is defined as", "stands for"
+ position_bonus * 0.2                       # Earlier = likely definition
- 0.2 (if > 50 words)                       # Length penalty
+ 0.1 * exact_term_matches                  # Bonus for key terms
```

### Position Bonus

```python
position_score = 1.0 - (position / total_sentences) * 0.3
```

**Rationale**: Definitions typically appear early in documents.

### Document Order Restoration

After selecting top sentences by score, re-sort by original position:
```python
top_sentences.sort(key=lambda x: x.position)
```

This ensures answers read naturally (chronological flow).

---

## 13. LLM Model Choices & Trade-offs

### Three-Model Architecture

| Model | Size | Purpose | Latency |
|-------|------|---------|---------|
| LFM2-Audio-1.5B | 1.2 GB | Speech-to-text | ~300ms |
| LFM2-ColBERT-350M | 1.4 GB | Semantic retrieval | ~100ms |
| LFM2-1.2B-RAG | 700 MB | Answer generation | ~500ms |

### Why Separate Models?

1. **LFM2-Audio**: Multimodal model that directly processes audio waveforms
2. **LFM2-ColBERT**: Late-interaction retrieval (not a generative model)
3. **LFM2-1.2B-RAG**: Trained specifically for RAG tasks

Using specialized models for each stage outperforms a single general-purpose model.

### Quantization Choice

Chose **Q4_K_M** (4-bit quantization) for LFM2-1.2B-RAG:
- 731 MB vs 2.4 GB for FP16
- Minimal quality loss for RAG tasks
- Fits in memory alongside other models

---

## 14. Fallback Chains

### RAG Engine Fallback

```
ColBERT → Jaccard Keyword Search
  ↓           ↓
(semantic)  (fallback if ColBERT fails)
```

**Env Variable**: `RAG_USE_FALLBACK=1` forces keyword search (for low-memory environments).

### Answer Generation Fallback

```
LFM2-1.2B-RAG → Extraction Bullets
  ↓                  ↓
(fluent answer)    (fallback if generation fails)
```

### Extraction Fallback

```
High-Confidence Sentences → "I don't have information on that"
  ↓                              ↓
(if score >= 0.25)            (if score < 0.25)
```

---

## 15. Known Constraints & Workarounds

| Constraint | Workaround |
|------------|------------|
| KV cache corruption | Reset before each generation |
| llama.cpp verbose logging | 20+ skip patterns in output filter |
| Transcription missing punctuation | Confidence scoring + word count |
| Small model context following | Structural extraction + specialized RAG model |
| Audio device name variation | Substring matching against available devices |

---

## 16. Performance Characteristics

### Latency Breakdown

| Stage | Latency |
|-------|---------|
| Audio capture | Real-time (4s chunks) |
| Transcription (LFM2-Audio) | ~300ms |
| Retrieval (ColBERT) | ~100ms |
| Extraction | ~10ms |
| Generation (LFM2-1.2B-RAG) | ~500ms |
| **Total** | **~900ms** |

### Memory Usage

| Component | Memory |
|-----------|--------|
| LFM2-Audio-1.5B | ~2 GB |
| LFM2-ColBERT-350M | ~1.5 GB |
| LFM2-1.2B-RAG | ~1 GB |
| **Total** | **~4.5 GB** |

Requires 16GB+ RAM Mac (M1/M2/M3/M4).

---

## 17. Key Learnings

### 1. Extraction Before Generation Prevents Hallucination

Don't feed raw retrieved chunks to an LLM. Extract relevant sentences first—this structurally prevents the model from seeing (and hallucinating from) irrelevant context.

### 2. Specialized Models Outperform General-Purpose

LFM2-1.2B-RAG (trained on 1M+ RAG samples) follows context far better than a general 1.2B model. Model training data matters more than size for specific tasks.

### 3. Time-Based > Count-Based for Real-Time

Audio chunk counts don't correspond to real pauses. Real timestamps are the only reliable way to detect speech boundaries.

### 4. Pattern Matching Beats Semantic Detection for Artifacts

Hallucinations, noise, and filler words follow predictable patterns. Pattern matching is more reliable and faster than trying to semantically classify "valid" content.

### 5. Fallback Chains Enable Graceful Degradation

Every stage should have a fallback: ColBERT→Jaccard, Generation→Extraction, Extraction→"No match". Users get an answer (even if degraded) rather than an error.

### 6. Late Interaction Captures Token Semantics

ColBERT's per-token embeddings with MaxSim scoring catch semantic relationships that single-vector embeddings miss. Worth the extra memory for quality retrieval.

### 7. Reset LLM State Between Calls

llama.cpp's KV cache can become corrupted between generation calls. Always reset model state before each generation to prevent `llama_decode` errors.

### 8. Section Headers Improve Retrieval Quality

Prepending section headers to chunks dramatically improves retrieval for "What is X?" questions. The model learns that chunks with matching headers are more relevant.

---

## 18. Turn-Based Transcript Architecture (v2)

### Problem: Raw ASR Chunks Create Unreadable Transcripts

LFM2.5-Audio processes audio in ~4-second chunks. Storing and displaying each chunk individually produces a fragmented, choppy transcript — each line is a partial phrase rather than a coherent thought.

A naive frontend fix (grouping by time gaps) was insufficient because:
- It didn't produce semantically meaningful groupings
- The frontend had no knowledge of speech patterns
- It couldn't handle real-time updates cleanly (groups would shift as new chunks arrived)

### Solution: Backend Turn Accumulation

Speech turns are accumulated on the **backend**, where the ASR pipeline has timing information and can detect pauses. The architecture introduces a new `TranscriptBuffer` layer between the ASR output and the transcript store.

```
Audio Chunk → LFM2.5-Audio → text
                                │
                    ┌───────────┴───────────┐
                    │                       │
            TranscriptBuffer         ConversationBuffer
            (for display)            (for trigger detection)
                    │                       │
              on_update / on_final    Trigger Engine
                    │
            TranscriptStore.upsert()
                    │
            WebSocket push
            (transcript_update / transcript_final)
                    │
            useTranscript.upsertSegment()
                    │
            TranscriptPane (paragraph blocks)
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| `TranscriptBuffer` | `src/api/transcript_buffer.py` | Accumulates raw chunks into turns. Detects boundaries via pause (2s) or max duration (30s). Fires `on_update` and `on_final` callbacks. |
| `TranscriptStore.upsert()` | `src/api/transcript_store.py` | Creates or updates a segment by ID. Turns are upserted on each update (text grows), then marked final. Edit overlay preserved. |
| `Session._on_turn_*` | `src/api/session.py` | Bridges buffer callbacks → asyncio queue → WebSocket via `loop.call_soon_threadsafe`. |
| WebSocket protocol | `src/api/routes/transcript.py` | `transcript_update` (partial turn) and `transcript_final` (completed turn). Late-connecting clients receive existing turns on connect. |
| `useTranscript` | `app/src/hooks/useTranscript.ts` | React hook with upsert semantics — finds by turn ID and updates, or creates new. Preserves user edits. |
| `TranscriptPane` | `app/src/components/TranscriptPane.tsx` | Renders turns as timestamped paragraphs. Active turns show pulsing blue indicator. Finalized turns are clean static text. Double-click to edit. |

### Turn Lifecycle

```
Chunk 1 arrives → TranscriptBuffer creates turn-1 (active)
    → on_update callback → Store.upsert(turn-1, "Hello") → WS: transcript_update

Chunk 2 arrives (within 2s) → extends turn-1
    → on_update callback → Store.upsert(turn-1, "Hello world") → WS: transcript_update

2s silence detected → turn-1 finalized
    → on_final callback → Store.upsert(turn-1, "Hello world", is_final=True) → WS: transcript_final

Chunk 3 arrives → new turn-2 created
```

### Dual Buffer Design

Two independent buffers receive the same raw chunks for different purposes:
- **TranscriptBuffer**: Optimized for display (simple pause-based turn boundaries)
- **ConversationBuffer**: Optimized for intelligence (question scoring, trigger evaluation, rolling 90s window)

They don't share state. This separation keeps each focused on its purpose.

### Speaker Identification (Planned)

The `speaker` field exists on `Turn`, `TranscriptSegment`, and in the WebSocket protocol, but is currently unused. When speaker diarization is added, it will populate this field and the UI will render speaker labels above turn blocks.

---

## 19. File Reference

### Core Pipeline (lib/)

| File | Purpose |
|------|---------|
| `coach.py` | CLI entry point, args, startup |
| `lib/orchestrator.py` | MeetingOrchestrator — central pipeline coordinator |
| `lib/config.py` | Typed dataclass config loader |
| `lib/filters.py` | Hallucination/noise/normalization filters |
| `lib/audio_capture.py` | Microphone streaming, silence detection |
| `lib/lfm2_wrapper.py` | LFM2.5-Audio subprocess management |
| `lib/answer_extractor.py` | Sentence extraction for grounding |
| `lib/rag_generator.py` | LFM2.5-1.2B-Instruct wrapper with ChatML |
| `lib/rag_engine.py` | ColBERT + Jaccard fallback |
| `lib/dashboard.py` | CLI dashboard with trigger-type coloring |
| `lib/colbert/retriever.py` | ColBERT model + PLAID index |
| `lib/colbert/chunker.py` | Section-aware document chunking |
| `lib/colbert/normalizer.py` | Sigmoid score normalization |

### Trigger Engine (lib/triggers/)

| File | Purpose |
|------|---------|
| `lib/triggers/types.py` | TriggerType enum, Trigger dataclass |
| `lib/triggers/engine.py` | Orchestrator: runs all triggers, priority sort |
| `lib/triggers/question_trigger.py` | Question detection (patterns + keywords) |
| `lib/triggers/alert_trigger.py` | Watch word scanning with cooldown |
| `lib/triggers/topic_trigger.py` | ColBERT-backed topic detection |
| `lib/triggers/followup_trigger.py` | Pause-based follow-up suggestions |

### Conversation Intelligence (lib/conversation/)

| File | Purpose |
|------|---------|
| `lib/conversation/buffer.py` | Rolling 90s transcript + trigger routing |
| `lib/conversation/meeting_context.py` | YAML meeting context loader |

### Mode-Aware Generation (lib/generation/)

| File | Purpose |
|------|---------|
| `lib/generation/prompts.py` | ChatML prompt templates per trigger type |
| `lib/generation/generator.py` | ModeAwareGenerator — trigger-routed generation |
| `lib/generation/types.py` | GenerationResult dataclass |

### FastAPI Backend (src/api/)

| File | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI server + WebSocket endpoints |
| `src/api/session.py` | Session manager (bridges audio pipeline → WebSocket) |
| `src/api/transcript_buffer.py` | Turn-based ASR chunk accumulator |
| `src/api/transcript_store.py` | Append-only transcript with edit overlay + upsert |
| `src/api/notes_generator.py` | Structured meeting notes via LLM |
| `src/api/routes/session.py` | Session start/stop/status/reindex endpoints |
| `src/api/routes/transcript.py` | WebSocket transcript stream (turn updates + edits) |
| `src/api/routes/prompts.py` | WebSocket trigger results stream |
| `src/api/routes/notes.py` | Notes generate/export/save/download |
| `src/api/routes/context.py` | Meeting context upload |

### Tauri + React Frontend (app/)

| File | Purpose |
|------|---------|
| `app/src-tauri/src/lib.rs` | Rust shell: spawns Python backend, manages lifecycle |
| `app/src/App.tsx` | Root component, layout, WebSocket connections |
| `app/src/components/TranscriptPane.tsx` | Left pane: turn-based editable transcript |
| `app/src/components/PromptsPane.tsx` | Right pane: live trigger results |
| `app/src/components/StatusBar.tsx` | Session controls, audio health |
| `app/src/components/MeetingSetup.tsx` | Pre-meeting context config dialog |
| `app/src/components/NoteEditor.tsx` | Post-meeting structured notes editor |
| `app/src/hooks/useWebSocket.ts` | WebSocket connection + reconnect hook |
| `app/src/hooks/useTranscript.ts` | Transcript state with turn-based upsert |

---

## 20. Evolution Timeline

```
Phase 1: Basic Pipeline
├── Keyword search (Jaccard) → Missed semantic queries
├── Direct LLM generation → Hallucinated answers
└── Chunk-count buffering → "Second question triggers first" bug

Phase 2: Improved Retrieval
├── Added ColBERT for semantic search → 74-77% matches on semantic queries
├── Section-aware chunking → Better context for "What is X?" questions
└── Multi-chunk retrieval (top-3) → Richer context

Phase 3: Extraction-Only
├── Removed LLM generation → Zero hallucination
├── Sentence scoring → Grounded answers
└── Bullet point formatting → Accurate but choppy

Phase 4: Hybrid RAG (Current)
├── Added LFM2-1.2B-RAG → Fluent answers
├── Extraction as grounding stage → Structural hallucination prevention
├── Time-based buffering → Reliable pause detection
└── KV cache reset → No more llama_decode errors
```

---

*Document generated from codebase analysis and development history.*
