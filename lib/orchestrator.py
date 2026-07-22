"""Meeting intelligence orchestrator — coordinates pipeline components.

Central class that wires together:
- Audio capture → transcription → filters
- Conversation buffer → trigger engine
- RAG retrieval → mode-aware generation
- Dashboard display

Exposes callback hooks (on_transcription, on_silence_detected,
on_trigger_result) so the API Session can observe pipeline events
without monkey-patching process_chunk.
"""

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from lib.audio_capture import AudioCapture
from lib.config import AppConfig
from lib.paths import get_models_dir, get_output_dir, get_runner_dir

if TYPE_CHECKING:
    from lib.audio_protocol import AudioCaptureProtocol
from lib.conversation.buffer import ConversationBuffer
from lib.conversation.meeting_context import MeetingContext, load_meeting_context
from lib.dashboard import Dashboard, display_header, display_status
from lib.filters import is_hallucination_only, is_noise, normalize_text
from lib.generation.generator import ModeAwareGenerator
from lib.generation.types import GenerationResult
from lib.lfm2_wrapper import LFM2Wrapper
from lib.rag_engine import RAGEngine
from lib.triggers.engine import TriggerEngine
from lib.triggers.types import Trigger, TriggerType

logger = logging.getLogger(__name__)

# Paths resolved via lib.paths (supports dev + packaged modes)
MODELS_DIR = get_models_dir()
AUDIO_MODELS_DIR = MODELS_DIR / "LFM2.5-Audio-1.5B-GGUF"
RUNNER_DIR = get_runner_dir()
OUTPUT_FILE = get_output_dir() / "live_analytics.txt"

# Type aliases for callback signatures
TranscriptionCallback = Callable[[str, float], None]
SilenceCallback = Callable[[float, str], None]  # (timestamp, source)
TriggerResultCallback = Callable[[Trigger, GenerationResult], None]


def _resolve_rag_model(config: AppConfig) -> Path:
    """Resolve the answer model path from config, with fallbacks.

    Primary comes from ``models.generation.model_file`` (D-03: default 2.6B).
    If it is missing on disk, fall back to the 1.2B instruct model, then the
    legacy LFM2 RAG model, so a fresh checkout still runs.
    """
    primary = MODELS_DIR / config.models.generation.model_file
    if primary.exists():
        return primary
    for fallback in ("LFM2.5-1.2B-Instruct-Q4_K_M.gguf", "LFM2-1.2B-RAG-Q4_K_M.gguf"):
        candidate = MODELS_DIR / fallback
        if candidate.exists():
            logger.warning(
                "Answer model %s not found — falling back to %s",
                config.models.generation.model_file,
                fallback,
            )
            return candidate
    return primary


class MeetingOrchestrator:
    """Real-time meeting intelligence pipeline.

    Coordinates audio capture, transcription, trigger detection,
    RAG retrieval, and mode-aware generation.

    Callback hooks allow external consumers (like the API Session) to
    observe pipeline events without overriding methods:
    - on_transcription(text, timestamp): valid speech transcribed
    - on_silence_detected(timestamp): ASR returned empty/hallucination
    - on_trigger_result(trigger, result): trigger fired with a result
    """

    def __init__(
        self,
        config: AppConfig,
        audio_device: str = "BlackHole 2ch",
        meeting_context_path: Optional[Path] = None,
        headless: bool = False,
        audio_capture: "Optional[AudioCaptureProtocol]" = None,
    ) -> None:
        self.config = config
        self._headless = headless

        if not headless:
            display_header()

        # --- Callback hooks (set by Session or other consumers) ---
        self.on_transcription: Optional[TranscriptionCallback] = None
        self.on_silence_detected: Optional[SilenceCallback] = None
        self.on_trigger_result: Optional[TriggerResultCallback] = None

        # Load meeting context if provided
        self.meeting_context: Optional[MeetingContext] = None
        if meeting_context_path:
            self.meeting_context = load_meeting_context(meeting_context_path)
            if self.meeting_context:
                self._status(self.meeting_context.summary())

        # Transcription model — LFM2.5 preferred, LFM2 fallback
        self._status("Loading audio model...")
        audio_dir = AUDIO_MODELS_DIR if AUDIO_MODELS_DIR.exists() else MODELS_DIR
        try:
            self.lfm2 = LFM2Wrapper(audio_dir, RUNNER_DIR, model_version="2.5")
            self._status("LFM2.5-Audio ready")
        except FileNotFoundError:
            self.lfm2 = LFM2Wrapper(MODELS_DIR, RUNNER_DIR, model_version="2.0")
            self._status("LFM2-Audio ready (legacy)")

        # RAG retrieval (hybrid FTS5 + vector). The docs dir honors an activated
        # prepared corpus (F-704: data/corpus_active.json, own DB) and otherwise
        # falls back to the configured pair.
        from lib.corpus.active import resolve_corpus

        docs_dir, db_path = resolve_corpus(config.paths.docs_dir, config.rag.db_path)
        self._status("Loading RAG engine...")
        from lib.rag import RAGConfig as _RAGConfig

        rag_config = _RAGConfig(
            max_chunk_tokens=config.rag.max_chunk_tokens,
            chunk_overlap_tokens=config.rag.chunk_overlap_tokens,
            lexical_weight=config.rag.lexical_weight,
            semantic_weight=config.rag.semantic_weight,
            lexical_top_k=config.rag.lexical_top_k,
            semantic_top_k=config.rag.semantic_top_k,
            embedding_model=config.rag.embedding_model,
            embedding_dimension=config.rag.embedding_dimension,
        )
        self.rag = RAGEngine(docs_dir, db_path=db_path, config=rag_config)
        self._status("RAG engine ready")
        if config.triggers.retrieval_first:
            # F-705: the live path is retrieval-only, so the embedder must be
            # warm before the first trigger (cold load ~1.6s vs ~150ms budget).
            # Embedder-only — no SQLite access — so a background thread is safe.
            threading.Thread(target=self.rag.warm_up, daemon=True).start()

        # Persistent warm-model runtime (F-508): single owner of the load-once
        # models. The encoder it hands out stays lazy (no weights until embed()),
        # so the probe head is wired-but-off at zero startup cost.
        from lib.warm_runtime import WarmModelRuntime

        self.runtime = WarmModelRuntime()
        self.runtime.register("embedder", getattr(self.rag, "_embedder", None))

        trigger_config = config.triggers
        if self.meeting_context:
            trigger_config.watch_words = self.meeting_context.watch_words

        self.encoder = self.runtime.encoder()
        self.trigger_engine = TriggerEngine(trigger_config, self.rag, encoder=self.encoder)

        # Conversation buffer (rolling transcript + trigger routing)
        self.buffer = ConversationBuffer(
            config=config.buffer,
            trigger_engine=self.trigger_engine,
        )

        # Mode-aware generation (replaces HybridAnswerer)
        rag_model = _resolve_rag_model(config)
        self._status("Loading generation model...")
        self.generator = ModeAwareGenerator(
            model_path=rag_model,
            max_context_chars=config.models.generation.max_context_chars,
            min_extraction_confidence=config.detection.extraction_confidence_minimum,
            min_answer_length=config.triggers.min_answer_length,
        )
        self.runtime.register("instruct", self.generator)
        self._status("Generation ready")

        # Audio capture: use provided capture (e.g. SystemAudioCapture) or default
        self.audio: "AudioCaptureProtocol" = audio_capture or AudioCapture(device=audio_device)

        # Dashboard (skipped in headless/API mode)
        self.dashboard: Optional[Dashboard] = None
        if not headless:
            self.dashboard = Dashboard()
            if self.meeting_context and self.meeting_context.title:
                self.dashboard.set_meeting_title(self.meeting_context.title)

        # State
        self._running = False
        self.chunk_count = 0
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _status(self, message: str) -> None:
        """Display status in CLI mode, log in headless mode."""
        if self._headless:
            logger.info("[status] %s", message)
        else:
            display_status(message)

    def run(self) -> None:
        """Start real-time processing loop."""
        self._status(f"Listening on {self.audio.device}...")
        self._running = True
        try:
            self.audio.start_stream(self.process_chunk)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            logger.info("Session ended after %d chunks", self.chunk_count)

    def process_chunk(self, audio_path: Path, timestamp: Optional[float] = None) -> None:
        """Process a single audio chunk through the full pipeline.

        Silence detection is now driven by ASR output: if the model
        returns empty text or a hallucination, we treat it as silence.
        """
        timestamp = timestamp or time.time()

        try:
            # 1. Transcribe
            text = self.lfm2.transcribe(audio_path)

            # 2. Handle empty/error results as silence
            if not text or text.startswith("["):
                self._notify_silence(timestamp)
                triggers = self.buffer.force_flush()
                self._handle_triggers(triggers)
                return

            # 3. Handle hallucinations as silence
            if is_hallucination_only(text):
                self._notify_silence(timestamp)
                triggers = self.buffer.force_flush()
                self._handle_triggers(triggers)
                return

            self.chunk_count += 1

            # 4. Notify callback of valid transcription
            if self.on_transcription:
                self.on_transcription(text, timestamp)

            # 5. Feed into conversation buffer → triggers
            #    (use strict is_noise filter for trigger pipeline only)
            if not is_noise(text):
                triggers = self.buffer.add_chunk(text, timestamp)
                self._handle_triggers(triggers)
            else:
                # Short/filler speech: still valid transcription (callback
                # fired above), but don't feed trigger pipeline
                logger.debug("Filler speech, skipping triggers: %r", text[:60])

            # 6. Update dashboard transcript preview
            if self.dashboard:
                self.dashboard.set_transcript_preview(text)
                self.dashboard.render()

        except Exception as e:
            logger.error("Error processing chunk: %s", e)

    def _notify_silence(self, timestamp: float) -> None:
        """Notify silence via callback and conversation buffer."""
        if self.on_silence_detected:
            self.on_silence_detected(timestamp, "system")
        triggers = self.buffer.on_silence(timestamp)
        self._handle_triggers(triggers)

    def _handle_triggers(self, triggers: List[Trigger]) -> None:
        """Process fired triggers: RAG lookup → generation → display."""
        for trigger in triggers:
            result = self._process_trigger(trigger)
            if result and result.answer and not result.answer.startswith("["):
                if self.dashboard:
                    self.dashboard.add_result(result)
                    self.dashboard.render()

                # Notify callback
                if self.on_trigger_result:
                    self.on_trigger_result(trigger, result)

                # Record Q&A for conversation memory
                if trigger.type == TriggerType.QUESTION:
                    self.buffer.add_qa_pair(trigger.text, result.answer)

                # Log to file
                self._log_result(trigger, result)

    def _process_trigger(self, trigger: Trigger) -> Optional[GenerationResult]:
        """Route a trigger: retrieval-first (default, F-705/D-08) or generation."""
        if self.config.triggers.retrieval_first:
            return self._process_trigger_retrieval(trigger)
        return self._process_trigger_generated(trigger)

    def _process_trigger_retrieval(self, trigger: Trigger) -> Optional[GenerationResult]:
        """Retrieval-first live path: show the best borrowable unit, no LLM.

        ~120-190ms warm (retrieval only). Silence beats noise: nothing
        answer-shaped over the confidence floor → no card. Generation remains
        available on demand via generate_for_text (POST /prompts/generate).
        """
        from lib.corpus.readiness import live_borrowable

        t0 = time.time()
        query_text = normalize_text(trigger.text)
        results = self.rag.retrieve(query_text, top_k=5)
        card = live_borrowable(
            results,
            query_text,
            min_confidence=self.config.detection.rag_confidence_minimum,
            min_answer_length=self.config.triggers.min_answer_length,
        )
        if card is None:
            logger.debug("retrieval-first: no borrowable unit for %s", trigger.type.value)
            return None
        return GenerationResult(
            answer=str(card["answer"]),
            trigger_type=trigger.type,
            confidence=float(card["confidence"]),
            method="retrieval",
            latency_ms=(time.time() - t0) * 1000,
            source=str(card["doc"]),
            heading=str(card["heading"]),
            source_text=str(card["full_text"]),
        )

    def generate_for_text(
        self, text: str, trigger_type_value: str = "question"
    ) -> Optional[GenerationResult]:
        """On-demand generation (user-gated, D-02): the demoted LLM path.

        Used by POST /prompts/generate when the user explicitly asks for a
        generated answer on top of the borrowable unit.
        """
        try:
            ttype = TriggerType(trigger_type_value)
        except ValueError:
            ttype = TriggerType.QUESTION
        trigger = Trigger(
            type=ttype,
            text=text,
            confidence=1.0,
            source_context=self.buffer.get_recent_context(),
            timestamp=time.time(),
        )
        return self._process_trigger_generated(trigger)

    def _process_trigger_generated(self, trigger: Trigger) -> Optional[GenerationResult]:
        """Generation path: RAG context → mode-aware LLM answer."""
        query_text = normalize_text(trigger.text)

        rag_context, confidence, source_file = self.rag.query(query_text)

        if confidence < self.config.detection.rag_confidence_minimum:
            logger.debug(
                "Skipping %s trigger: low RAG confidence %.2f",
                trigger.type.value,
                confidence,
            )
            return None

        conversation = self.buffer.get_recent_context()

        if self.meeting_context:
            meeting_info = self.meeting_context.as_prompt_context()
            conversation = f"{meeting_info}\n\n{conversation}"

        result = self.generator.process_trigger(
            trigger=trigger,
            rag_context=rag_context,
            conversation_context=conversation,
        )
        result.source = source_file
        return result

    def _log_result(self, trigger: Trigger, result: GenerationResult) -> None:
        """Append trigger result to output file."""
        try:
            ts = time.strftime("%H:%M:%S")
            emoji = trigger.type.emoji
            label = trigger.type.label
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"[{ts}] {emoji} {label}: {trigger.text}\n")
                f.write(f"[{ts}]   → {result.answer}\n")
                f.write(
                    f"[{ts}]   ({result.method}, {result.confidence:.0%}, "
                    f"{result.latency_ms:.0f}ms)\n\n"
                )
        except Exception as e:
            logger.debug("Failed to log result: %s", e)
