"""Application configuration loader with typed dataclasses."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    chunk_duration: float = 4.0
    sample_rate: int = 16000
    overlap: float = 0.5
    rms_threshold: float = 0.005
    peak_threshold: float = 0.02
    device_default: str = "BlackHole 2ch"
    device_mic: str = "MacBook Pro Microphone"
    capture_mode: str = "auto"  # "auto", "app", "device"


@dataclass
class AudioModelConfig:
    version: str = "2.5"
    system_prompt: str = "Perform ASR."
    timeout: int = 30


@dataclass
class GenerationModelConfig:
    n_ctx: int = 4096
    max_tokens: int = 200
    temperature: float = 0.0
    top_p: float = 1.0
    max_context_chars: int = 6000
    max_question_chars: int = 500


@dataclass
class RetrievalConfig:
    top_k: int = 5


@dataclass
class ModelsConfig:
    audio: AudioModelConfig = field(default_factory=AudioModelConfig)
    generation: GenerationModelConfig = field(default_factory=GenerationModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


@dataclass
class BufferConfig:
    # ConversationBuffer (trigger pipeline)
    pause_threshold: float = 1.5
    max_buffer_time: float = 8.0
    min_words: int = 4
    confidence_threshold: float = 0.3
    min_start_score: float = 0.1
    # TranscriptBuffer (turn detection for UI)
    turn_pause: float = 2.0
    max_turn_duration: float = 30.0


@dataclass
class DetectionConfig:
    question_score_threshold: float = 0.25
    rag_confidence_minimum: float = 0.30
    extraction_confidence_minimum: float = 0.25


@dataclass
class RAGPipelineConfig:
    """Hybrid FTS5 + vector RAG pipeline settings."""

    max_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    lexical_weight: float = 0.05
    semantic_weight: float = 0.95
    lexical_top_k: int = 20
    semantic_top_k: int = 20
    db_path: str = "data/rag.db"


@dataclass
class TriggerConfig:
    question_score_threshold: float = 0.25
    topic_match_threshold: float = 0.50
    topic_cooldown_seconds: float = 30.0
    followup_pause_threshold: float = 3.0
    followup_rag_threshold: float = 0.40
    watch_words: List[str] = field(default_factory=list)
    min_answer_length: int = 10  # suppress answers shorter than this (F-202)
    dismiss_persistent_ms: int = 0  # 0 = no auto-dismiss (Answer, Heads Up)
    dismiss_standard_ms: int = 90_000  # Suggest cards auto-dismiss (ms)
    dismiss_ephemeral_ms: int = 45_000  # FYI cards auto-dismiss (ms)


@dataclass
class RefinerConfig:
    """Configuration for the LFM-based text refiner."""

    enabled: bool = True
    min_words_to_refine: int = 5
    max_tokens_ratio: float = 1.5


@dataclass
class DiarizationConfig:
    """Speaker diarization settings (Tier 2).

    Uses speechbrain ECAPA-TDNN embeddings with online centroid
    clustering. Runs on system audio turns only — mic turns stay "You".
    """

    enabled: bool = False
    max_speakers: int = 6
    similarity_threshold: float = 0.65
    min_audio_duration: float = 1.0
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class DualStreamConfig:
    """Cross-stream deduplication settings for dual audio pipelines.

    When mic and system audio both capture the same speech (acoustic
    coupling without headphones), the deduplicator suppresses the echo
    using text similarity between streams.
    """

    enabled: bool = True
    window_seconds: float = 8.0  # temporal window for comparing chunks
    similarity_threshold: float = 0.55  # SequenceMatcher ratio to suppress
    short_text_threshold: float = 0.75  # stricter threshold for short chunks
    short_text_min_words: int = 4  # word count below which short_text_threshold applies


@dataclass
class NotionConfig:
    """Notion integration settings for RAG ingestion and meeting export."""

    enabled: bool = False
    api_token_env: str = "NOTION_API_TOKEN"  # env var name (token never in config)
    export_parent_page_id: str = ""
    rag_source_page_ids: List[str] = field(default_factory=list)
    rag_source_database_ids: List[str] = field(default_factory=list)
    max_pages_per_database: int = 100
    max_retries: int = 3
    initial_backoff_s: float = 1.0
    timeout_s: int = 30
    max_block_depth: int = 10


@dataclass
class PathsConfig:
    docs_dir: str = "context"
    output_dir: str = "output"
    index_dir: str = "data/colbert_index"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    rag: RAGPipelineConfig = field(default_factory=RAGPipelineConfig)
    triggers: TriggerConfig = field(default_factory=TriggerConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    dual_stream: DualStreamConfig = field(default_factory=DualStreamConfig)
    notion: NotionConfig = field(default_factory=NotionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build_dataclass(cls: type, data: Optional[dict]) -> object:
    """Build a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults.

    Args:
        config_path: Path to config.yaml. If None, uses lib.paths resolver.

    Returns:
        Populated AppConfig with values from file or defaults.
    """
    from lib.paths import get_config_path

    config_path = get_config_path(config_path)

    if not config_path.exists():
        logger.info("No config.yaml found, using defaults")
        return AppConfig()

    try:
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("pyyaml not installed, using default config")
        return AppConfig()
    except Exception as e:
        logger.warning("Failed to load config.yaml: %s, using defaults", e)
        return AppConfig()

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        audio=_build_dataclass(AudioModelConfig, models_raw.get("audio")),
        generation=_build_dataclass(GenerationModelConfig, models_raw.get("generation")),
        retrieval=_build_dataclass(RetrievalConfig, models_raw.get("retrieval")),
    )

    return AppConfig(
        audio=_build_dataclass(AudioConfig, raw.get("audio")),
        models=models,
        buffer=_build_dataclass(BufferConfig, raw.get("buffer")),
        detection=_build_dataclass(DetectionConfig, raw.get("detection")),
        rag=_build_dataclass(RAGPipelineConfig, raw.get("rag")),
        triggers=_build_dataclass(TriggerConfig, raw.get("triggers")),
        refiner=_build_dataclass(RefinerConfig, raw.get("refiner")),
        diarization=_build_dataclass(DiarizationConfig, raw.get("diarization")),
        dual_stream=_build_dataclass(DualStreamConfig, raw.get("dual_stream")),
        notion=_build_dataclass(NotionConfig, raw.get("notion")),
        paths=_build_dataclass(PathsConfig, raw.get("paths")),
    )
