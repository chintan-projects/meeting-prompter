"""Speaker diarization — ECAPA-TDNN embeddings with online clustering.

Tier 2 speaker attribution: distinguishes individual remote speakers
on the system audio stream. Runs per-finalized-turn (not streaming),
so it adds zero latency to transcription.

Architecture:
    Finalized turn audio → ECAPA-TDNN embedding (192-dim) → cosine
    similarity against speaker centroids → assign or create speaker.

Thread safety: all model inference guarded by a lock so multiple
pipeline threads can call process_turn() concurrently.
"""
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from lib.config import DiarizationConfig

logger = logging.getLogger(__name__)

# Label format for auto-discovered speakers
_SPEAKER_PREFIX = "Speaker"
_SPEAKER_LABELS = [
    f"{_SPEAKER_PREFIX} {chr(65 + i)}" for i in range(26)
]  # Speaker A .. Speaker Z


class SpeakerDiarizer:
    """Turn-level speaker diarization using ECAPA-TDNN embeddings.

    For each finalized system-audio turn, extracts a speaker embedding
    and assigns a consistent label via online centroid clustering.

    Args:
        config: Diarization settings (thresholds, model name, limits).
    """

    def __init__(self, config: DiarizationConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Speaker state: list of (centroid_embedding, count) pairs
        self._centroids: List[Tuple[np.ndarray, int]] = []

        # Load speechbrain embedding model
        self._model: Optional[object] = None
        self._classifier: Optional[object] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the speechbrain ECAPA-TDNN speaker embedding model."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("Loading speaker embedding model: %s", self._config.embedding_model)
            self._classifier = EncoderClassifier.from_hparams(
                source=self._config.embedding_model,
                run_opts={"device": "cpu"},
            )
            logger.info("Speaker embedding model loaded")
        except ImportError:
            logger.warning(
                "speechbrain not installed — diarization disabled. "
                "Install with: pip install speechbrain torch"
            )
        except Exception as e:
            logger.warning("Failed to load speaker embedding model: %s", e)

    @property
    def available(self) -> bool:
        """Whether the embedding model is loaded and ready."""
        return self._classifier is not None

    @property
    def speaker_count(self) -> int:
        """Number of distinct speakers identified so far."""
        return len(self._centroids)

    def process_turn(
        self, audio_data: np.ndarray, sample_rate: int = 16000,
    ) -> Optional[str]:
        """Extract embedding and assign speaker label for a turn's audio.

        Args:
            audio_data: Raw audio samples (1D float32 array).
            sample_rate: Sample rate in Hz.

        Returns:
            Speaker label (e.g. "Speaker A") or None if unavailable.
        """
        if not self.available:
            return None

        duration = len(audio_data) / sample_rate
        if duration < self._config.min_audio_duration:
            logger.debug("Turn too short (%.1fs) for diarization", duration)
            return None

        with self._lock:
            embedding = self._extract_embedding(audio_data, sample_rate)
            if embedding is None:
                return None
            return self._assign_speaker(embedding)

    def _extract_embedding(
        self, audio_data: np.ndarray, sample_rate: int,
    ) -> Optional[np.ndarray]:
        """Extract a speaker embedding from raw audio.

        Returns:
            192-dim float32 embedding vector, or None on failure.
        """
        try:
            import torch

            # speechbrain expects (batch, time) tensor
            waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

            # Resample if needed (ECAPA-TDNN expects 16kHz)
            if sample_rate != 16000:
                import torchaudio

                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            embeddings = self._classifier.encode_batch(waveform)
            # Shape: (1, 1, 192) → (192,)
            return embeddings.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning("Embedding extraction failed: %s", e)
            return None

    def _assign_speaker(self, embedding: np.ndarray) -> str:
        """Assign a speaker label via online centroid clustering.

        Algorithm:
        1. Compute cosine similarity to all existing centroids
        2. If best match > threshold → assign that speaker, update centroid
        3. If no match and under max_speakers → create new speaker
        4. If no match and at max_speakers → assign closest speaker

        Returns:
            Speaker label (e.g. "Speaker A").
        """
        if len(self._centroids) == 0:
            self._centroids.append((embedding.copy(), 1))
            label = _SPEAKER_LABELS[0]
            logger.info("New speaker: %s (first speaker)", label)
            return label

        # Compute cosine similarities
        similarities = []
        for centroid, _count in self._centroids:
            sim = self._cosine_similarity(embedding, centroid)
            similarities.append(sim)

        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]

        if best_sim >= self._config.similarity_threshold:
            # Match — update centroid as running average
            old_centroid, count = self._centroids[best_idx]
            new_centroid = (old_centroid * count + embedding) / (count + 1)
            # Re-normalize to unit length
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            self._centroids[best_idx] = (new_centroid, count + 1)
            label = _SPEAKER_LABELS[best_idx] if best_idx < len(_SPEAKER_LABELS) else f"{_SPEAKER_PREFIX} {best_idx}"
            logger.debug("Matched %s (sim=%.3f, count=%d)", label, best_sim, count + 1)
            return label

        if len(self._centroids) < self._config.max_speakers:
            # New speaker
            self._centroids.append((embedding.copy(), 1))
            idx = len(self._centroids) - 1
            label = _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS) else f"{_SPEAKER_PREFIX} {idx}"
            logger.info("New speaker: %s (sim=%.3f to nearest)", label, best_sim)
            return label

        # At max speakers — assign to closest
        old_centroid, count = self._centroids[best_idx]
        new_centroid = (old_centroid * count + embedding) / (count + 1)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self._centroids[best_idx] = (new_centroid, count + 1)
        label = _SPEAKER_LABELS[best_idx] if best_idx < len(_SPEAKER_LABELS) else f"{_SPEAKER_PREFIX} {best_idx}"
        logger.debug(
            "Max speakers reached — assigned to %s (sim=%.3f)", label, best_sim,
        )
        return label

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def reset(self) -> None:
        """Clear all speaker clusters for a new meeting."""
        with self._lock:
            self._centroids.clear()
            logger.info("Speaker diarization state reset")

    def get_speaker_summary(self) -> Dict[str, int]:
        """Get summary of identified speakers and their turn counts."""
        summary: Dict[str, int] = {}
        for i, (_centroid, count) in enumerate(self._centroids):
            label = _SPEAKER_LABELS[i] if i < len(_SPEAKER_LABELS) else f"{_SPEAKER_PREFIX} {i}"
            summary[label] = count
        return summary
