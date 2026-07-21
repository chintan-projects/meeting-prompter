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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from lib.config import DiarizationConfig

if TYPE_CHECKING:
    from lib.speaker_enrollment import VoiceEnrollment

logger = logging.getLogger(__name__)

# Label format for auto-discovered speakers
_SPEAKER_PREFIX = "Speaker"
_SPEAKER_LABELS = [f"{_SPEAKER_PREFIX} {chr(65 + i)}" for i in range(26)]  # Speaker A .. Speaker Z


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

        # Roster-bounded clustering (F-604): when the expected participant count
        # is known (from meeting context), it caps the number of clusters instead
        # of the looser max_speakers default.
        self._roster_size: Optional[int] = None

        # Per-cluster names, parallel to _centroids: a real name when the cluster
        # was identified by voice enrollment (F-605), else None → "Speaker X".
        self._names: List[Optional[str]] = []
        self._enrollment: Optional["VoiceEnrollment"] = None

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

    def set_roster_size(self, size: Optional[int]) -> None:
        """Bound clustering to the known roster size (F-604).

        When set (> 0), the diarizer creates at most ``size`` clusters and
        re-assigns further speech to the nearest existing speaker — the roster
        is the ground-truth ceiling on distinct remote speakers.
        """
        self._roster_size = size if (size and size > 0) else None
        logger.info("Diarization roster size: %s", self._roster_size)

    def _max_clusters(self) -> int:
        """Effective cluster cap: roster size when known, else max_speakers."""
        if self._roster_size is not None:
            return self._roster_size
        return self._config.max_speakers

    def set_enrollment(self, enrollment: Optional["VoiceEnrollment"]) -> None:
        """Attach voice-enrollment profiles that name matching clusters (F-605)."""
        self._enrollment = enrollment
        logger.info(
            "Diarization voice enrollment: %d profiles",
            len(enrollment) if enrollment else 0,
        )

    def _default_label(self, idx: int) -> str:
        """Anonymous label for cluster index (Speaker A, B, ...)."""
        return _SPEAKER_LABELS[idx] if idx < len(_SPEAKER_LABELS) else f"{_SPEAKER_PREFIX} {idx}"

    def _label_for(self, idx: int) -> str:
        """Enrolled name for this cluster if known, else the anonymous label."""
        if idx < len(self._names) and self._names[idx]:
            return self._names[idx]  # type: ignore[return-value]
        return self._default_label(idx)

    def process_turn(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[str]:
        """Assign the dominant speaker label for a turn's audio.

        A turn is not guaranteed to be one speaker: with a shared far-field mic,
        a second person can interject mid-turn. So the turn is segmented at
        speaker-change points (F-604) and each homogeneous slice is assigned;
        the label covering the most windows is returned. Backward-compatible —
        a single-speaker turn still yields one label.

        Returns:
            Speaker label (e.g. "Speaker A") or None if unavailable/too short.
        """
        segments = self.process_turn_segments(audio_data, sample_rate)
        if not segments:
            return None
        # Dominant speaker by total window span.
        by_speaker: Dict[str, int] = {}
        for label, w_start, w_end in segments:
            by_speaker[label] = by_speaker.get(label, 0) + (w_end - w_start)
        return max(by_speaker, key=lambda k: by_speaker[k])

    def process_turn_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Tuple[str, int, int]]:
        """Segment a turn at speaker-change points and label each slice.

        Windows the audio, embeds each window, detects speaker-change boundaries
        by a drop in adjacent-window cosine similarity, then assigns each
        homogeneous slice to a (roster-bounded) speaker cluster.

        Returns:
            List of (speaker_label, window_start_idx, window_end_idx). Empty if
            the model is unavailable or the turn is below min_audio_duration.
        """
        if not self.available:
            return []

        duration = len(audio_data) / sample_rate
        if duration < self._config.min_audio_duration:
            logger.debug("Turn too short (%.1fs) for diarization", duration)
            return []

        windows = self._window_audio(audio_data, sample_rate)
        with self._lock:
            embeddings: List[np.ndarray] = []
            for w in windows:
                emb = self._extract_embedding(w, sample_rate)
                if emb is not None:
                    embeddings.append(emb)
            if not embeddings:
                return []

            boundaries = self.detect_change_points(embeddings, self._config.change_threshold)
            # Group windows into homogeneous slices, assign each to a speaker.
            segments: List[Tuple[str, int, int]] = []
            start = 0
            cut_points = boundaries + [len(embeddings)]
            for cut in cut_points:
                slice_embs = embeddings[start:cut]
                if not slice_embs:
                    continue
                mean_emb = np.mean(np.stack(slice_embs), axis=0).astype(np.float32)
                label = self._assign_speaker(mean_emb)
                segments.append((label, start, cut))
                start = cut
            return segments

    def _window_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Slice audio into overlapping windows for intra-turn segmentation.

        Short turns (< 2 windows) return a single whole-turn window, preserving
        the original single-embedding behavior.
        """
        win = max(1, int(self._config.window_seconds * sample_rate))
        hop = max(1, int(self._config.window_hop_seconds * sample_rate))
        if len(audio_data) <= win + hop:
            return [audio_data]
        windows: List[np.ndarray] = []
        start = 0
        while start < len(audio_data):
            chunk = audio_data[start : start + win]
            if len(chunk) < win // 2 and windows:
                break  # trailing sliver — fold into the previous window
            windows.append(chunk)
            start += hop
        return windows or [audio_data]

    @staticmethod
    def detect_change_points(embeddings: List[np.ndarray], threshold: float) -> List[int]:
        """Indices where adjacent windows differ enough to mark a speaker change.

        Returns boundary indices i (1..n-1) where cosine(emb[i-1], emb[i]) is
        below ``threshold`` — i.e. a new segment starts at index i.
        """
        boundaries: List[int] = []
        for i in range(1, len(embeddings)):
            sim = SpeakerDiarizer._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                boundaries.append(i)
        return boundaries

    def _extract_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
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
            self._names.append(self._identify_name(embedding))
            label = self._label_for(0)
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
            # Backfill an enrolled name if this cluster is still anonymous.
            if best_idx < len(self._names) and self._names[best_idx] is None:
                self._names[best_idx] = self._identify_name(embedding)
            label = self._label_for(best_idx)
            logger.debug("Matched %s (sim=%.3f, count=%d)", label, best_sim, count + 1)
            return label

        if len(self._centroids) < self._max_clusters():
            # New speaker
            self._centroids.append((embedding.copy(), 1))
            idx = len(self._centroids) - 1
            self._names.append(self._identify_name(embedding))
            label = self._label_for(idx)
            logger.info("New speaker: %s (sim=%.3f to nearest)", label, best_sim)
            return label

        # At the cluster cap (roster-bounded) — re-assign to the closest speaker
        old_centroid, count = self._centroids[best_idx]
        new_centroid = (old_centroid * count + embedding) / (count + 1)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self._centroids[best_idx] = (new_centroid, count + 1)
        label = self._label_for(best_idx)
        logger.debug(
            "Max speakers reached — assigned to %s (sim=%.3f)",
            label,
            best_sim,
        )
        return label

    def _identify_name(self, embedding: np.ndarray) -> Optional[str]:
        """Name this voice via enrollment profiles, or None if unknown (F-605)."""
        if self._enrollment is None:
            return None
        return self._enrollment.identify(embedding)

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
            self._names.clear()
            logger.info("Speaker diarization state reset")

    def get_speaker_summary(self) -> Dict[str, int]:
        """Get summary of identified speakers and their turn counts."""
        summary: Dict[str, int] = {}
        for i, (_centroid, count) in enumerate(self._centroids):
            summary[self._label_for(i)] = count
        return summary
