"""LFM2/LFM2.5 Audio Wrapper - Subprocess interface to llama.cpp for ASR."""
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Model file mappings per version
_MODEL_FILES = {
    "2.0": {
        "model": "LFM2-Audio-1.5B-Q8_0.gguf",
        "mmproj": "mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf",
        "vocoder": "audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf",
        "tokenizer": None,
        "binary": "lfm2-audio-macos-arm64/llama-lfm2-audio",
        "mmproj_flag": "--mmproj",
    },
    "2.5": {
        "model": "LFM2.5-Audio-1.5B-Q4_0.gguf",
        "mmproj": "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
        "vocoder": "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
        "tokenizer": "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
        "binary": "llama-liquid-audio-macos-arm64/llama-liquid-audio-cli",
        "mmproj_flag": "-mm",
    },
}


class LFM2Wrapper:
    """Wrapper for llama.cpp audio binary (supports LFM2 and LFM2.5)."""

    def __init__(
        self,
        model_dir: Path,
        runner_dir: Path,
        model_version: str = "2.5",
        timeout: int = 30,
    ) -> None:
        if model_version not in _MODEL_FILES:
            raise ValueError(f"Unsupported model version: {model_version}. Use '2.0' or '2.5'")

        files = _MODEL_FILES[model_version]
        self.model = model_dir / files["model"]
        self.mmproj = model_dir / files["mmproj"]
        self.vocoder = model_dir / files["vocoder"]
        self.tokenizer: Optional[Path] = (
            model_dir / files["tokenizer"] if files["tokenizer"] else None
        )
        self.runner = runner_dir / files["binary"]
        self.mmproj_flag = files["mmproj_flag"]
        self.timeout = timeout

        # Validate required files exist
        required = [self.model, self.mmproj, self.vocoder, self.runner]
        for f in required:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")
        if self.tokenizer and not self.tokenizer.exists():
            raise FileNotFoundError(f"Required tokenizer not found: {self.tokenizer}")

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file to text using ASR."""
        cmd = [
            str(self.runner),
            "-m", str(self.model),
            self.mmproj_flag, str(self.mmproj),
            "-mv", str(self.vocoder),
            "--audio", str(audio_path),
            "-sys", "Perform ASR.",
            "--temp", "0",
        ]
        # LFM2.5 requires the tokenizer file for the detokenizer
        if self.tokenizer:
            cmd.extend(["--tts-speaker-file", str(self.tokenizer)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout,
                cwd=self.runner.parent,
            )
            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("Transcription timed out after %ds", self.timeout)
            return "[Transcription timeout]"
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return f"[Error: {e}]"

    def _parse_output(self, raw: bytes) -> str:
        """Filter llama.cpp logging, extract clean transcription."""
        text = raw.decode('utf-8', errors='replace')

        # LFM2.5 outputs transcription followed by metadata markers.
        # Strip everything from "=== GENERATED TEXT ===" onwards.
        gen_marker = "=== GENERATED TEXT ==="
        idx = text.find(gen_marker)
        if idx != -1:
            text = text[:idx]

        # Also strip "audio samples per second:" metadata (appears inline)
        audio_marker = "audio samples per second:"
        idx = text.lower().find(audio_marker.lower())
        if idx != -1:
            text = text[:idx]

        lines = text.split('\n')

        # Skip llama.cpp verbose logging
        skip_keywords = [
            'loading', 'loaded', 'gguf', 'tensors', 'model', 'backend',
            'metal', 'gpu', 'cpu', 'simd', 'memory', 'init', 'build',
            'llama_', 'ggml_', 'load_', 'mtmd', 'sampler', 'token',
            'system_info', 'n_ctx', 'n_batch', 'flash_attn',
            'encoding audio', 'audio slice', 'audio decoded', 'decoding audio',
            'clip_', 'alloc_', 'print_info', 'common_init', 'main:',
            'audio samples per second', 'generated text',
        ]

        clean_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            if any(kw in line_lower for kw in skip_keywords):
                continue
            # Skip lines that look like logging (contain timestamps, brackets, etc.)
            if line.startswith('[') or line.startswith('llama') or line.startswith('---'):
                continue
            # Skip timing info like "47 ms"
            if line_lower.endswith(' ms') or line_lower.endswith(' ms)'):
                continue
            # Skip lines that are just "nan" or whitespace-padded numbers
            if line_lower.strip() in ('nan', 'inf', '-inf'):
                continue
            clean_lines.append(line.strip())

        return ' '.join(clean_lines).strip()
