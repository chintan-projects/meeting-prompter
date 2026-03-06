"""Tests for LFM2 wrapper output parsing."""
import pytest

from lib.lfm2_wrapper import LFM2Wrapper


class TestParseOutput:
    """Tests for _parse_output — stripping llama.cpp logs and LFM2.5 metadata."""

    @pytest.fixture
    def wrapper(self, tmp_path):
        """Create a wrapper with dummy model files for testing _parse_output."""
        # Create dummy files so __init__ doesn't raise FileNotFoundError
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        runner_dir = tmp_path / "runners"
        runner_dir.mkdir()

        for name in [
            "LFM2.5-Audio-1.5B-Q4_0.gguf",
            "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
        ]:
            (model_dir / name).touch()
        binary = runner_dir / "llama-liquid-audio-macos-arm64" / "llama-liquid-audio-cli"
        binary.parent.mkdir(parents=True)
        binary.touch()

        return LFM2Wrapper(model_dir, runner_dir, model_version="2.5")

    def test_clean_transcription(self, wrapper):
        """Plain transcription text passes through unchanged."""
        raw = b"Hello, how are you today?\n"
        assert wrapper._parse_output(raw) == "Hello, how are you today?"

    def test_strips_generated_text_marker(self, wrapper):
        """Text after '=== GENERATED TEXT ===' is removed."""
        raw = b"What is the deployment timeline? === GENERATED TEXT === What is the\n"
        result = wrapper._parse_output(raw)
        assert result == "What is the deployment timeline?"
        assert "GENERATED TEXT" not in result

    def test_strips_audio_samples_metadata(self, wrapper):
        """Inline 'audio samples per second: nan' metadata is stripped."""
        raw = b"How does LFM work? audio samples per second:        nan\n"
        result = wrapper._parse_output(raw)
        assert result == "How does LFM work?"
        assert "audio samples" not in result

    def test_strips_both_markers_combined(self, wrapper):
        """Both metadata markers in one output are handled."""
        raw = (
            b"Okay, can you explain to me how does ZLFM work? "
            b"audio samples per second:        nan "
            b"=== GENERATED TEXT === repeated text\n"
        )
        result = wrapper._parse_output(raw)
        assert result == "Okay, can you explain to me how does ZLFM work?"

    def test_skips_llama_cpp_logging(self, wrapper):
        """llama.cpp log lines are filtered out."""
        raw = (
            b"llama_model_loader: loaded meta data\n"
            b"loading model from path\n"
            b"ggml_metal_init: using Metal\n"
            b"Hello world\n"
        )
        assert wrapper._parse_output(raw) == "Hello world"

    def test_skips_timing_lines(self, wrapper):
        """Lines ending in 'ms' are timing info and filtered."""
        raw = b"transcription text\n47 ms\nencoding audio took 123 ms\n"
        assert wrapper._parse_output(raw) == "transcription text"

    def test_empty_output(self, wrapper):
        """Empty output returns empty string."""
        assert wrapper._parse_output(b"") == ""
        assert wrapper._parse_output(b"\n\n\n") == ""

    def test_only_metadata_returns_empty(self, wrapper):
        """If output is only metadata, return empty string."""
        raw = b"audio samples per second:        nan === GENERATED TEXT ===\n"
        assert wrapper._parse_output(raw) == ""

    def test_nan_line_filtered(self, wrapper):
        """Standalone 'nan' lines are filtered."""
        raw = b"Hello\nnan\nworld\n"
        assert wrapper._parse_output(raw) == "Hello world"

    def test_multiline_with_logging_and_speech(self, wrapper):
        """Real-world output with logging, speech, and metadata."""
        raw = (
            b"llama_model_loader: loaded meta data with 26 key-value pairs\n"
            b"loading model from /path/to/model.gguf\n"
            b"ggml_metal_init: allocating\n"
            b"[main] initializing\n"
            b"encoding audio slice 0/0\n"
            b"What are example use cases of LFM? "
            b"audio samples per second:        nan "
            b"=== GENERATED TEXT === What are example\n"
        )
        result = wrapper._parse_output(raw)
        assert result == "What are example use cases of LFM?"
