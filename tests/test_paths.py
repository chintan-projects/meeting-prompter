"""Tests for lib.paths — central path resolver (dev vs packaged mode)."""

from pathlib import Path
from unittest.mock import patch


from lib.paths import (
    get_config_path,
    get_data_dir,
    get_docs_dir,
    get_models_dir,
    get_output_dir,
    get_project_root,
    get_runner_dir,
    is_packaged,
)


class TestDevMode:
    """In dev mode, no env var is set — paths resolve from source tree."""

    def test_is_packaged_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert is_packaged() is False

    def test_project_root(self) -> None:
        root = get_project_root()
        assert root.is_dir()
        assert (root / "lib" / "paths.py").exists()

    def test_models_dir_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            models = get_models_dir()
            assert models == get_project_root() / "models"

    def test_models_dir_env_override(self) -> None:
        with patch.dict("os.environ", {"MODELS_DIR": "/tmp/test-models"}):
            assert get_models_dir() == Path("/tmp/test-models")

    def test_runner_dir(self) -> None:
        runner = get_runner_dir()
        assert runner == get_project_root() / "runners" / "macos-arm64"

    def test_docs_dir_default(self) -> None:
        docs = get_docs_dir()
        assert docs == get_project_root() / "context"

    def test_docs_dir_custom(self) -> None:
        docs = get_docs_dir("my-docs")
        assert docs == get_project_root() / "my-docs"

    def test_data_dir(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            data = get_data_dir()
            assert data == get_project_root() / "data"

    def test_output_dir(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            output = get_output_dir()
            assert output == get_project_root() / "output"

    def test_config_path_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = get_config_path()
            assert config == get_project_root() / "config.yaml"

    def test_config_path_override(self) -> None:
        override = Path("/tmp/custom-config.yaml")
        assert get_config_path(override) == override


class TestPackagedMode:
    """In packaged mode, MEETING_PROMPTER_ROOT env var is set."""

    def test_is_packaged_true(self) -> None:
        with patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/meeting-prompter"}):
            assert is_packaged() is True

    def test_project_root_from_env(self) -> None:
        with patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/meeting-prompter"}):
            assert get_project_root() == Path("/opt/meeting-prompter")

    def test_data_dir_in_app_support(self, tmp_path: Path) -> None:
        app_support = tmp_path / "Library" / "Application Support" / "com.meetingprompter.app"
        with (
            patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/mp"}),
            patch("lib.paths._app_support", return_value=app_support),
        ):
            data = get_data_dir()
            assert data == app_support / "data"

    def test_output_dir_in_app_support(self, tmp_path: Path) -> None:
        app_support = tmp_path / "Library" / "Application Support" / "com.meetingprompter.app"
        with (
            patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/mp"}),
            patch("lib.paths._app_support", return_value=app_support),
        ):
            output = get_output_dir()
            assert output == app_support / "output"

    def test_config_path_packaged_fallback(self, tmp_path: Path) -> None:
        """When packaged and no app support config, falls back to project root."""
        app_support = tmp_path / "Library" / "Application Support" / "com.meetingprompter.app"
        app_support.mkdir(parents=True)
        with (
            patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/mp"}),
            patch("lib.paths._app_support", return_value=app_support),
        ):
            config = get_config_path()
            assert config == Path("/opt/mp") / "config.yaml"

    def test_config_path_packaged_app_support(self, tmp_path: Path) -> None:
        """When packaged and app support config exists, use it."""
        app_support = tmp_path / "Library" / "Application Support" / "com.meetingprompter.app"
        app_support.mkdir(parents=True)
        config_file = app_support / "config.yaml"
        config_file.write_text("audio:\n  sample_rate: 16000\n")
        with (
            patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/mp"}),
            patch("lib.paths._app_support", return_value=app_support),
        ):
            config = get_config_path()
            assert config == config_file

    def test_runner_dir_from_env_root(self) -> None:
        with patch.dict("os.environ", {"MEETING_PROMPTER_ROOT": "/opt/mp"}):
            runner = get_runner_dir()
            assert runner == Path("/opt/mp") / "runners" / "macos-arm64"


class TestAppSupport:
    """Tests for _app_support directory creation."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        from lib.paths import _app_support

        with patch("lib.paths.Path.home", return_value=tmp_path):
            result = _app_support()
            assert result.exists()
            assert result.is_dir()
            assert result.name == "com.meetingprompter.app"
