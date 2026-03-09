"""Central path resolver — dev vs packaged mode.

In dev mode, paths resolve relative to the repo root (Path(__file__).parent.parent).
In packaged mode (.app bundle), the MEETING_PROMPTER_ROOT env var points to the
source tree, and mutable data lives in ~/Library/Application Support/.

All modules should import from here instead of computing BASE_DIR locally.
"""

import os
from pathlib import Path
from typing import Optional

_APP_SUPPORT_NAME = "com.meetingprompter.app"


def is_packaged() -> bool:
    """True when running inside a packaged .app bundle."""
    return bool(os.environ.get("MEETING_PROMPTER_ROOT"))


def get_project_root() -> Path:
    """Root of the project source tree."""
    env_root = os.environ.get("MEETING_PROMPTER_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).parent.parent


def get_models_dir() -> Path:
    """Directory containing LFM model files."""
    env_models = os.environ.get("MODELS_DIR")
    if env_models:
        return Path(os.path.expandvars(env_models))
    return get_project_root() / "models"


def get_runner_dir() -> Path:
    """Directory containing llama.cpp binaries."""
    return get_project_root() / "runners" / "macos-arm64"


def get_docs_dir(config_docs_dir: Optional[str] = None) -> Path:
    """Directory containing RAG source documents."""
    subdir = config_docs_dir or "context"
    return get_project_root() / subdir


def get_data_dir() -> Path:
    """Directory for mutable data (RAG index, etc.)."""
    if is_packaged():
        return _app_support() / "data"
    return get_project_root() / "data"


def get_output_dir() -> Path:
    """Directory for session recordings and meeting notes."""
    if is_packaged():
        return _app_support() / "output"
    return get_project_root() / "output"


def get_config_path(override: Optional[Path] = None) -> Path:
    """Path to config.yaml."""
    if override:
        return override
    if is_packaged():
        app_config = _app_support() / "config.yaml"
        if app_config.exists():
            return app_config
    return get_project_root() / "config.yaml"


def _app_support() -> Path:
    """macOS Application Support directory for mutable data."""
    base = Path.home() / "Library" / "Application Support" / _APP_SUPPORT_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base
