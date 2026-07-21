"""Tests for the /session/start permission gate (BUG-005).

Per-app capture (system_audio_pid > 0) is a dual-stream guarantee. When Screen
Recording permission is missing it used to silently degrade to mic-only. The
backend is the authoritative gate: it must refuse to start under a false
dual-stream expectation. Explicit mic-only (pid == 0) must bypass the gate.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

import src.api.routes.session as routes_session
from src.api.routes.session import StartRequest, start_session


@pytest.fixture(autouse=True)
def _reset_session_singleton():
    """Ensure each test starts with a clean module-level session singleton."""
    routes_session._session = None
    yield
    routes_session._session = None


@pytest.mark.asyncio
async def test_per_app_start_blocked_when_permission_denied() -> None:
    """pid > 0 with denied permission → 412, and the pipeline never starts."""
    with (
        patch(
            "lib.system_audio_capture.SystemAudioCapture.check_permission",
            return_value=False,
        ),
        patch("src.api.session.Session.start") as mock_start,
    ):
        with pytest.raises(HTTPException) as exc:
            await start_session(
                StartRequest(system_audio_pid=1234, system_audio_app="Google Chrome")
            )

    assert exc.value.status_code == 412
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail["error"] == "screen_recording_permission_denied"
    assert "remedy" in detail
    # The session pipeline must not have started under a false dual-stream promise.
    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_per_app_start_allowed_when_permission_granted() -> None:
    """pid > 0 with granted permission → starts in app_tap mode."""
    with (
        patch(
            "lib.system_audio_capture.SystemAudioCapture.check_permission",
            return_value=True,
        ),
        patch("src.api.session.Session.start") as mock_start,
    ):
        result = await start_session(
            StartRequest(system_audio_pid=1234, system_audio_app="Google Chrome")
        )

    assert result["status"] == "started"
    assert result["capture_mode"] == "app_tap"
    mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_mic_only_start_bypasses_permission_gate() -> None:
    """pid == 0 is explicit mic-only — it must never be gated on permission.

    check_permission is forced False to prove the gate is not even consulted
    for the mic-only path (no false dual-stream expectation to protect).
    """
    with (
        patch(
            "lib.system_audio_capture.SystemAudioCapture.check_permission",
            return_value=False,
        ) as mock_check,
        patch("src.api.session.Session.start") as mock_start,
    ):
        result = await start_session(StartRequest(system_audio_pid=0))

    assert result["status"] == "started"
    assert result["capture_mode"] == "device"
    mock_start.assert_called_once()
    mock_check.assert_not_called()
