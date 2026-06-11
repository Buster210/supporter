"""Tests for opencode CLI backend."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tools.delegate.opencode_backend import (
    OPENCODE_BIN,
    _resolve_binary,
    run_opencode,
)


class TestResolveBinary:
    def test_returns_none_when_neither_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert _resolve_binary() is None

    def test_returns_fixed_bin_when_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        assert _resolve_binary() == OPENCODE_BIN

    def test_falls_back_to_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/opencode")
        assert _resolve_binary() == "/usr/local/bin/opencode"


class TestRunOpencode:
    @pytest.mark.asyncio
    async def test_raises_when_binary_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "supporter.tools.delegate.opencode_backend._resolve_binary",
            lambda: None,
        )
        with pytest.raises(RuntimeError, match="opencode CLI not found"):
            await run_opencode({"id": "t1", "task": "do something", "timeout": 5})

    @pytest.mark.asyncio
    async def test_model_arg_appended_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When OPENCODE_MODEL env var is set, -m <model> is appended to argv."""
        monkeypatch.setattr(
            "supporter.tools.delegate.opencode_backend._resolve_binary",
            lambda: "/usr/bin/opencode",
        )
        monkeypatch.setenv("OPENCODE_MODEL", "claude-3-haiku")

        captured: list[list[str]] = []

        async def fake_stream_stdout() -> None:
            return None

        async def fake_proc_wait() -> int:
            return 0

        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(return_value=b"")
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.returncode = 0

        async def fake_create_subprocess_exec(
            *args: str, **kwargs: object
        ) -> MagicMock:
            captured.append(list(args))
            return fake_proc

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ):
            task = {"id": "t2", "task": "run tests", "timeout": 5}
            await run_opencode(task)

        assert captured, "subprocess was never called"
        argv = captured[0]
        assert "-m" in argv
        assert "claude-3-haiku" in argv
