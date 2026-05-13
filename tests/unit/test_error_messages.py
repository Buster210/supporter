from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.containers import Vertical

from supporter.tools.base import ToolError


def _make_app() -> tuple[MagicMock, MagicMock]:
    app = MagicMock()
    app.agent = MagicMock()
    app._is_processing = False
    app._user_message_queue = []
    app.active_queries = 0
    app.status_label = "Thinking"

    chat_view = MagicMock(spec=Vertical)
    chat_view.mount = AsyncMock()
    chat_view.turn_start_time = 0.0
    app.query_one = MagicMock(return_value=chat_view)
    app.active_turn = chat_view

    app._mount_user_turn = AsyncMock()
    app._start_thinking = MagicMock()
    app._stop_thinking = MagicMock()
    app._flush_queued_messages = AsyncMock()

    return app, chat_view


@pytest.mark.asyncio
async def test_tool_error_renders_user_message() -> None:
    from supporter.tui import SupporterApp

    app, chat_view = _make_app()
    app._process_streaming_execution = AsyncMock(
        side_effect=ToolError("File not found: config.yaml")
    )

    bound = SupporterApp._process_message_cycle.__get__(app, SupporterApp)
    await bound("hello", mount_user=False)

    mounted_bubble = chat_view.mount.call_args[0][0]
    assert mounted_bubble.content == "File not found: config.yaml"


@pytest.mark.asyncio
async def test_tool_error_does_not_crash_continues_processing() -> None:
    from supporter.tui import SupporterApp

    app, _chat_view = _make_app()
    app._process_streaming_execution = AsyncMock(side_effect=ToolError("Oops"))

    bound = SupporterApp._process_message_cycle.__get__(app, SupporterApp)
    await bound("test", mount_user=False)

    assert app._is_processing is False
    app._flush_queued_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_generic_exception_still_uses_sanitized_message() -> None:
    from supporter.tui import SupporterApp

    app, chat_view = _make_app()
    app._process_streaming_execution = AsyncMock(side_effect=RuntimeError("boom"))

    bound = SupporterApp._process_message_cycle.__get__(app, SupporterApp)
    await bound("test", mount_user=False)

    mounted_bubble = chat_view.mount.call_args[0][0]
    assert "An error occurred" in mounted_bubble.content
    assert "boom" not in mounted_bubble.content


def test_tool_error_exposes_user_message() -> None:
    error = ToolError("Cannot read file")
    assert str(error) == "Cannot read file"
    assert error.user_message == "Cannot read file"
