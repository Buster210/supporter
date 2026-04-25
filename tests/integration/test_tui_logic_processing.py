from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.tui.message_processor import ChatMessageProcessor
from tests.tui_mocks import MockApp, MockBubble, MockTurn, MockWidget


@pytest.mark.asyncio
async def test_process_streaming_basic() -> None:
    app = MockApp()
    processor = ChatMessageProcessor(app)
    target = MockWidget()
    agent = MagicMock()

    async def mock_stream(text: Any) -> AsyncGenerator[Any, Any]:
        yield MagicMock(
            text="Hello",
            is_tool_call=False,
            is_last=False,
            model="gemini-3.1",
            is_thought=False,
        )
        yield MagicMock(
            text=" World",
            is_tool_call=False,
            is_last=True,
            model="gemini-3.1",
            is_thought=False,
        )

    agent.execute_stream.side_effect = mock_stream
    with (
        patch("supporter.tui.widgets.MessageBubble", MockBubble),
        patch("supporter.tui.message_processor.MessageBubble", MockBubble, create=True),
        patch("supporter.tui.widgets.ChatTurn", MockTurn),
        patch("supporter.tui.message_processor.ChatTurn", MockTurn, create=True),
    ):
        bubble = await processor.process_streaming("Hi", target, 0.0, agent)
    assert isinstance(bubble, MockBubble)
    assert bubble.content == "Hello World"
    assert bubble.finalized is True
    assert app.status_label == "Streaming"


@pytest.mark.asyncio
async def test_process_streaming_with_tool_calls() -> None:
    app = MockApp()
    processor = ChatMessageProcessor(app)
    target = MockWidget()
    agent = MagicMock()

    async def mock_stream(text: Any) -> AsyncGenerator[Any, Any]:
        yield MagicMock(
            is_tool_call=True,
            tool_name="read_file",
            tool_args={"path": "a.txt"},
            is_last=False,
            model=None,
            text="",
        )
        yield MagicMock(
            is_tool_call=False,
            is_thought=True,
            text="Thinking...",
            is_last=False,
            model=None,
        )
        yield MagicMock(
            is_tool_call=False,
            is_thought=False,
            text="Result",
            is_last=True,
            model="gemini-3.1",
        )

    agent.execute_stream.side_effect = mock_stream
    with (
        patch("supporter.tui.widgets.MessageBubble", MockBubble),
        patch("supporter.tui.message_processor.MessageBubble", MockBubble, create=True),
        patch("supporter.tui.widgets.ChatTurn", MockTurn),
        patch("supporter.tui.message_processor.ChatTurn", MockTurn, create=True),
    ):
        bubble = await processor.process_streaming("task", target, 0.0, agent)
    assert len(bubble.tool_calls) == 1
    assert bubble.tool_calls[0][0] == "read_file"
    assert "Thinking..." in bubble.tokens
    assert "Result" in bubble.tokens
    assert app.status_label == "Streaming"
