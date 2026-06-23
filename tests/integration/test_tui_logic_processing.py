from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from supporter.tui.chat import ChatTurn
from supporter.tui.message_processor import ChatMessageProcessor
from tests.tui_mocks import MockApp, MockBubble, MockTurn, MockWidget


@pytest.mark.asyncio
async def test_process_streaming_basic() -> None:
    app = MockApp()
    processor = ChatMessageProcessor(app)
    target = MockWidget()
    agent = MagicMock()

    async def mock_stream(text: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
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
        patch("supporter.tui.bubble.MessageBubble", MockBubble),
        patch("supporter.tui.message_processor.MessageBubble", MockBubble, create=True),
        patch("supporter.tui.chat.ChatTurn", MockTurn),
        patch("supporter.tui.message_processor.ChatTurn", MockTurn, create=True),
    ):
        bubble = await processor.process_streaming("Hi", target, 0.0, agent)
    assert isinstance(bubble, MockBubble)
    assert bubble.content == "Hello World"
    assert bubble.finalized is True
    # After the stream loop ends the label must leave "Streaming" (bug fix:
    # post-turn formatting/verify window otherwise stranded it there).
    assert app.status_label == "Thinking"


@pytest.mark.asyncio
async def test_process_streaming_with_tool_calls() -> None:
    app = MockApp()
    processor = ChatMessageProcessor(app)
    target = MockWidget()
    agent = MagicMock()

    async def mock_stream(text: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
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
        patch("supporter.tui.bubble.MessageBubble", MockBubble),
        patch("supporter.tui.message_processor.MessageBubble", MockBubble, create=True),
        patch("supporter.tui.chat.ChatTurn", MockTurn),
        patch("supporter.tui.message_processor.ChatTurn", MockTurn, create=True),
    ):
        bubble = await processor.process_streaming("task", target, 0.0, agent)
    assert len(bubble.tool_calls) == 1
    assert bubble.tool_calls[0][0] == "read_file"
    assert "Thinking..." in bubble.tokens
    assert "Result" in bubble.tokens
    # After the stream loop ends the label must leave "Streaming" (bug fix:
    # post-turn formatting/verify window otherwise stranded it there).
    assert app.status_label == "Thinking"


@pytest.mark.asyncio
async def test_process_streaming_empty_chunk() -> None:
    app = MagicMock()
    processor = ChatMessageProcessor(app)
    chunk = MagicMock()
    chunk.is_tool_call = False
    chunk.is_last = False
    chunk.text = "   "
    chunk.is_thought = False
    chunk.model = None

    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        yield chunk

    agent = MagicMock()
    agent.execute_stream = mock_stream
    target = MagicMock()
    result = await processor.process_streaming("test", target, 0, agent)
    assert result is None


@pytest.mark.asyncio
async def test_process_streaming_chat_turn_mount() -> None:
    app = MagicMock()
    processor = ChatMessageProcessor(app)
    chunk = MagicMock()
    chunk.is_tool_call = False
    chunk.is_last = False
    chunk.text = "Hello"
    chunk.is_thought = False
    chunk.model = "test-model"

    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        yield chunk

    agent = MagicMock()
    agent.execute_stream = mock_stream
    target = MagicMock(spec=ChatTurn)

    async def mock_mount_bubble(*args: Any, **kwargs: Any) -> Any:
        return None

    target.mount_bubble = MagicMock(side_effect=mock_mount_bubble)
    with patch(
        "supporter.tui.bubble.MessageBubble.app", new_callable=PropertyMock
    ) as mock_app_prop:
        mock_app_prop.return_value = app
        result = await processor.process_streaming("test", target, 0, agent)
    assert result is not None
    target.mount_bubble.assert_called()


@pytest.mark.asyncio
async def test_handle_tool_chunk_google_search() -> None:
    app = MagicMock()
    app.status_label = "Ready"
    processor = ChatMessageProcessor(app)
    processor._handle_tool_call_status("google_search")
    assert app.status_label == "Searching"
