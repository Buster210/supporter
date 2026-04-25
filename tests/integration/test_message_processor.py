from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from supporter.tui.message_processor import ChatMessageProcessor
from supporter.tui.widgets import ChatTurn


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

    async def mock_stream(*args: Any) -> AsyncGenerator[Any, Any]:
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

    async def mock_stream(*args: Any) -> AsyncGenerator[Any, Any]:
        yield chunk

    agent = MagicMock()
    agent.execute_stream = mock_stream
    target = MagicMock(spec=ChatTurn)

    async def mock_mount_bubble(*args: Any, **kwargs: Any) -> Any:
        return None

    target.mount_bubble = MagicMock(side_effect=mock_mount_bubble)
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
