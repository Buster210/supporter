from unittest.mock import MagicMock

import pytest

from supporter.agent import ChatAgent
from supporter.index import LLMChunk


@pytest.mark.asyncio
async def test_chat_agent_execute_stream() -> None:
    mock_chunks = [
        LLMChunk(text="Hello", is_last=False),
        LLMChunk(text=" world", is_last=False),
    ]

    async def mock_generator(*args: object, **kwargs: object) -> object:
        for chunk in mock_chunks:
            yield chunk

    mock_provider = MagicMock()
    mock_provider.generate_stream = mock_generator

    agent = ChatAgent(mock_provider)

    collected = []
    async for chunk in agent.execute_stream("Hi"):
        collected.append(chunk.text)

    assert "".join(collected) == "Hello world"

    history = agent.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "model"
    assert history[1].parts[0].text == "Hello world"


@pytest.mark.asyncio
async def test_chat_agent_execute_stream_empty() -> None:
    async def mock_empty_generator(*args: object, **kwargs: object) -> object:
        if False:
            yield

    mock_provider = MagicMock()
    mock_provider.generate_stream = mock_empty_generator

    agent = ChatAgent(mock_provider)

    collected = []
    async for chunk in agent.execute_stream("Hi"):
        collected.append(chunk.text)

    assert "".join(collected) == ""
    history = agent.get_history()
    assert len(history) == 2
    assert history[1].parts[0].text == ""
