from unittest.mock import MagicMock

import pytest
from google.genai.types import Content, Part

from supporter.agent import ChatAgent
from supporter.pool import LLMChunk, LLMResult


def _build_user_message(prompt: str) -> Content:
    return Content(role="user", parts=[Part(text=prompt)])


def _extract_assistant_message(result: LLMResult) -> Content | None:
    if not result.candidates or not result.candidates[0].content:
        return None
    return Content(role="model", parts=result.candidates[0].content.parts)


def _build_assistant_message(text: str) -> Content:
    return Content(role="model", parts=[Part(text=text)])


def _wire_message_methods(mock_provider: MagicMock) -> None:
    mock_provider.build_user_message = _build_user_message
    mock_provider.extract_assistant_message = _extract_assistant_message
    mock_provider.build_assistant_message = _build_assistant_message


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
    _wire_message_methods(mock_provider)
    agent = ChatAgent(mock_provider)
    collected = []
    async for chunk in agent.execute_stream("Hi"):
        collected.append(chunk.text)
    assert "".join(collected) == "Hello world"
    history = agent.history
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "model"
    assert history[1].parts and history[1].parts[0].text == "Hello world"


@pytest.mark.asyncio
async def test_chat_agent_execute_stream_empty() -> None:

    async def mock_empty_generator(*args: object, **kwargs: object) -> object:
        if False:
            yield

    mock_provider = MagicMock()
    mock_provider.generate_stream = mock_empty_generator
    _wire_message_methods(mock_provider)
    agent = ChatAgent(mock_provider)
    collected = []
    async for chunk in agent.execute_stream("Hi"):
        collected.append(chunk.text)
    assert "".join(collected) == ""
    history = agent.history
    assert len(history) == 2
    assert history[1].parts and history[1].parts[0].text == ""
