from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types
from google.genai.types import Content, Part

from supporter.agent import ChatAgent, CrewAgent
from supporter.index import LLMChunk, LLMResult


@pytest.mark.asyncio
async def test_tool_dispatch_to_registry() -> None:
    mock_history = [
        types.Content(role="user", parts=[types.Part(text="What time is it?")]),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="get_current_time", args={})
                )
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_current_time", response={"time": "12:00 PM"}
                    )
                )
            ],
        ),
        types.Content(role="model", parts=[types.Part(text="It is 12:00 PM.")]),
    ]
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"
    mock_provider.generate = AsyncMock(
        return_value=LLMResult(
            text="It is 12:00 PM.",
            candidates=[MagicMock()],
            automatic_function_calling_history=mock_history,
        )
    )
    time_called = False

    def mock_get_time() -> dict[str, str]:
        nonlocal time_called
        time_called = True
        return {"time": "12:00 PM"}

    agent = ChatAgent(
        mock_provider,
        tools=cast(Any, [{"name": "get_current_time"}]),
        registry={"get_current_time": mock_get_time},
    )
    response = await agent.execute("What time is it?")
    assert response.text == "It is 12:00 PM."
    assert agent.history == mock_history


@pytest.mark.asyncio
async def test_chat_agent_streaming() -> None:
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"

    async def mock_stream(*args: Any, **kwargs: Any) -> Any:
        yield LLMChunk(text="Chunk 1", is_last=False)
        yield LLMChunk(text="Chunk 2", is_last=True)

    mock_provider.generate_stream.side_effect = mock_stream
    agent = ChatAgent(mock_provider)
    chunks = []
    async for chunk in agent.execute_stream("hello"):
        chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[0].text == "Chunk 1"
    assert len(agent.history) == 2


@pytest.mark.asyncio
async def test_chat_agent_execute_appends_user_and_model_history() -> None:
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"
    mock_provider.generate = AsyncMock(
        return_value=LLMResult(
            text="Hello back",
            interaction_id="interaction-123",
            candidates=[
                MagicMock(
                    content=Content(role="model", parts=[Part(text="Hello back")])
                )
            ],
        )
    )
    agent = ChatAgent(mock_provider)
    result = await agent.execute("Hello")
    assert result.text == "Hello back"
    assert agent.current_interaction_id == "interaction-123"
    assert len(agent.history) == 2
    assert agent.history[0].role == "user"
    assert agent.history[0].parts and agent.history[0].parts[0].text == "Hello"
    assert agent.history[1].role == "model"
    assert agent.history[1].parts and agent.history[1].parts[0].text == "Hello back"


@pytest.mark.asyncio
async def test_chat_agent_execute_appends_only_user_when_content_missing() -> None:
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"
    mock_provider.generate = AsyncMock(
        return_value=LLMResult(text="", candidates=[MagicMock(content=None)])
    )
    agent = ChatAgent(mock_provider)
    await agent.execute("Hello")
    assert len(agent.history) == 1
    assert agent.history[0].role == "user"
    assert agent.history[0].parts and agent.history[0].parts[0].text == "Hello"


def test_chat_agent_clear_history() -> None:
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"
    agent = ChatAgent(mock_provider)
    agent.history = [Content(role="user", parts=[Part(text="some history")])]
    agent.clear_history()
    assert agent.history == []


@pytest.mark.asyncio
async def test_crew_agent_execute() -> None:
    mock_provider = MagicMock()
    mock_provider.get_name.return_value = "mock"
    with patch("supporter.crew.CrewManager") as mock_manager_cls:
        mock_manager = MagicMock()
        mock_manager.coordinate_execution = AsyncMock(
            return_value=LLMResult(text="Crew Result")
        )
        mock_manager_cls.return_value = mock_manager
        agent = CrewAgent(mock_provider)
        result = await agent.execute("run crew")
        assert result.text == "Crew Result"
        assert result.model == "CrewAI (Multi-Agent)"


@pytest.mark.asyncio
async def test_crew_agent_streaming_unsupported() -> None:
    mock_provider = MagicMock()
    agent = CrewAgent(mock_provider)
    with pytest.raises(NotImplementedError):
        await agent.execute_stream("hi")


def test_crew_agent_clear_history_is_noop() -> None:
    mock_provider = MagicMock()
    agent = CrewAgent(mock_provider)
    agent.clear_history()
