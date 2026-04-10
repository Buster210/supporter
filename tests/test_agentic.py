from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.agent import ChatAgent
from supporter.index import LLMResult


@pytest.mark.asyncio
async def test_tool_dispatch_to_registry():
    # Setup a provider with a mocked automatic call loop
    from google.genai import types

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

    # Setup agent with registry
    time_called = False

    def mock_get_time():
        nonlocal time_called
        time_called = True
        return {"time": "12:00 PM"}

    agent = ChatAgent(
        mock_provider,
        tools=[{"name": "get_current_time"}],
        registry={"get_current_time": mock_get_time},
    )

    response = await agent.execute("What time is it?")

    assert response.text == "It is 12:00 PM."
    assert agent.get_history() == mock_history
