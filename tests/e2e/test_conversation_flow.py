from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from supporter.agent import ChatAgent
from supporter.llm.types import GenOptions
from supporter.pool import (
    DynamicPool,
    LazyFallbackProvider,
    LLMProvider,
)
from supporter.types import LLMChunk, LLMResult

from .conftest import MockLLMProvider


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_single_message() -> None:
    mock_provider = MockLLMProvider("Hello from mock provider")
    agent = ChatAgent(provider=mock_provider)
    result = await agent.execute("Say hello")
    assert "Hello from mock provider" in result.text
    assert result.model == "mock-model"
    assert len(agent.history) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_multi_turn() -> None:
    mock_provider = MockLLMProvider("Response")
    agent = ChatAgent(provider=mock_provider)
    result1 = await agent.execute("First message")
    assert "call #1" in result1.text
    assert len(agent.history) == 2
    result2 = await agent.execute("Second message")
    assert "call #2" in result2.text
    assert len(agent.history) == 4


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_streaming() -> None:
    mock_provider = MockLLMProvider("Streaming test response")
    agent = ChatAgent(provider=mock_provider)
    accumulated = ""
    async for chunk in agent.execute_stream("Stream me something"):
        accumulated += chunk.text
    assert accumulated == "Streaming test response "
    assert len(agent.history) == 2


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_clear_history() -> None:
    mock_provider = MockLLMProvider("Test")
    agent = ChatAgent(provider=mock_provider)
    await agent.execute("Message 1")
    await agent.execute("Message 2")
    assert len(agent.history) > 0
    agent.clear_history()
    assert len(agent.history) == 0
    assert agent.current_interaction_id is None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_with_pool() -> None:
    mock_provider_instance = MockLLMProvider("Pool response")
    with patch("supporter.pool.GeminiProvider") as mock_gemini:
        mock_gemini.return_value = mock_provider_instance
        mock_gemini.get_name = lambda self: "MockedGemini"
        pool = DynamicPool(
            keys=["test-key-1", "test-key-2"],
            model_name="mock-model",
            pool_size=2,
        )
        agent = ChatAgent(provider=pool)
        result = await agent.execute("Test with pool")
        assert "Pool response" in result.text
        await pool.shutdown()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_fallback_trigger() -> None:
    call_count = {"primary": 0, "fallback": 0}

    class FailingProvider(LLMProvider):
        def get_name(self) -> str:
            return "FailingProvider"

        async def generate(
            self, prompt: str | list[Any], options: GenOptions | None = None
        ) -> LLMResult:
            call_count["primary"] += 1

            class StatusError(Exception):
                status: int

            error = StatusError("Service unavailable")
            error.status = 503
            raise error

        async def generate_stream(
            self, prompt: str | list[Any], options: GenOptions | None = None
        ) -> AsyncIterator[LLMChunk]:
            return
            yield

    class WorkingProvider(LLMProvider):
        def get_name(self) -> str:
            return "FallbackProvider"

        async def generate(
            self, prompt: str | list[Any], options: GenOptions | None = None
        ) -> LLMResult:
            call_count["fallback"] += 1
            return LLMResult(
                text="Fallback response", model="fallback", interaction_id="fallback-1"
            )

        async def generate_stream(
            self, prompt: str | list[Any], options: GenOptions | None = None
        ) -> AsyncIterator[LLMChunk]:
            yield LLMChunk(text="Fallback ", model="fallback", is_last=True)

    provider = LazyFallbackProvider(
        lambda: FailingProvider(), lambda: WorkingProvider()
    )
    agent = ChatAgent(provider=provider)
    result = await agent.execute("Trigger fallback")
    assert call_count["primary"] >= 1
    assert call_count["fallback"] >= 1
    assert "Fallback response" in result.text
