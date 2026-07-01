"""Unit tests for router module."""

from unittest.mock import AsyncMock

import pytest

from supporter.router import RouteDecision, route_prompt


def _provider(text: str) -> AsyncMock:
    provider = AsyncMock()
    provider.generate.return_value = AsyncMock(text=text)
    return provider


@pytest.mark.asyncio
async def test_route_direct() -> None:
    provider = _provider('{"route": "direct", "needs_research": false}')
    decision = await route_prompt(provider, "hi there")
    assert decision == RouteDecision(route="direct", needs_research=False)


@pytest.mark.asyncio
async def test_route_research() -> None:
    provider = _provider('{"route": "research", "needs_research": false}')
    decision = await route_prompt(provider, "compare X vs Y")
    assert decision.route == "research"


@pytest.mark.asyncio
async def test_route_task_with_needs_research() -> None:
    provider = _provider('{"route": "task", "needs_research": true}')
    decision = await route_prompt(provider, "build a feature using latest docs")
    assert decision.route == "task"
    assert decision.needs_research is True


@pytest.mark.asyncio
async def test_strips_markdown_fence() -> None:
    provider = _provider('```json\n{"route": "direct", "needs_research": false}\n```')
    decision = await route_prompt(provider, "hi")
    assert decision.route == "direct"


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_task() -> None:
    provider = _provider("not json at all")
    decision = await route_prompt(provider, "anything")
    assert decision == RouteDecision(route="task", needs_research=False)


@pytest.mark.asyncio
async def test_empty_response_falls_back_to_task() -> None:
    provider = _provider("")
    decision = await route_prompt(provider, "anything")
    assert decision == RouteDecision(route="task", needs_research=False)


@pytest.mark.asyncio
async def test_invalid_route_value_falls_back_to_task() -> None:
    provider = _provider('{"route": "bogus", "needs_research": false}')
    decision = await route_prompt(provider, "anything")
    assert decision == RouteDecision(route="task", needs_research=False)


@pytest.mark.asyncio
async def test_non_object_json_falls_back_to_task() -> None:
    provider = _provider("[1, 2, 3]")
    decision = await route_prompt(provider, "anything")
    assert decision == RouteDecision(route="task", needs_research=False)


@pytest.mark.asyncio
async def test_provider_exception_falls_back_to_task() -> None:
    provider = AsyncMock()
    provider.generate.side_effect = RuntimeError("boom")
    decision = await route_prompt(provider, "anything")
    assert decision == RouteDecision(route="task", needs_research=False)
