"""RUNG 5 — executable proof of the provider seam.

A new provider = one class implementing ``LLMProvider`` with neutral types
(``LLMResult`` / ``LLMChunk``) + one ``PROVIDER_FACTORIES`` line. This fake has
ZERO ``google.genai`` dependency, yet ``get_provider`` dispatches to it and the
agent consumes its neutral result unchanged — that is the seam working.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from supporter.agent import ChatAgent
from supporter.pool import get_provider
from supporter.providers.registry import PROVIDER_FACTORIES
from supporter.types import LLMChunk, LLMProvider, LLMResult


class FakeProvider:
    """Neutral-types-only provider — no vendor SDK anywhere."""

    def __init__(self, *, model_name: str | None = None) -> None:
        self._model = model_name or "fake-model"

    async def generate(self, prompt: Any, options: Any = None) -> LLMResult:
        text = prompt if isinstance(prompt, str) else "fake"
        return LLMResult(text=f"fake:{text}", model=self._model)

    async def generate_stream(
        self, prompt: Any, options: Any = None
    ) -> AsyncIterator[LLMChunk]:
        text = prompt if isinstance(prompt, str) else "fake"
        yield LLMChunk(text=f"fake:{text}", is_last=True, model=self._model)

    def get_name(self) -> str:
        return self._model


def _fake_factory(
    *,
    keys: list[str],
    model_name: str | None = None,
    pool_size: int = 2,
    registry: dict[str, Any] | None = None,
    system_instruction: str | None = None,
    live: bool = False,
) -> LLMProvider:
    return FakeProvider(model_name=model_name)


@pytest.fixture
def fake_registered() -> Any:
    PROVIDER_FACTORIES["fake"] = _fake_factory
    try:
        yield
    finally:
        PROVIDER_FACTORIES.pop("fake", None)


@pytest.mark.asyncio
async def test_get_provider_dispatches_to_fake(fake_registered: None) -> None:
    provider = get_provider("fake", shared=False)
    assert isinstance(provider, FakeProvider)

    result = await provider.generate("hello")
    assert isinstance(result, LLMResult)
    assert result.text == "fake:hello"
    assert result.model == "fake-model"


@pytest.mark.asyncio
async def test_unknown_provider_lists_registered(fake_registered: None) -> None:
    # True dispatch: an unknown type raises and names what IS registered.
    with pytest.raises(ValueError, match="fake"):
        get_provider("nope-not-real", shared=False)


@pytest.mark.asyncio
async def test_agent_consumes_fake_result(fake_registered: None) -> None:
    provider = get_provider("fake", shared=False)
    with patch("supporter.agent.config") as cfg:
        cfg.durable_history_enabled = False
        cfg.history_compaction_enabled = False
        cfg.history_max_turns = 200
        agent = ChatAgent(provider=provider)
        result = await agent.execute("ping")
    assert result.text == "fake:ping"


def test_fake_provider_has_no_genai_dependency(fake_registered: None) -> None:
    # The whole point: the provider + its factory reference no vendor SDK at all.
    source = inspect.getsource(FakeProvider) + inspect.getsource(_fake_factory)
    assert "genai" not in source
