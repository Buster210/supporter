from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.pool import DynamicPool, LLMChunk, LLMResult, clear_providers


@pytest.fixture(autouse=True)
async def pool_cleanup() -> AsyncGenerator[None, None]:
    yield
    await DynamicPool.shutdown_all()
    clear_providers()


def create_mock_provider(name: str) -> MagicMock:
    provider = MagicMock()
    provider.get_name.return_value = name
    provider.generate = AsyncMock(return_value=LLMResult(text=f"Response from {name}"))

    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text=f"Stream from {name}", is_last=True)

    provider.generate_stream = MagicMock(side_effect=mock_stream)
    return provider


@pytest.mark.asyncio
async def test_round_robin_cycling() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        res1 = await lb.generate("test")
        assert res1.text == "Response from P1"
        res2 = await lb.generate("test")
        assert res2.text == "Response from P2"
        res3 = await lb.generate("test")
        assert res3.text == "Response from P1"


@pytest.mark.asyncio
async def test_round_robin_streaming() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        stream1 = lb.generate_stream("test")
        chunk1 = await stream1.__anext__()
        assert chunk1.text == "Stream from P1"
        stream2 = lb.generate_stream("test")
        chunk2 = await stream2.__anext__()
        assert chunk2.text == "Stream from P2"


@pytest.mark.asyncio
async def test_load_balancer_name() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        await lb.generate("test")
        await lb.generate("test")
        assert lb.get_name() == "P1 (Dynamic Pool x2)"


def test_error_categorization() -> None:
    from supporter.pool import is_model_error, is_rate_limit, should_trigger_fallback

    mock_rate_limit = MagicMock()
    mock_rate_limit.status = 429
    assert is_rate_limit(mock_rate_limit) is True
    assert is_rate_limit(Exception("Quota exceeded")) is True
    mock_503 = MagicMock()
    mock_503.status = 503
    assert is_model_error(mock_503) is True
    assert is_model_error(Exception("Service Unavailable")) is True
    assert should_trigger_fallback(mock_503) is True


def test_model_cooldown() -> None:
    from supporter.pool import (
        _is_model_in_cooldown,
        _mark_model_cooldown,
        _model_cooldowns,
    )

    model = "test-model"
    _model_cooldowns.clear()
    assert _is_model_in_cooldown(model) is False
    _mark_model_cooldown(model, minutes=1)
    assert _is_model_in_cooldown(model) is True
    with patch("supporter.pool.datetime") as mock_dt:
        from datetime import datetime, timedelta

        mock_dt.now.return_value = datetime.now() + timedelta(minutes=2)
        assert _is_model_in_cooldown(model) is False


@pytest.mark.asyncio
async def test_lazy_fallback_provider() -> None:
    from supporter.pool import LazyFallbackProvider

    primary = create_mock_provider("Primary")
    fallback = create_mock_provider("Fallback")
    lfp = LazyFallbackProvider(
        primary_factory=lambda: primary, fallback_factory=lambda: fallback
    )
    assert lfp._primary is None
    res = await lfp.generate("hi")
    assert res.text == "Response from Primary"
    assert lfp._primary is not None
    primary.generate.side_effect = Exception("quota")
    res = await lfp.generate("hi")
    assert res.text == "Response from Fallback"
    assert lfp._fallback is not None


def test_get_provider_registry() -> None:
    from supporter.pool import clear_providers, get_provider

    clear_providers()
    p1 = get_provider("gemini")
    p2 = get_provider("gemini")
    assert p1 is p2
    clear_providers()
    p3 = get_provider("gemini")
    assert p3 is not p1
