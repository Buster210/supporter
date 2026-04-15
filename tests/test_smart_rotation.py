from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.index import DynamicPool, LLMResult


@pytest.mark.asyncio
async def test_retry_on_429() -> None:
    call_count = 0

    async def mock1_generate(*args: Any, **kwargs: Any) -> LLMResult:
        nonlocal call_count
        call_count += 1
        error = Exception("Quota exceeded")
        error.status = 429  # type: ignore[attr-defined]
        raise error

    p1 = MagicMock()
    p1.get_name.return_value = "key1"
    p1.generate = AsyncMock(side_effect=mock1_generate)

    async def mock2_generate(*args: Any, **kwargs: Any) -> LLMResult:
        nonlocal call_count
        call_count += 1
        return LLMResult(text="Success from Key 2")

    p2 = MagicMock()
    p2.get_name.return_value = "key2"
    p2.generate = AsyncMock(side_effect=mock2_generate)

    p3 = MagicMock()
    p3.get_name.return_value = "key3"

    with MagicMock() as _:
        import supporter.index

        setattr(supporter.index, "GeminiProvider", MagicMock(side_effect=[p1, p2, p3]))  # noqa: B010
        lb = DynamicPool(["key1", "key2", "key3"], model_name="test-model")
        result = await lb.generate("test")

    assert result.text == "Success from Key 2"


@pytest.mark.asyncio
async def test_fast_fail_on_503_no_retry() -> None:
    call_count = 0

    async def mock1_generate(*args: Any, **kwargs: Any) -> LLMResult:
        nonlocal call_count
        call_count += 1
        error = Exception("Service Unavailable")
        error.status = 503  # type: ignore[attr-defined]
        raise error

    p1 = MagicMock()
    p1.get_name.return_value = "key1"
    p1.generate = AsyncMock(side_effect=mock1_generate)

    p2 = MagicMock()
    p2.get_name.return_value = "key2"
    p2.generate = AsyncMock(return_value=LLMResult(text="Success from Key 2"))

    p3 = MagicMock()
    p3.get_name.return_value = "key3"

    with MagicMock() as _:
        import supporter.index

        setattr(supporter.index, "GeminiProvider", MagicMock(side_effect=[p1, p2, p3]))  # noqa: B010
        lb = DynamicPool(["key1", "key2", "key3"], model_name="test-model")

        result = await lb.generate("test")
        assert (
            result.model and "key2" in result.model
        ) or result.text == "Response from P2"
        assert call_count == 1
