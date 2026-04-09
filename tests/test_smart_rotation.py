from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.index import LLMResult, RoundRobinKeyProvider


@pytest.mark.asyncio
async def test_retry_on_429():
    call_count = 0

    async def mock1_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        error = Exception("Quota exceeded")
        error.status = 429
        raise error

    p1 = MagicMock()
    p1.get_name.return_value = "key1"
    p1.generate = AsyncMock(side_effect=mock1_generate)

    async def mock2_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return LLMResult(text="Success from Key 2")

    p2 = MagicMock()
    p2.get_name.return_value = "key2"
    p2.generate = AsyncMock(side_effect=mock2_generate)

    lb = RoundRobinKeyProvider([p1, p2])
    result = await lb.generate("test")

    assert result.text == "Success from Key 2"
    assert call_count == 2


@pytest.mark.asyncio
async def test_fast_fail_on_503_no_retry():
    call_count = 0

    async def mock1_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        error = Exception("Service Unavailable")
        error.status = 503
        raise error

    p1 = MagicMock()
    p1.get_name.return_value = "key1"
    p1.generate = AsyncMock(side_effect=mock1_generate)

    p2 = MagicMock()
    p2.get_name.return_value = "key2"
    p2.generate = AsyncMock()  # Should not be called

    lb = RoundRobinKeyProvider([p1, p2])

    with pytest.raises(Exception) as excinfo:
        await lb.generate("test")

    assert excinfo.value.status == 503
    assert call_count == 1
    p2.generate.assert_not_called()
