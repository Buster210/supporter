from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.index import LazyFallbackProvider


@pytest.mark.asyncio
async def test_fallback_on_503() -> None:
    class StatusError(Exception):
        status: int

    primary = MagicMock()
    primary.get_name.return_value = "primary"
    error = StatusError("Service Unavailable")
    error.status = 503
    primary.generate = AsyncMock(side_effect=error)
    fallback_provider = MagicMock()
    fallback_provider.get_name.return_value = "fallback"
    fallback_result = MagicMock()
    fallback_result.text = "Success from fallback"
    fallback_provider.generate = AsyncMock(return_value=fallback_result)
    provider = LazyFallbackProvider(lambda: primary, lambda: fallback_provider)
    result = await provider.generate("test")
    assert result.text == "Success from fallback"
    primary.generate.assert_called_once()
    fallback_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_no_fallback_on_generic_error() -> None:
    primary = MagicMock()
    primary.get_name.return_value = "primary"
    primary.generate = AsyncMock(side_effect=Exception("Generic error"))
    fallback_provider = MagicMock()
    fallback_provider.generate = AsyncMock()
    provider = LazyFallbackProvider(lambda: primary, lambda: fallback_provider)
    with pytest.raises(Exception, match="Generic error"):
        await provider.generate("test")
    fallback_provider.generate.assert_not_called()
