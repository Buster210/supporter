from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.index import FallbackProvider


@pytest.mark.asyncio
async def test_fallback_on_503():
    primary = MagicMock()
    primary.get_name.return_value = "primary"

    # Mock a 503 error
    error = Exception("Service Unavailable")
    error.status = 503
    primary.generate = AsyncMock(side_effect=error)

    fallback = MagicMock()
    fallback.get_name.return_value = "fallback"
    fallback_result = MagicMock()
    fallback_result.text = "Success from fallback"
    fallback.generate = AsyncMock(return_value=fallback_result)

    provider = FallbackProvider(primary, fallback)
    result = await provider.generate("test")

    assert result.text == "Success from fallback"
    primary.generate.assert_called_once()
    fallback.generate.assert_called_once()


@pytest.mark.asyncio
async def test_no_fallback_on_generic_error():
    primary = MagicMock()
    primary.get_name.return_value = "primary"
    primary.generate = AsyncMock(side_effect=Exception("Generic error"))

    fallback = MagicMock()
    fallback.generate = AsyncMock()

    provider = FallbackProvider(primary, fallback)

    with pytest.raises(Exception, match="Generic error"):
        await provider.generate("test")

    fallback.generate.assert_not_called()
