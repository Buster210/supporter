from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tools.base import ToolError
from supporter.tools.search import google_search


@pytest.mark.asyncio
async def test_google_search_returns_provider_text_verbatim() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = (
        "Detailed answer with inline citations [1].\n\n"
        "SOURCES FOUND:\n- Example Title: https://example.com"
    )
    mock_provider.generate.return_value = mock_result

    with patch("supporter.pool.get_provider", return_value=mock_provider):
        response = await google_search("test query")

    assert response == mock_result.text
    mock_provider.generate.assert_called_once_with(prompt="test query")


@pytest.mark.asyncio
async def test_google_search_passes_system_instruction_to_pool() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = "answer"
    mock_provider.generate.return_value = mock_result

    with patch(
        "supporter.pool.get_provider", return_value=mock_provider
    ) as mock_get_provider:
        await google_search("test query")

    from supporter.tools.search import _SEARCH_SYSTEM_INSTRUCTION

    kwargs = mock_get_provider.call_args.kwargs
    assert kwargs["live"] is True
    assert kwargs["model_name"]
    assert kwargs["system_instruction"] == _SEARCH_SYSTEM_INSTRUCTION


@pytest.mark.asyncio
async def test_google_search_failure() -> None:
    mock_provider = AsyncMock()
    mock_provider.generate.side_effect = Exception("API Error")
    with (
        patch("supporter.pool.get_provider", return_value=mock_provider),
        pytest.raises(ToolError, match="Search failed for 'test query': API Error"),
    ):
        await google_search("test query")
