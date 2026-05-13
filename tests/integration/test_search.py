from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tools.base import ToolError
from supporter.tools.search import google_search


@pytest.mark.asyncio
async def test_google_search_success_with_sources() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = "Search results text"
    mock_chunk = MagicMock()
    mock_chunk.web.uri = "https://example.com"
    mock_chunk.web.title = "Example Title"
    mock_meta = MagicMock()
    mock_meta.grounding_chunks = [mock_chunk]
    mock_candidate = MagicMock()
    mock_candidate.grounding_metadata = mock_meta
    mock_result.raw.candidates = [mock_candidate]
    mock_provider.generate.return_value = mock_result
    with patch("supporter.pool.get_provider", return_value=mock_provider):
        response = await google_search("test query")
        assert "Search results text" in response
        assert "SOURCES FOUND:" in response
        assert "Example Title: https://example.com" in response
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_google_search_success_no_sources() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = "Just text"
    mock_result.raw.candidates = []
    mock_provider.generate.return_value = mock_result
    with patch("supporter.pool.get_provider", return_value=mock_provider):
        response = await google_search("test query")
        assert response == "Just text"
        assert "SOURCES FOUND:" not in response


@pytest.mark.asyncio
async def test_google_search_returns_text_when_grounding_metadata_missing() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = "Just text"
    mock_candidate = MagicMock()
    mock_candidate.grounding_metadata = None
    mock_result.raw.candidates = [mock_candidate]
    mock_provider.generate.return_value = mock_result
    with patch("supporter.pool.get_provider", return_value=mock_provider):
        response = await google_search("test query")
    assert response == "Just text"
    assert "SOURCES FOUND:" not in response


@pytest.mark.asyncio
async def test_google_search_skips_invalid_chunks_and_returns_text() -> None:
    mock_provider = AsyncMock()
    mock_result = MagicMock()
    mock_result.text = "Just text"
    chunk_without_web = MagicMock()
    chunk_without_web.web = None
    chunk_with_missing_url = MagicMock()
    chunk_with_missing_url.web.uri = ""
    chunk_with_missing_url.web.title = "Missing URL"
    mock_meta = MagicMock()
    mock_meta.grounding_chunks = [chunk_without_web, chunk_with_missing_url]
    mock_candidate = MagicMock()
    mock_candidate.grounding_metadata = mock_meta
    mock_result.raw.candidates = [mock_candidate]
    mock_provider.generate.return_value = mock_result
    with patch("supporter.pool.get_provider", return_value=mock_provider):
        response = await google_search("test query")
    assert response == "Just text"
    assert "SOURCES FOUND:" not in response


@pytest.mark.asyncio
async def test_google_search_failure() -> None:
    mock_provider = AsyncMock()
    mock_provider.generate.side_effect = Exception("API Error")
    with (
        patch("supporter.pool.get_provider", return_value=mock_provider),
        pytest.raises(ToolError, match="Search failed for 'test query': API Error"),
    ):
        await google_search("test query")
