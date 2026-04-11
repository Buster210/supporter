import os
import pytest
from unittest.mock import patch

from supporter.index import get_provider


@pytest.mark.asyncio
async def test_provider_streaming(mock_genai_client):
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "GEMINI_FALLBACK_MODEL": "",
            "GOOGLE_API_KEY": "",
        },
        clear=True,
    ):
        provider = get_provider("gemini")
        chunks = []
        async for chunk in provider.generate_stream("Say 'Test Success'"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "Chunk 1"
        assert chunks[1].text == "Chunk 2"
        assert chunks[1].is_last is False
        assert "".join(c.text for c in chunks) == "Chunk 1Chunk 2"


@pytest.mark.asyncio
async def test_provider_name():
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "GEMINI_FALLBACK_MODEL": "",
            "GOOGLE_API_KEY": "",
        },
        clear=True,
    ):
        provider = get_provider("gemini")
        assert "gemini" in provider.get_name().lower()
