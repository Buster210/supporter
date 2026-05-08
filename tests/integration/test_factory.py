import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import supporter.index as index
from supporter.config import load_config
from supporter.index import LLMResult, get_provider


def test_default_to_gemini() -> None:
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key"},  # pragma: allowlist secret
        clear=True,
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider()
        name = provider.get_name().lower()
        assert "gemini" in name or "gemma" in name


def test_detect_provider_from_env() -> None:
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",  # pragma: allowlist secret
        },
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider()
        assert "gemini" in provider.get_name().lower()


def test_unsupported_provider_type() -> None:
    index.clear_providers()
    with pytest.raises(ValueError, match="Unsupported provider type"):
        get_provider("unsupported")


@pytest.mark.asyncio
async def test_multiple_api_keys_round_robin() -> None:

    def mock_provider_factory(*args: Any, **kwargs: Any) -> MagicMock:
        mock_instance = MagicMock()
        mock_instance.get_name.return_value = "MockedProvider"
        mock_instance.generate = AsyncMock(return_value=LLMResult(text="Mocked"))
        return mock_instance

    with (
        patch.dict(
            os.environ,
            {
                "GEMINI_API_KEYS": "key1, key2, key3",  # pragma: allowlist secret
                "GEMINI_FALLBACK_MODEL": "",
            },
            clear=True,
        ),
        patch("supporter.index.GeminiProvider", side_effect=mock_provider_factory),
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider()
        await provider.generate("test")
        await provider.generate("test")
        assert "Dynamic Pool x2" in provider.get_name()


@pytest.mark.integration
def test_invalid_api_key_handling() -> None:
    index.clear_providers()
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
        index.config = load_config()
        with pytest.raises(ValueError, match="missing"):
            get_provider()


@pytest.mark.integration
def test_missing_env_vars() -> None:
    index.clear_providers()
    with patch.dict(os.environ, {}, clear=True):
        index.config = load_config()
        with pytest.raises((ValueError, KeyError)):
            get_provider()


@pytest.mark.integration
def test_multiple_provider_selection() -> None:
    index.clear_providers()
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key"},  # pragma: allowlist secret
        clear=True,
    ):
        index.config = load_config()
        provider = get_provider("gemini")
        assert provider is not None
        name = provider.get_name().lower()
        assert "gem" in name


@pytest.mark.integration
def test_provider_priority_order() -> None:
    index.clear_providers()
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "test-key"},  # pragma: allowlist secret
        clear=True,
    ):
        index.config = load_config()
        provider = get_provider()
        assert "gem" in provider.get_name().lower()
