import os
from unittest.mock import patch

import pytest

from supporter import index
from supporter.config import load_config
from supporter.index import get_provider


def test_default_to_gemini() -> None:
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        index.config = load_config()
        provider = get_provider()
        name = provider.get_name().lower()
        assert "gemini" in name or "gemma" in name


def test_detect_provider_from_env() -> None:
    with patch.dict(
        os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}
    ):
        index.config = load_config()
        provider = get_provider()
        assert "gemini" in provider.get_name().lower()


def test_unsupported_provider_type() -> None:
    with pytest.raises(ValueError, match="Unsupported provider type"):
        get_provider("unsupported")


def test_multiple_api_keys_round_robin() -> None:
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEYS": "key1, key2, key3", "GEMINI_FALLBACK_MODEL": ""},
        clear=True,
    ):
        index.config = load_config()
        provider = get_provider()
        assert "Dynamic Pool x2" in provider.get_name()
