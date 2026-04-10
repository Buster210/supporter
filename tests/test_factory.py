import os
from unittest.mock import patch

import pytest

from supporter.index import RoundRobinPool, get_provider


def test_default_to_gemini():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
        provider = get_provider()
        assert "gemini" in provider.get_name().lower()


def test_detect_provider_from_env():
    with patch.dict(
        os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}
    ):
        provider = get_provider()
        assert "gemini" in provider.get_name().lower()


def test_unsupported_provider_type():
    with pytest.raises(ValueError, match="Unsupported provider type"):
        get_provider("unsupported")


def test_multiple_api_keys_round_robin():
    with patch.dict(
        os.environ,
        {"GEMINI_API_KEYS": "key1, key2, key3", "GEMINI_FALLBACK_MODEL": ""},
        clear=True,
    ):
        provider = get_provider()
        assert "Pool x3" in provider.get_name()
