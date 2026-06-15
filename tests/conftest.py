import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.mocks import create_mock_genai_client


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("GEMINI_API_KEY", "test-key")  # pragma: allowlist secret
    os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    for var in ("GEMINI_API_KEYS",):
        os.environ.pop(var, None)

    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components in isolation"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests that test multiple components together",
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests that test the full application flow"
    )


@pytest.fixture(autouse=True)
def setup_env() -> Generator[None, None, None]:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",  # pragma: allowlist secret
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "LOG_LEVEL": "DEBUG",
        },
        clear=True,
    ):
        yield


@pytest.fixture(autouse=True)
def clear_project_root_cache() -> Generator[None, None, None]:
    from supporter.tools import _resolve_path

    _resolve_path.cache_clear()
    yield
    _resolve_path.cache_clear()


@pytest.fixture(autouse=True)
def isolate_trust_store(tmp_path: Path) -> Generator[None, None, None]:
    """Give each test a fresh, tmp-backed browser TrustStore.

    The store is a module-level singleton that persists to ~/.supporter; without
    isolation, a test that promotes/confirms a host (e.g. finish() recording a
    clean interaction) leaks into later tests' host_is_fast() checks and writes
    to the developer's real trusted.json.
    """
    from supporter.tools.browser import guardrails

    fresh = guardrails.TrustStore.__new__(guardrails.TrustStore)
    fresh._store_path = tmp_path / "trusted.json"
    fresh._data = {}
    fresh._dirty = False
    saved = guardrails._trust_store
    guardrails._trust_store = fresh
    yield
    guardrails._trust_store = saved


@pytest.fixture
def mock_file_ops_config() -> Generator[MagicMock, None, None]:
    from supporter.config import config as real_config

    saved = {
        "allowed_directories": real_config.allowed_directories,
        "require_write_confirmation": real_config.require_write_confirmation,
        "log_file": real_config.log_file,
        "blacklist": getattr(real_config, "blacklist", []),
    }
    real_config.allowed_directories = []
    real_config.require_write_confirmation = False
    real_config.log_file = ""
    yield real_config  # type: ignore[misc]
    for k, v in saved.items():
        setattr(real_config, k, v)


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock, None, None]:
    with patch("google.genai.Client") as mock_client:
        instance = create_mock_genai_client()
        mock_client.return_value = instance
        mock_client.side_effect = lambda **kwargs: instance
        yield mock_client


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
