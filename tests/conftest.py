import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from tests.mocks import create_mock_genai_client


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


def pytest_configure(config: pytest.Config) -> None:
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
