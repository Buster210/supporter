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


@pytest.fixture
def mock_file_ops_config() -> Generator[MagicMock, None, None]:
    with patch("supporter.tools.file_ops.config") as mock_config:
        mock_config.allowed_directories = []
        mock_config.require_write_confirmation = False
        mock_config.log_file = ""
        mock_config.blacklist = []
        mock_config.gitignore_spec = None
        yield mock_config


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
