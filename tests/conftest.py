import os
from collections.abc import Generator
from typing import Any
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
def mock_genai_client() -> Generator[MagicMock, None, None]:
    with patch("supporter.providers.gemini_provider.genai.Client") as mock_client:

        def side_effect(**kwargs: dict[str, Any]) -> MagicMock:
            instance = create_mock_genai_client(**kwargs)
            mock_client.return_value = instance
            return instance

        mock_client.side_effect = side_effect
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
