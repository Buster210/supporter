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
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "LOG_LEVEL": "DEBUG",
        },
        clear=True,
    ):
        yield


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock, None, None]:

    with patch("supporter.gemini_provider.genai.Client") as mock_client:

        def side_effect(**kwargs: dict[str, Any]) -> MagicMock:
            instance = create_mock_genai_client(**kwargs)
            mock_client.return_value = instance
            return instance

        mock_client.side_effect = side_effect
        yield mock_client
