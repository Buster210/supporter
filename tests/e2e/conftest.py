import os
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from google.genai.types import Content, Part

from supporter.config import config as real_config
from supporter.pool import LLMProvider, clear_providers
from supporter.tools.browser import guardrails, session
from supporter.types import LLMChunk, LLMOptions, LLMResult

TEST_MODEL = "gemini-3.1-flash-lite-preview"
TEST_API_KEY = "test-key-for-e2e"  # pragma: allowlist secret


@dataclass
class MockCandidate:
    content: Any


class MockLLMProvider(LLMProvider):
    def __init__(self, text_response: str = "Mocked response") -> None:
        self._text = text_response
        self._call_count = 0

    def get_name(self) -> str:
        return "MockProvider"

    async def generate(
        self, prompt: str | list[Any], options: LLMOptions | None = None
    ) -> LLMResult:
        self._call_count += 1
        text = f"{self._text} (call #{self._call_count})"
        content = Content(role="model", parts=[Part(text=text)])
        return LLMResult(
            text=text,
            model="mock-model",
            interaction_id=f"mock-{self._call_count}",
            candidates=[MockCandidate(content)],
        )

    async def generate_stream(
        self, prompt: str | list[Any], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        self._call_count += 1
        words = self._text.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield LLMChunk(text=word + " ", model="mock-model", is_last=is_last)


@pytest.fixture(autouse=True)
def setup_test_env() -> Generator[None, None, None]:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": TEST_API_KEY,
            "GEMINI_MODEL": TEST_MODEL,
            "LOG_LEVEL": "DEBUG",
        },
        clear=True,
    ):
        yield
    clear_providers()


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    return MockLLMProvider()


# ---------------------------------------------------------------------------
# throwaway browser — one real-Chromium session per test
# ---------------------------------------------------------------------------


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


@pytest.fixture
async def throwaway_browser(tmp_path: Path) -> AsyncIterator[None]:
    saved_path = real_config.browser_profile_path
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    guardrails.register_browse_callback(confirmation=_always_allow)
    try:
        yield
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
