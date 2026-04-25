import os
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from supporter.index import LLMProvider, clear_providers
from supporter.llm_types import LLMChunk, LLMOptions, LLMResult
from supporter.tui import SupporterApp


class MockProvider(LLMProvider):  # type: ignore[misc]  # type: ignore[misc]
    def get_name(self) -> str:
        return "MockProvider"

    async def generate(
        self, prompt: str | list[Any], options: LLMOptions | None = None
    ) -> LLMResult:
        return LLMResult(text="Mock response", model="mock", interaction_id="mock-1")

    async def generate_stream(
        self, prompt: str | list[Any], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text="Mock ", model="mock", is_last=False)


@pytest.fixture(autouse=True)
def setup_test_env() -> Any:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key-e2e-tui",  # pragma: allowlist secret
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "LOG_LEVEL": "DEBUG",
        },
        clear=True,
    ):
        clear_providers()
        yield
    clear_providers()


@pytest.fixture
def mock_provider() -> Any:
    return MockProvider()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_can_be_instantiated(mock_provider: Any) -> None:
    app = SupporterApp()
    assert app is not None
    assert app._is_processing is False
    assert app.active_queries == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_compose_produces_widgets(mock_provider: Any) -> None:
    app = SupporterApp()
    with patch("supporter.tui.SupporterApp._setup_agent", new_callable=MagicMock):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#user-input") is not None
            assert app.query_one("#chat-view") is not None
            assert app.query_one("#supporter-header") is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_handles_input_submission(mock_provider: Any) -> None:
    from textual.widgets import Input

    app = SupporterApp()
    with patch("supporter.tui.SupporterApp._setup_agent", new_callable=MagicMock):
        async with app.run_test() as pilot:
            await pilot.pause()
            input_widget = cast(Input, app.query_one("#user-input"))
            input_widget.value = "test message"
            event = Input.Submitted(input_widget, "test message")
            await app.on_input_submitted(event)
            await pilot.pause()
            await pilot.pause()
            assert input_widget.value == ""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_clear_screen_action(mock_provider: Any) -> None:
    app = SupporterApp()
    with patch("supporter.tui.SupporterApp._setup_agent", new_callable=MagicMock):
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_clear_screen()
            assert len(list(app.query_one("#chat-view").children)) == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_mode_indicator_exists(mock_provider: Any) -> None:
    app = SupporterApp()
    with patch("supporter.tui.SupporterApp._setup_agent", new_callable=MagicMock):
        async with app.run_test() as pilot:
            await pilot.pause()
            mode_indicator = app.query_one("#mode-indicator")
            assert cast(Any, mode_indicator.render()).plain == "[LIVE]"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_key_bindings_registered(mock_provider: Any) -> None:
    app = SupporterApp()
    with patch("supporter.tui.SupporterApp._setup_agent", new_callable=MagicMock):
        async with app.run_test() as pilot:
            await pilot.pause()
            from textual.binding import Binding

            binding_keys = [b.key for b in app.BINDINGS if isinstance(b, Binding)]
            assert "ctrl+c" in binding_keys
            assert "ctrl+l" in binding_keys
