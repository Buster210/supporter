from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.binding import Binding
from textual.widgets import Input

from supporter.tui import SupporterApp

from .conftest import MockLLMProvider


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_can_be_instantiated(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()
    assert app is not None
    assert app._is_processing is False
    assert app.active_queries == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_compose_produces_widgets(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()

    async def side_effect(*args: Any, **kwargs: Any) -> None:
        app.agent = MagicMock()
        app.agent.provider = MagicMock()
        app.agent.provider.close = AsyncMock()

    with (
        patch(
            "supporter.tui.SupporterApp._setup_agent",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ),
        patch(
            "supporter.tui.mode_manager.ModeManager.trigger_live_greeting",
            new_callable=AsyncMock,
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#user-input") is not None
            assert app.query_one("#chat-view") is not None
            assert app.query_one("#supporter-header") is not None


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_app_handles_input_submission(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()

    async def side_effect(*args: Any, **kwargs: Any) -> None:
        app.agent = MagicMock()
        app.agent.provider = MagicMock()
        app.agent.provider.close = AsyncMock()

    with (
        patch(
            "supporter.tui.SupporterApp._setup_agent",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ),
        patch(
            "supporter.tui.mode_manager.ModeManager.trigger_live_greeting",
            new_callable=AsyncMock,
        ),
    ):
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
async def test_tui_clear_screen_action(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()

    async def side_effect(*args: Any, **kwargs: Any) -> None:
        app.agent = MagicMock()
        app.agent.provider = MagicMock()
        app.agent.provider.close = AsyncMock()

    with (
        patch(
            "supporter.tui.SupporterApp._setup_agent",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ),
        patch(
            "supporter.tui.mode_manager.ModeManager.trigger_live_greeting",
            new_callable=AsyncMock,
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_clear_screen()
            assert len(list(app.query_one("#chat-view").children)) == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_mode_indicator_exists(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()

    async def side_effect(*args: Any, **kwargs: Any) -> None:
        app.agent = MagicMock()
        app.agent.provider = MagicMock()
        app.agent.provider.close = AsyncMock()

    with (
        patch(
            "supporter.tui.SupporterApp._setup_agent",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ),
        patch(
            "supporter.tui.mode_manager.ModeManager.trigger_live_greeting",
            new_callable=AsyncMock,
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            mode_indicator = app.query_one("#mode-indicator")
            assert str(mode_indicator.render()) == "[LIVE]"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tui_key_bindings_registered(mock_provider: MockLLMProvider) -> None:
    app = SupporterApp()

    async def side_effect(*args: Any, **kwargs: Any) -> None:
        app.agent = MagicMock()
        app.agent.provider = MagicMock()
        app.agent.provider.close = AsyncMock()

    with (
        patch(
            "supporter.tui.SupporterApp._setup_agent",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ),
        patch(
            "supporter.tui.mode_manager.ModeManager.trigger_live_greeting",
            new_callable=AsyncMock,
        ),
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            binding_keys = [b.key for b in app.BINDINGS if isinstance(b, Binding)]
            assert "ctrl+c" in binding_keys
            assert "ctrl+l" in binding_keys
