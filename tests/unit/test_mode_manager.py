from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tui.mode_manager import ModeManager


@pytest.mark.asyncio
async def test_setup_agent_bash_unavailable() -> None:
    app = MagicMock()
    manager = ModeManager(app)
    with (
        patch("supporter.tools.check_bash_availability", return_value=False),
        patch("supporter.tools.notify_bash_unavailable") as mock_notify,
        patch("supporter.get_provider"),
        patch("supporter.agent.ChatAgent"),
    ):
        await manager.setup_agent(use_live=False)
        mock_notify.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_coroutine() -> None:
    app = MagicMock()

    async def async_exit() -> None:
        pass

    app.exit = async_exit
    manager = ModeManager(app)
    result = await manager.handle_command("/exit")
    assert result is True


@pytest.mark.asyncio
async def test_handle_command_non_coroutine() -> None:
    app = MagicMock()
    app.action_clear_screen = MagicMock(return_value=None)
    manager = ModeManager(app)
    result = await manager.handle_command("/clear")
    assert result is True
    app.action_clear_screen.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_unknown() -> None:
    app = MagicMock()
    manager = ModeManager(app)
    result = await manager.handle_command("/unknown")
    assert result is False


@pytest.mark.asyncio
async def test_toggle_mode_live_set() -> None:
    app = MagicMock()
    app.live_mode = False
    app._start_thinking = MagicMock()
    app._stop_thinking = MagicMock()
    app.is_activating_mode = False
    app.post_message = MagicMock()
    manager = ModeManager(app)
    with (
        patch.object(manager, "setup_agent", new_callable=AsyncMock),
        patch.object(manager, "_update_ui_state"),
    ):
        await manager.toggle_mode(live=True)
    assert app.live_mode is True


@pytest.mark.asyncio
async def test_toggle_mode_agent_set() -> None:
    app = MagicMock()
    app.live_mode = True
    app._start_thinking = MagicMock()
    app._stop_thinking = MagicMock()
    app.is_activating_mode = False
    app.post_message = MagicMock()
    manager = ModeManager(app)
    with (
        patch.object(manager, "setup_agent", new_callable=AsyncMock),
        patch.object(manager, "_update_ui_state"),
    ):
        await manager.toggle_mode(live=False)
    assert app.live_mode is False


@pytest.mark.asyncio
async def test_handle_command_agent() -> None:
    app = MagicMock()
    app._toggle_mode = AsyncMock()
    manager = ModeManager(app)
    result = await manager.handle_command("/agent")
    assert result is True
    app._toggle_mode.assert_called_once_with(live=False)
