from unittest.mock import ANY, MagicMock, patch

import pytest

from supporter.tui.mode_manager import ModeManager
from tests.tui_mocks import MockApp


@pytest.mark.asyncio
async def test_handle_command_routing() -> None:
    app = MockApp()
    manager = ModeManager(app)
    assert await manager.handle_command("/exit") is True
    assert app.exited is True
    assert await manager.handle_command("/clear") is True
    assert app.cleared is True
    assert await manager.handle_command("/live") is True
    assert app.toggled_live is True
    assert await manager.handle_command("/unknown") is False


@pytest.mark.asyncio
async def test_toggle_mode_logic() -> None:
    app = MockApp()
    manager = ModeManager(app)
    assert app.live_mode is True
    with patch.object(manager, "setup_agent") as mock_setup:
        # Test explicit set to False
        await manager.toggle_mode(live=False)
        assert app.live_mode is False
        mock_setup.assert_called_with(use_live=False)

        # Test explicit set to True
        await manager.toggle_mode(live=True)
        assert app.live_mode is True
        mock_setup.assert_called_with(use_live=True)

        # Test toggle (no args)
        await manager.toggle_mode()
        assert app.live_mode is False
        mock_setup.assert_called_with(use_live=False)


@pytest.mark.asyncio
async def test_setup_agent_dispatch() -> None:
    app = MockApp()
    manager = ModeManager(app)
    with (
        patch("supporter.get_provider") as mock_get_provider,
        patch("supporter.agent.ChatAgent") as mock_chat_agent,
        patch("supporter.tools.check_bash_availability", return_value=True),
        patch("supporter.tools.execute_bash"),
        patch("supporter.tools.read_file"),
        patch("supporter.tools.write_file"),
        patch("supporter.tools.list_dir"),
    ):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        await manager.setup_agent(use_live=True)
        mock_get_provider.assert_called_with(live=True, registry=ANY)
        mock_chat_agent.assert_called_once()
        assert app.agent is not None
