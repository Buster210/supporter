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
    assert await manager.handle_command("/crew") is True
    assert app.toggled_crew is True
    assert await manager.handle_command("/live") is True
    assert app.toggled_live is True
    assert await manager.handle_command("/unknown") is False


@pytest.mark.asyncio
async def test_toggle_mode_logic() -> None:
    app = MockApp()
    manager = ModeManager(app)
    assert app.live_mode is True
    assert app.crew_mode is False
    with patch.object(manager, "setup_agent") as mock_setup:
        await manager.toggle_mode(crew=True)
        assert app.crew_mode is True
        assert app.live_mode is False
        mock_setup.assert_called_with(use_crew=True, use_live=False)
    with patch.object(manager, "setup_agent") as mock_setup:
        await manager.toggle_mode(live=True)
        assert app.live_mode is True
        assert app.crew_mode is False
        mock_setup.assert_called_with(use_crew=False, use_live=True)


@pytest.mark.asyncio
async def test_setup_agent_dispatch() -> None:
    app = MockApp()
    manager = ModeManager(app)
    with (
        patch("supporter.get_provider") as mock_get_provider,
        patch("supporter.agent.ChatAgent") as mock_chat_agent,
        patch("supporter.agent.CrewAgent") as mock_crew_agent,
        patch("supporter.tools.check_bash_availability", return_value=True),
        patch("supporter.tools.execute_bash"),
        patch("supporter.tools.read_file"),
        patch("supporter.tools.write_file"),
        patch("supporter.tools.list_dir"),
    ):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        await manager.setup_agent(use_crew=False, use_live=True)
        mock_get_provider.assert_called_with(live=True, registry=ANY)
        mock_chat_agent.assert_called_once()
        assert app.agent is not None
        mock_get_provider.reset_mock()
        mock_chat_agent.reset_mock()
        await manager.setup_agent(use_crew=True, use_live=False)
        mock_get_provider.assert_called_with(live=False, registry=ANY)
        mock_crew_agent.assert_called_once()
        assert app.agent is not None
