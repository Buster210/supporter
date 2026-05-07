import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from supporter.tui.mode_manager import ModeManager
from tests.tui_mocks import MockApp


def _make_manager(app: Any) -> ModeManager:
    with patch(
        "supporter.providers.gemini_live_provider.GeminiLiveProvider"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        mock_cls.return_value.warmup = AsyncMock()
        mock_cls.return_value.close = AsyncMock()
        mock_cls.return_value.generate_stream = AsyncMock()
        return ModeManager(app)


@pytest.mark.asyncio
async def test_setup_agent_bash_unavailable() -> None:
    app = MagicMock()
    manager = _make_manager(app)
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
    manager = _make_manager(app)
    result = await manager.handle_command("/exit")
    assert result is True


@pytest.mark.asyncio
async def test_handle_command_non_coroutine() -> None:
    app = MagicMock()
    app.action_clear_screen = MagicMock(return_value=None)
    manager = _make_manager(app)
    result = await manager.handle_command("/clear")
    assert result is True
    app.action_clear_screen.assert_called_once()


@pytest.mark.asyncio
async def test_handle_command_unknown() -> None:
    app = MagicMock()
    manager = _make_manager(app)
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
    manager = _make_manager(app)
    with (
        patch.object(manager, "setup_agent", new_callable=AsyncMock),
        patch.object(manager, "_update_ui_state"),
        patch.object(manager, "trigger_live_greeting", new_callable=AsyncMock),
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
    manager = _make_manager(app)
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
    manager = _make_manager(app)
    result = await manager.handle_command("/agent")
    assert result is True
    app._toggle_mode.assert_called_once_with(live=False)


@pytest.mark.asyncio
async def test_handle_command_routing() -> None:
    app = MockApp()
    manager = _make_manager(app)
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
    manager = _make_manager(app)
    assert app.live_mode is True
    with (
        patch.object(manager, "setup_agent", new_callable=AsyncMock) as mock_setup,
        patch.object(manager, "trigger_live_greeting", new_callable=AsyncMock),
    ):
        await manager.toggle_mode(live=False)
        assert app.live_mode is False
        mock_setup.assert_called_with(use_live=False)

        await manager.toggle_mode(live=True)
        assert app.live_mode is True
        mock_setup.assert_called_with(use_live=True)

        await manager.toggle_mode()
        assert app.live_mode is False
        mock_setup.assert_called_with(use_live=False)


@pytest.mark.asyncio
async def test_setup_agent_dispatch() -> None:
    app = MockApp()
    manager = _make_manager(app)
    with (
        patch("supporter.get_provider") as mock_get_provider,
        patch("supporter.agent.ChatAgent") as mock_chat_agent,
        patch("supporter.tools.check_bash_availability", return_value=True),
        patch("supporter.tools.execute_bash"),
        patch("supporter.tools.read_file"),
        patch("supporter.tools.write_file"),
        patch("supporter.tools.google_search"),
    ):
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        await manager.setup_agent(use_live=True)
        mock_get_provider.assert_called_with(live=True, registry=ANY)
        mock_chat_agent.assert_called_once()
        assert app.agent is not None


@pytest.mark.asyncio
async def test_trigger_live_greeting_shows_loading_then_replaces() -> None:
    class Banner:
        def __init__(self) -> None:
            self.history: list[str] = []
            self._message = ""

        @property
        def message(self) -> str:
            return self._message

        @message.setter
        def message(self, value: str) -> None:
            self._message = value
            self.history.append(value)

    banner = Banner()
    app = MagicMock()
    app.query_one = MagicMock(return_value=banner)
    manager = _make_manager(app)

    class FakeProvider:
        async def warmup(self) -> None:
            await asyncio.sleep(0.45)

        async def generate_stream(self, prompt: str) -> AsyncIterator[Any]:
            assert prompt
            assert banner.message.startswith("wait, Loading")
            for text in ("Hello ", "there!"):
                yield type("Chunk", (), {"text": text})()

    manager._greeting_provider = cast(Any, FakeProvider())
    await manager.trigger_live_greeting()

    loading_frames = [
        item for item in banner.history if item.startswith("wait, Loading")
    ]
    assert len(set(loading_frames)) >= 2
    assert "Hello" in banner.history
    assert banner.history[-1] == "Hello there!"
    assert banner.message == "Hello there!"


@pytest.mark.asyncio
async def test_toggle_mode_already_in_same_mode() -> None:
    app = MagicMock()
    app.live_mode = True
    active_turn = AsyncMock()
    app.active_turn = active_turn
    manager = _make_manager(app)
    await manager.toggle_mode(live=True)
    active_turn.mount.assert_called_once()


@pytest.mark.asyncio
async def test_toggle_mode_already_in_agent_mode() -> None:
    app = MagicMock()
    app.live_mode = False
    app.active_turn = None
    chat_view = AsyncMock()
    app.query_one = MagicMock(return_value=chat_view)
    manager = _make_manager(app)
    await manager.toggle_mode(live=False)
    chat_view.mount.assert_called_once()


@pytest.mark.asyncio
async def test_toggle_mode_live_greeting_failure() -> None:
    app = MagicMock()
    app.live_mode = False
    app._start_thinking = MagicMock()
    app._stop_thinking = MagicMock()
    app.is_activating_mode = False
    app.post_message = MagicMock()
    manager = _make_manager(app)

    with (
        patch.object(manager, "setup_agent", new_callable=AsyncMock),
        patch.object(manager, "_update_ui_state"),
        patch.object(
            manager,
            "trigger_live_greeting",
            new_callable=AsyncMock,
            side_effect=Exception("greeting boom"),
        ),
    ):
        await manager.toggle_mode(live=True)
    assert app.live_mode is True
    assert app.is_activating_mode is False
    app._stop_thinking.assert_called_once()


@pytest.mark.asyncio
async def test_trigger_live_greeting_empty_text_fallback() -> None:

    class Banner:
        def __init__(self) -> None:
            self.message = ""

    app = MagicMock()
    banner = Banner()
    app.query_one = MagicMock(return_value=banner)
    manager = _make_manager(app)

    async def empty_stream(prompt: str) -> AsyncIterator[Any]:
        yield type("Chunk", (), {"text": ""})()

    manager._greeting_provider = MagicMock()
    manager._greeting_provider.warmup = AsyncMock()
    manager._greeting_provider.generate_stream = empty_stream
    manager._warmup_task = None

    await manager.trigger_live_greeting()
    assert banner.message.startswith("Hello ")


@pytest.mark.asyncio
async def test_trigger_live_greeting_exception_fallback() -> None:
    app = MagicMock()
    banner = MagicMock()
    app.query_one = MagicMock(return_value=banner)
    manager = _make_manager(app)

    async def failing_stream(prompt: str) -> AsyncIterator[Any]:
        raise Exception("provider down")
        yield  # make it a generator  # type: ignore[misc]

    manager._greeting_provider = MagicMock()
    manager._greeting_provider.warmup = AsyncMock()
    manager._greeting_provider.generate_stream = failing_stream
    manager._warmup_task = None

    await manager.trigger_live_greeting()


@pytest.mark.asyncio
async def test_trigger_live_greeting_cancels_loading_in_finally() -> None:
    app = MagicMock()
    banner = MagicMock()
    app.query_one = MagicMock(return_value=banner)
    manager = _make_manager(app)

    async def immediate_fail_stream(prompt: str) -> AsyncIterator[Any]:
        raise Exception("immediate fail")
        yield

    manager._greeting_provider = MagicMock()
    manager._greeting_provider.warmup = AsyncMock()
    manager._greeting_provider.generate_stream = immediate_fail_stream
    manager._warmup_task = None

    await manager.trigger_live_greeting()
