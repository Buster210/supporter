from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.tui import SupporterApp
from supporter.types import ModeChanged
from tests.tui_mocks import MockWidget

pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine 'AsyncMockMixin._execute_mock_call' "
    "was never awaited:RuntimeWarning"
)


@pytest.mark.asyncio
async def test_mode_changed_handler_updates_indicator_label() -> None:
    app = SupporterApp()
    mock_indicator = MockWidget("mode-indicator")
    mock_chat_view = MockWidget("chat-view")
    with patch.object(
        app,
        "query_one",
        side_effect=lambda selector, type=None: (
            mock_indicator if selector == "#mode-indicator" else mock_chat_view
        ),
    ):
        mock_indicator.content = ""
        event = ModeChanged(mode="LIVE", enabled=True)
        await app.on_mode_changed(event)
        assert mock_indicator.content == "[LIVE]"


@pytest.mark.asyncio
async def test_mode_changed_handler_mounts_system_messages() -> None:
    app = SupporterApp()
    mock_indicator = MockWidget("mode-indicator")
    mock_chat_view = MockWidget("chat-view")
    with patch.object(
        app,
        "query_one",
        side_effect=lambda selector, type=None: (
            mock_indicator if selector == "#mode-indicator" else mock_chat_view
        ),
    ):
        event = ModeChanged(mode="LIVE", enabled=True)
        await app.on_mode_changed(event)
        assert len(mock_chat_view.mounted) == 1
        assert "Single Agent" in mock_chat_view.mounted[0].content


@pytest.mark.asyncio
async def test_mode_changed_uses_active_turn_when_available() -> None:
    app = SupporterApp()
    mock_indicator = MockWidget("mode-indicator")
    mock_chat_view = MockWidget("chat-view")

    class ActiveTurn:
        def __init__(self) -> None:
            self.mounted: list[Any] = []

        async def mount(self, widget: Any) -> None:
            self.mounted.append(widget)

    mock_active_turn = ActiveTurn()
    from typing import cast

    from supporter.tui.chat import ChatTurn

    with patch.object(
        app,
        "query_one",
        side_effect=lambda selector, type=None: (
            mock_indicator if selector == "#mode-indicator" else mock_chat_view
        ),
    ):
        app.active_turn = cast(ChatTurn, mock_active_turn)
        event = ModeChanged(mode="LIVE", enabled=True)
        await app.on_mode_changed(event)
        assert len(mock_active_turn.mounted) == 1
        assert len(mock_chat_view.mounted) == 0


@pytest.mark.asyncio
async def test_handle_command_routes_to_mode_manager() -> None:
    app = SupporterApp()
    app._is_processing = True
    app._user_message_queue = []
    mock_queue_display = MockWidget("queue-display")
    with patch.object(app, "query_one", return_value=mock_queue_display):
        await app._flush_queued_messages()
        mock_queue_display.update_queue.assert_not_called()


@pytest.mark.asyncio
async def test_flush_queued_messages_processes_batch() -> None:
    app = SupporterApp()
    app._user_message_queue = ["Hello", "World", "Test"]
    mock_queue_display = MockWidget("queue-display")
    with (
        patch.object(app, "query_one", return_value=mock_queue_display),
        patch.object(app, "run_worker") as mock_run_worker,
    ):
        mock_run_worker.side_effect = lambda coro: (
            coro.close() if hasattr(coro, "close") else None
        )
        await app._flush_queued_messages()
    assert len(app._user_message_queue) == 0
    mock_queue_display.update_queue.assert_called_with([])
    mock_run_worker.assert_called_once()


@pytest.mark.asyncio
async def test_flush_queued_messages_sets_processing_flag() -> None:
    app = SupporterApp()
    app._user_message_queue = ["Message 1", "Message 2"]
    app._is_processing = False
    mock_queue_display = MockWidget("queue-display")
    with (
        patch.object(app, "query_one", return_value=mock_queue_display),
        patch.object(app, "run_worker") as mock_run_worker,
    ):
        mock_run_worker.side_effect = lambda coro: (
            coro.close() if hasattr(coro, "close") else None
        )
        await app._flush_queued_messages()
    assert app._is_processing is True


@pytest.mark.asyncio
async def test_queue_accumulates_during_processing() -> None:
    app = SupporterApp()
    app._is_processing = True
    app._user_message_queue = []
    mock_input = MagicMock()
    mock_input.value = "Test message"
    mock_queue_display = MockWidget("queue-display")
    with patch.object(
        app,
        "query_one",
        side_effect=lambda selector, type=None: (
            mock_queue_display if selector == "#queue-display" else mock_input
        ),
    ):
        event = MagicMock()
        event.value = "Test message"
        await app.on_input_submitted(event)
    assert len(app._user_message_queue) == 1
    assert app._user_message_queue[0] == "Test message"
    mock_queue_display.update_queue.assert_called_with(["Test message"])


@pytest.mark.asyncio
async def test_queue_combines_multiple_messages() -> None:
    app = SupporterApp()
    app._user_message_queue = ["First message", "Second message", "Third message"]
    mock_queue_display = MockWidget("queue-display")
    with (
        patch.object(app, "query_one", return_value=mock_queue_display),
        patch.object(app, "run_worker") as mock_run_worker,
        patch.object(app, "_process_message_cycle") as mock_cycle,
    ):
        await app._flush_queued_messages()
    mock_run_worker.assert_called_once()
    mock_cycle.assert_called_once()
    call_args = mock_cycle.call_args[0][0]
    assert "First message" in str(call_args)
    assert "Second message" in str(call_args)
    assert "Third message" in str(call_args)


@pytest.mark.asyncio
async def test_on_input_submitted_queues_when_processing() -> None:
    app = SupporterApp()
    app._is_processing = True
    app._user_message_queue = []
    mock_input = MagicMock()
    mock_input.value = " queued message "
    mock_queue_display = MockWidget("queue-display")

    def mock_query(selector: Any, type: Any = None) -> Any:
        if selector == "#queue-display":
            return mock_queue_display
        return mock_input

    with patch.object(app, "query_one", side_effect=mock_query):
        event = MagicMock()
        event.value = " queued message "
        await app.on_input_submitted(event)
    assert "queued message" in app._user_message_queue


@pytest.mark.asyncio
async def test_on_input_submitted_ignores_empty() -> None:
    app = SupporterApp()
    app._is_processing = False
    app._user_message_queue = []
    mock_input = MagicMock()
    mock_input.value = "   "
    with patch.object(app, "query_one", return_value=mock_input):
        event = MagicMock()
        event.value = "   "
        await app.on_input_submitted(event)
    assert len(app._user_message_queue) == 0


@pytest.mark.asyncio
async def test_on_input_submitted_starts_processing() -> None:
    app = SupporterApp()
    app._is_processing = False
    app._user_message_queue = []
    app.agent = MagicMock()
    mock_input = MagicMock()
    mock_input.value = "Hello"
    mock_chat_view = MockWidget("chat-view")

    def mock_query(selector: Any, type: Any = None) -> Any:
        if selector == "#chat-view":
            return mock_chat_view
        return mock_input

    with (
        patch.object(app, "query_one", side_effect=mock_query),
        patch.object(app, "run_worker") as mock_run_worker,
    ):
        mock_run_worker.side_effect = lambda coro: (
            coro.close() if hasattr(coro, "close") else None
        )
        event = MagicMock()
        event.value = "Hello"
        await app.on_input_submitted(event)
        assert app._is_processing is True
        mock_run_worker.assert_called_once()
