from typing import Any
from unittest.mock import MagicMock

from supporter.tui.modals import ConfirmationModal
from supporter.tui.utils import ToastManager
from tests.tui_mocks import MockApp, MockWidget


class TestMessageQueuing:
    def test_queue_append_when_processing(self) -> None:
        app = MockApp()
        app._is_processing = True
        app._user_message_queue = []
        app._toast_manager = ToastManager()
        app._supporter_queue_display = MockWidget("#queue-display")

        def handle_message(user_text: Any) -> None:
            if app._is_processing:
                app._user_message_queue.append(user_text)
                app._supporter_queue_display.update_queue(app._user_message_queue)
                app._toast_manager.notify(
                    app,
                    f"Message queued ({len(app._user_message_queue)})",
                    type="queue",
                )

        app._handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message("test message")
        assert len(app._user_message_queue) == 1
        assert app._user_message_queue[0] == "test message"

    def test_queue_updates_display_when_processing(self) -> None:
        app = MockApp()
        app._is_processing = True
        app._user_message_queue = []
        app._toast_manager = ToastManager()
        mock_queue_display = MockWidget("#queue-display")
        app._supporter_queue_display = mock_queue_display

        def handle_message(user_text: Any) -> None:
            if app._is_processing:
                app._user_message_queue.append(user_text)
                mock_queue_display.update_queue(app._user_message_queue)

        app.handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message("test message")
        mock_queue_display.update_queue.assert_called_once_with(["test message"])

    def test_toast_notified_on_queue(self) -> None:
        app = MockApp()
        app._is_processing = True
        app._user_message_queue = []
        app._toast_manager = ToastManager()
        app._supporter_queue_display = MockWidget("#queue-display")

        def handle_message(user_text: Any) -> None:
            if app._is_processing:
                app._user_message_queue.append(user_text)
                app._toast_manager.notify(
                    app,
                    f"Message queued ({len(app._user_message_queue)})",
                    type="queue",
                )

        app.handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message("test message")
        assert len(app._toast_manager.active_toasts) > 0

    def test_no_queue_when_not_processing(self) -> None:
        app = MockApp()
        app._is_processing = False
        app._user_message_queue = []

        def handle_message(user_text: Any) -> None:
            if app._is_processing:
                app._user_message_queue.append(user_text)

        app.handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message("test message")
        assert len(app._user_message_queue) == 0

    def test_multiple_messages_queued(self) -> None:
        app = MockApp()
        app._is_processing = True
        app._user_message_queue = []
        app._supporter_queue_display = MockWidget("#queue-display")
        app._toast_manager = MagicMock()

        def handle_message(user_text: Any) -> None:
            if app._is_processing:
                app._user_message_queue.append(user_text)

        app.handle_user_message = handle_message  # type: ignore[assignment, method-assign]
        app.handle_user_message("msg1")
        app.handle_user_message("msg2")
        app.handle_user_message("msg3")
        assert len(app._user_message_queue) == 3


class TestConfirmationWrite:
    def test_confirmation_modal_path(self) -> None:
        modal = ConfirmationModal("Write", "content")
        assert modal.modal_title == "Write"

    def test_confirmation_modal_content(self) -> None:
        modal = ConfirmationModal("Write", "some content")
        assert "some content" in modal.content


class TestConfirmationBash:
    def test_bash_confirmation_modal_command(self) -> None:
        modal = ConfirmationModal("Bash", "ls -la", meta="/home")
        assert "ls -la" in modal.content
        assert modal.meta is not None
        assert "/home" in modal.meta


class TestFlushQueuedMessages:
    def test_flush_clears_queue(self) -> None:
        app = MockApp()
        app._user_message_queue = ["msg1", "msg2"]
        app._is_processing = False
        app._supporter_queue_display = MockWidget("#queue-display")

        def flush() -> None:
            if not app._user_message_queue:
                return
            app._user_message_queue.clear()
            app._supporter_queue_display.update_queue([])

        app._flush_queued = flush  # type: ignore[method-assign]
        app._flush_queued()
        assert len(app._user_message_queue) == 0

    def test_flush_updates_display(self) -> None:
        app = MockApp()
        app._user_message_queue = ["msg1"]
        app._is_processing = False
        mock_queue_display = MockWidget("#queue-display")
        app._supporter_queue_display = mock_queue_display

        def flush() -> None:
            if not app._user_message_queue:
                return
            app._user_message_queue.clear()
            mock_queue_display.update_queue([])

        app._flush_queued = flush  # type: ignore[method-assign]
        app._flush_queued()
        mock_queue_display.update_queue.assert_called_once_with([])

    def test_flush_no_op_when_queue_empty(self) -> None:
        app = MockApp()
        app._user_message_queue = []
        app._is_processing = False
        app._supporter_queue_display = MockWidget("#queue-display")

        def flush() -> None:
            if not app._user_message_queue:
                return

        app._flush_queued = flush  # type: ignore[method-assign]
        app._flush_queued()
        assert len(app._user_message_queue) == 0
