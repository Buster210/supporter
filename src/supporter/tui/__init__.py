from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label

from .. import ChatAgent, DynamicPool
from ..logger import init_logger, logger
from ..tools import (
    set_bash_confirmation_callback,
    set_bash_notification_callback,
    set_confirmation_callback,
)
from ..types import ModeChanged
from .bubble import MessageBubble
from .chat import (
    ChatContainer,
    ChatTurn,
    QueuedMessagesDisplay,
    SupporterHeader,
    ThinkingIndicator,
)
from .message_processor import ChatMessageProcessor
from .mode_manager import ModeManager
from .utils import ToastManager

CSS = (Path(__file__).parent / "styles.tcss").read_text()


class SupporterApp(App[None]):
    CSS = CSS

    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)
    live_mode = reactive(True)
    active_turn: reactive[ChatTurn | None] = reactive(None)

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: ChatAgent | None = None
        self._mode_manager = ModeManager(self)
        self._message_processor = ChatMessageProcessor(self)
        self._is_processing = False
        self._user_message_queue: list[str] = []
        self._toast_manager = ToastManager()

    async def on_mode_changed(self, event: ModeChanged) -> None:
        indicator = self.query_one("#mode-indicator", Label)
        indicator.update(f"[{event.mode}]")
        status = "ENABLED" if event.enabled else "DISABLED"

        target = self.active_turn or self.query_one("#chat-view")
        await target.mount(
            MessageBubble(role="agent", content=f"Single Agent {status}")
        )

    async def on_mount(self) -> None:
        init_logger()
        set_confirmation_callback(self._confirm_write)
        set_bash_confirmation_callback(self._confirm_bash)
        set_bash_notification_callback(self._notify_error)

        logger.info("Supporter TUI dashboard active")
        try:
            await self._setup_agent(use_live=True)
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure [{type(e).__name__}]: {e}"
            logger.error(msg)
            self._toast_manager.notify(self, msg, type="system")

    async def on_unmount(self) -> None:
        set_confirmation_callback(None)
        set_bash_confirmation_callback(None)
        set_bash_notification_callback(None)

        if (
            self.agent
            and hasattr(self.agent, "provider")
            and hasattr(self.agent.provider, "close")
        ):
            await self.agent.provider.close()

        await DynamicPool.shutdown_all()

    async def _setup_agent(self, use_live: bool = False) -> None:
        await self._mode_manager.setup_agent(use_live=use_live)

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader(id="supporter-header")
            with ChatContainer(id="chat-view"):
                pass
            yield QueuedMessagesDisplay(id="queue-display")
            yield ThinkingIndicator(id="thinking-indicator")
            with Horizontal(id="scroll-btn-wrapper", classes="hidden"):
                yield Button("↓ Go to bottom", id="scroll-bottom-btn")
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label("[LIVE]", id="mode-indicator", markup=False)
                yield Label(">", id="prompt-symbol")
                yield Input(
                    placeholder="Type a message... (/agent, /live, /clear, /exit)",
                    id="user-input",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scroll-bottom-btn":
            chat_view = self.query_one("#chat-view", Vertical)
            chat_view.scroll_end(animate=True)
            event.button.add_class("hidden")

    def action_clear_screen(self) -> None:
        chat_view = self.query_one("#chat-view")
        if not chat_view.query("*") and (not self.agent or not self.agent.history):
            self._toast_manager.notify(self, "Session already clear", type="system")
            return

        if self.agent:
            self.agent.clear_history()
        chat_view.query("*").remove()
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

    def _start_thinking(self) -> None:
        self.active_queries += 1

    def _stop_thinking(self) -> None:
        self.active_queries = max(0, self.active_queries - 1)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        input_widget.focus()

        if user_text.startswith("/") and await self._handle_command(user_text):
            return

        if self._is_processing:
            self._user_message_queue.append(user_text)
            self.query_one("#queue-display", QueuedMessagesDisplay).update_queue(
                self._user_message_queue
            )
            self._toast_manager.notify(
                self, f"Message queued ({len(self._user_message_queue)})", type="queue"
            )
            return

        chat_view = self.query_one("#chat-view", Vertical)
        user_bubble = MessageBubble(role="user", content=user_text)
        new_turn = ChatTurn(user_bubble)
        if self.active_turn:
            self.active_turn.auto_collapse()
        self.active_turn = new_turn
        await chat_view.mount(new_turn)
        chat_view.scroll_end()

        self._is_processing = True
        self.run_worker(
            self._process_message_cycle(user_text, mount_user=False, target=new_turn)
        )

    async def _handle_command(self, command: str) -> bool:
        return await self._mode_manager.handle_command(command)

    async def _toggle_mode(self, live: bool = False) -> None:
        await self._mode_manager.toggle_mode(live=live)

    async def _process_message_cycle(
        self, text: str, mount_user: bool = True, target: Vertical | None = None
    ) -> None:
        self._is_processing = True
        chat_view = self.query_one("#chat-view", Vertical)
        if mount_user:
            user_bubble = MessageBubble(role="user", content=text)
            new_turn = ChatTurn(user_bubble)
            if self.active_turn:
                self.active_turn.auto_collapse()
            self.active_turn = new_turn
            await chat_view.mount(new_turn)
            chat_view.scroll_end()
            self.query_one("#user-input").focus()

        target_container = target or (
            self.active_turn if hasattr(self, "active_turn") else chat_view
        )
        if not isinstance(target_container, Vertical):
            raise RuntimeError("Invalid UI state: chat view container missing")

        self.status_label = "Thinking"
        self._start_thinking()
        start_time = time.perf_counter()

        try:
            if not self.agent:
                raise RuntimeError("Agent is not initialized")
            await self._process_streaming_execution(
                text, target_container, start_time, self.agent
            )
        except Exception as e:
            logger.error(f"UI Message Cycle Error [{type(e).__name__}]: {e}")
            await chat_view.mount(MessageBubble(role="agent", content=f"Error: {e}"))
        finally:
            self._is_processing = False
            self._stop_thinking()
            await self._flush_queued_messages()

    async def _flush_queued_messages(self) -> None:
        if not self._user_message_queue:
            return

        texts = list(self._user_message_queue)
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

        combined_text = "\n\n".join(texts)
        self._is_processing = True
        self.run_worker(self._process_message_cycle(combined_text, mount_user=True))

    async def _process_streaming_execution(
        self, text: str, target: Vertical, start_time: float, agent: ChatAgent
    ) -> None:
        await self._message_processor.process_streaming(text, target, start_time, agent)

    def _confirm_write(self, path: Path, content: str) -> bool:
        import threading

        from .modals import ConfirmationModal

        event = threading.Event()
        result = [False]

        def callback(value: bool) -> None:
            result[0] = value
            event.set()

        self.call_from_thread(
            self.push_screen, ConfirmationModal(str(path), content), callback
        )
        event.wait()
        return result[0]

    def _confirm_bash(self, tokens: list[str], cwd: str) -> bool:
        import threading

        from .modals import BashConfirmationModal

        event = threading.Event()
        result = [False]

        def callback(value: bool) -> None:
            result[0] = value
            event.set()

        self.call_from_thread(
            self.push_screen, BashConfirmationModal(tokens, cwd), callback
        )
        event.wait()
        return result[0]

    def _notify_error(self, message: str) -> None:
        import threading

        if threading.current_thread() is threading.main_thread():
            self._toast_manager.notify(self, message, type="error")
        else:
            self.call_from_thread(
                self._toast_manager.notify, self, message, type="error"
            )


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
