from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Input, Label

from .. import ChatAgent, CrewAgent
from ..logger import init_logger, logger
from ..tools import set_confirmation_callback
from .message_processor import AgentActive, ChatMessageProcessor, ModeChanged
from .mode_manager import ModeManager
from .widgets import (
    ChatContainer,
    ChatTurn,
    MessageBubble,
    QueuedMessagesDisplay,
    SupporterHeader,
    ThinkingIndicator,
    ToastManager,
)

CSS = (Path(__file__).parent / "styles.tcss").read_text()


class SupporterApp(App[None]):
    CSS = CSS

    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)
    crew_mode = reactive(False)
    live_mode = reactive(True)
    active_turn: reactive[ChatTurn | None] = reactive(None)
    current_active_agent = reactive("")

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: ChatAgent | CrewAgent | None = None
        self._mode_manager = ModeManager(self)
        self._message_processor = ChatMessageProcessor(self)
        self._is_processing = False
        self._user_message_queue: list[str] = []
        self._toast_manager = ToastManager()

    def _on_agent_active(self, agent_role: str) -> None:
        self.post_message(AgentActive(agent_role=agent_role))

    def on_agent_active(self, event: AgentActive) -> None:
        self.current_active_agent = event.agent_role

    async def on_mode_changed(self, event: ModeChanged) -> None:
        indicator = self.query_one("#mode-indicator", Label)
        indicator.update(f"[{event.mode}]")

        label = "Multi-Agent Crew" if event.mode == "CREW" else "Single Agent"
        status = "ENABLED" if event.enabled else "DISABLED"

        target = self.active_turn or self.query_one("#chat-view")
        await target.mount(MessageBubble(role="agent", content=f"{label} {status}"))

    async def on_mount(self) -> None:
        init_logger()
        set_confirmation_callback(self._confirm_write)

        logger.info("Supporter TUI dashboard active")
        try:
            await self._setup_agent(use_live=True)
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure: {e}"
            logger.error(msg)
            self._toast_manager.notify(self, msg, type="system")

    async def on_unmount(self) -> None:
        set_confirmation_callback(None)

        if (
            self.agent
            and hasattr(self.agent, "provider")
            and hasattr(self.agent.provider, "close")
        ):
            await self.agent.provider.close()

    async def _setup_agent(
        self, use_crew: bool = False, use_live: bool = False
    ) -> None:
        await self._mode_manager.setup_agent(use_crew=use_crew, use_live=use_live)

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader()
            with ChatContainer(id="chat-view"):
                pass
            yield QueuedMessagesDisplay(id="queue-display")
            yield ThinkingIndicator(id="thinking-indicator")
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label("[LIVE]", id="mode-indicator", markup=False)
                yield Label(">", id="prompt-symbol")
                yield Input(
                    placeholder="Type a message... (/live, /crew, /clear, /exit)",
                    id="user-input",
                )

    def action_clear_screen(self) -> None:
        if self.agent:
            self.agent.clear_history()
        self.query_one("#chat-view").query("*").remove()
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
        chat_view.mount(new_turn)
        chat_view.scroll_end()

        self._is_processing = True
        self.run_worker(
            self._process_message_cycle(user_text, mount_user=False, target=new_turn)
        )

    async def _handle_command(self, command: str) -> bool:
        return await self._mode_manager.handle_command(command)

    async def _toggle_mode(self, crew: bool = False, live: bool = False) -> None:
        await self._mode_manager.toggle_mode(crew=crew, live=live)

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
            agent = self.agent
            if not agent:
                raise RuntimeError("Agent is not initialized")

            if isinstance(agent, CrewAgent):
                await self._process_crew_execution(text, target_container, start_time)
            else:
                await self._process_streaming_execution(
                    text, target_container, start_time, agent
                )
        except Exception as e:
            logger.error(f"Execution error: {e}")
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

    async def _process_crew_execution(
        self, text: str, target: Vertical, start_time: float
    ) -> None:
        await self._message_processor.process_crew(text, target, start_time)

    async def _process_streaming_execution(
        self, text: str, target: Vertical, start_time: float, agent: ChatAgent
    ) -> None:
        await self._message_processor.process_streaming(text, target, start_time, agent)

    def _confirm_write(self, path: Path, content: str) -> bool:
        import threading

        from .widgets import ConfirmationModal

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


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
