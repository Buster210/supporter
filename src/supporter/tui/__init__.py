from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Static

from .. import ChatAgent, CrewAgent
from ..logger import init_logger, logger
from .message_processor import ChatMessageProcessor
from .mode_manager import ModeManager, SpinnerController
from .widgets import (
    ChatContainer,
    ChatTurn,
    MessageBubble,
    SupporterHeader,
)

CSS = (Path(__file__).parent / "styles.css").read_text()


class SupporterApp(App[None]):
    CSS = CSS

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: ChatAgent | CrewAgent | None = None
        self.active_queries = 0
        self.crew_mode = False
        self.live_mode = True
        self.is_activating_mode = False
        self.status_label = "Thinking"
        self.current_active_agent: str = ""
        self._mode_manager = ModeManager(self)
        self._message_processor = ChatMessageProcessor(self)

    def _on_agent_active(self, agent_role: str) -> None:
        self.current_active_agent = agent_role
        self.call_from_thread(self._tick_spinner)

    async def on_mount(self) -> None:
        init_logger()
        logger.info("Supporter TUI dashboard active")
        try:
            await self._setup_agent(use_live=True)
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure: {e}"
            logger.error(msg)
            self.notify(msg, severity="error")

    async def on_unmount(self) -> None:
        self._spinner_controller.shutdown()
        self._mode_manager.shutdown()

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
            yield Static("", id="thinking-indicator", markup=False)
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

    def _start_thinking(self) -> None:
        self._spinner_controller.start()

    def _stop_thinking(self) -> None:
        self._spinner_controller.stop()

    @property
    def _spinner_controller(self) -> SpinnerController:
        if not hasattr(self, "__spinner_controller"):
            object.__setattr__(self, "__spinner_controller", SpinnerController(self))
        return cast(
            SpinnerController,
            object.__getattribute__(self, "__spinner_controller"),
        )

    def _tick_spinner(self) -> None:
        self._spinner_controller._tick_spinner()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        input_widget.focus()
        chat_view = self.query_one("#chat-view", Vertical)
        self.active_turn = None

        for turn in chat_view.query(ChatTurn):
            turn.collapse()

        user_bubble = MessageBubble(role="user", content=user_text)
        self.active_turn = ChatTurn(user_bubble)
        chat_view.mount(self.active_turn)
        chat_view.scroll_end()

        if user_text.startswith("/"):
            await self._handle_command(user_text.lower())
            return

        self.run_worker(self._process_message_cycle(user_text, mount_user=False))

    async def _handle_command(self, command: str) -> None:
        await self._mode_manager.handle_command(command)

    async def _toggle_mode(self, crew: bool = False, live: bool = False) -> None:
        await self._mode_manager.toggle_mode(crew=crew, live=live)

    async def _process_message_cycle(self, text: str, mount_user: bool = True) -> None:
        chat_view = self.query_one("#chat-view", Vertical)
        if mount_user:
            user_bubble = MessageBubble(role="user", content=text)
            self.active_turn = ChatTurn(user_bubble)
            await chat_view.mount(self.active_turn)
            chat_view.scroll_end()
            self.query_one("#user-input").focus()

        target_container = (
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
            self._stop_thinking()

    async def _process_crew_execution(
        self, text: str, target: Vertical, start_time: float
    ) -> None:
        await self._message_processor.process_crew(text, target, start_time)

    async def _process_streaming_execution(
        self, text: str, target: Vertical, start_time: float, agent: ChatAgent
    ) -> None:
        await self._message_processor.process_streaming(text, target, start_time, agent)


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
