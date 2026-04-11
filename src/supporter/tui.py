from __future__ import annotations

import time
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import Input, Label, Static

from . import ChatAgent, CrewAgent, get_provider
from .logger import init_logger, logger

logger.debug("--- Loading tui module ---")

THEME = {
    "background": "#121212",
    "bubble_bg": "#1e1e1e",
    "header_teal": "#00ffcc",
    "magenta": "#ff06b5",
    "green": "#00ff00",
    "blue": "#0080ff",
    "yellow": "#ffeb3b",
    "meta_gray": "#666",
}
CRISTAL_STOPS: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (0, 255, 180),
    (0, 180, 255),
    (100, 200, 255),
]
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _apply_cristal_gradient(text: str) -> Text:
    rich_text = Text(justify="center")
    n = len(text)
    stops = len(CRISTAL_STOPS) - 1
    for i, ch in enumerate(text):
        t = i / max(n - 1, 1)
        seg = min(int(t * stops), stops - 1)
        local_t = t * stops - seg
        (r1, g1, b1) = CRISTAL_STOPS[seg]
        (r2, g2, b2) = CRISTAL_STOPS[seg + 1]
        r = int(r1 + (r2 - r1) * local_t)
        g = int(g1 + (g2 - g1) * local_t)
        b = int(b1 + (b2 - b1) * local_t)
        rich_text.append(ch, style=f"bold rgb({r},{g},{b})")
    return rich_text


class SupporterHeader(Static):
    _ART = (
        " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ "
        + "\n ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ "
        + "\n ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
    )

    def render(self) -> Text:
        return _apply_cristal_gradient(self._ART)


class ChatContainer(Vertical):
    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        for _ in range(5):
            self.scroll_down()
        event.prevent_default()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        for _ in range(5):
            self.scroll_up()
        event.prevent_default()


class MessageBubble(Vertical):
    def __init__(
        self,
        role: str,
        content: str,
        model: str | None = None,
        duration: float | None = None,
        agents: list[str] | None = None,
        streaming: bool = False,
    ):
        super().__init__()
        self.role = role
        self.content = content
        self.model = model
        self.duration = duration
        self.agents = agents
        self.streaming = streaming
        self._bubble_static = None
        self._meta_label = None
        self.add_class("right" if role == "user" else "left")

    def compose(self) -> ComposeResult:
        is_user = self.role == "user"
        border_color = "green" if is_user else "blue"
        self._bubble_static = Static(
            self.content, classes=f"bubble {border_color}", expand=False, markup=False
        )
        yield self._bubble_static

        if (
            not is_user
            and not self.streaming
            and (self.model or self.duration is not None)
        ):
            self._meta_label = Label(self._get_meta_text(), classes="message-meta")
            yield self._meta_label

    def _get_meta_text(self) -> str:
        model_info = self.model or "Unknown"
        if self.duration is not None:
            model_info += f" in {self.duration:.2f}s"
        if self.agents:
            return f"({', '.join(self.agents)} via {model_info})"
        return f"({model_info})"

    def append_token(self, token: str) -> None:
        self.content += token
        if self._bubble_static:
            self._bubble_static.update(self.content)

    async def finalize(
        self,
        model: str | None = None,
        duration: float | None = None,
        agents: list[str] | None = None,
    ) -> None:
        self.model = model or self.model
        self.duration = duration or self.duration
        self.agents = agents or self.agents
        self.streaming = False
        if not self._meta_label:
            self._meta_label = Label(self._get_meta_text(), classes="message-meta")
            await self.mount(self._meta_label)
        else:
            self._meta_label.update(self._get_meta_text())


class SupporterApp(App):
    CSS = f"""
    Screen {{
        background: {THEME["background"]};
        width: 100%;
        height: 100%;
        padding: 0;
        margin: 0;
    }}

    #main-container {{
        width: 100%;
        height: 100%;
        layout: vertical;
        background: transparent;
        overflow-y: scroll;
        scrollbar-size: 0 0;
    }}

    SupporterHeader {{
        height: 5;
        content-align: center middle;
        background: transparent;
        margin-top: 1;
        color: {THEME["header_teal"]};
        text-style: bold;
    }}

    #chat-view {{
        width: 100%;
        height: 1fr;
        padding: 0;
        margin: 0;
        background: transparent;
        layout: vertical;
        overflow-y: scroll;
        scrollbar-size: 0 0;
    }}

    MessageBubble {{
        width: 1fr;
        height: auto;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }}

    MessageBubble.left  {{ align-horizontal: left; }}
    MessageBubble.right {{ align-horizontal: right; }}

    .bubble {{
        width: auto;
        max-width: 60%;
        height: auto;
        padding: 0 1;
        margin: 0;
        background: {THEME["bubble_bg"]};
        border: round #444;
    }}

    .bubble.green {{
        border: round {THEME["magenta"]};
        color: {THEME["magenta"]};
    }}

    .bubble.blue {{
        border: round {THEME["header_teal"]};
        color: {THEME["header_teal"]};
    }}

    .message-meta {{
        color: {THEME["yellow"]};
        text-style: italic;
        margin: 0 0 0 1;
        width: auto;
    }}

    #input-area {{
        height: 3;
        border: solid {THEME["magenta"]};
        margin: 0;
        padding: 0 1;
        background: transparent;
    }}

    #prompt-row {{
        height: 100%;
    }}

    #prompt-row {{
        height: 1;
        align: left middle;
    }}

    #prompt-symbol {{
        color: {THEME["green"]};
        text-style: bold;
        width: 2;
    }}

    Input {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 1 !important;
        color: {THEME["green"]};
    }}

    #user-input {{
        width: 1fr;
    }}

    #thinking-indicator {{
        color: {THEME["yellow"]};
        text-style: italic;
        margin-left: 2;
        height: 1;
    }}
    #user-input:focus {{
        border: solid {THEME["magenta"]};
    }}
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        logger.debug("Initializing SupporterApp")
        super().__init__()
        self.agent = None
        self.active_queries = 0
        self._spinner_timer = None
        self._spinner_idx: int = 0
        self.crew_mode = True
        self.is_activating_crew = False
        self.current_active_agent = ""
        self.status_label = "Thinking"

    def _on_agent_active(self, agent_role: str) -> None:
        self.current_active_agent = agent_role
        self.call_from_thread(self._tick_spinner)

    async def on_mount(self) -> None:
        logger.debug("Entering SupporterApp.on_mount")
        init_logger()
        logger.info("Starting Supporter TUI")
        try:
            await self._setup_agent(use_crew=True)
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Failed to initialize agent: {e}"
            logger.error(msg)
            self.notify(msg, severity="error")

    async def on_unmount(self) -> None:
        logger.debug("SupporterApp unmounting")
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

    async def _setup_agent(self, use_crew: bool = False) -> None:
        logger.debug(f"Entering _setup_agent (use_crew={use_crew})")
        provider = get_provider()
        if use_crew:
            self.agent = CrewAgent(
                provider=provider, status_callback=self._on_agent_active
            )
            logger.info("CrewAgent successfully initialized")
        else:
            self.agent = ChatAgent(
                provider,
                system_instruction=(
                    "You are a helpful assistant. Be concise and professional. "
                    "You can use Google Search and Code Execution when needed."
                ),
                use_search=True,
                use_code_execution=True,
            )
            logger.info("ChatAgent successfully initialized")

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader()
            with ChatContainer(id="chat-view"):
                pass
            yield Static("", id="thinking-indicator", markup=False)
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label(">", id="prompt-symbol")
                yield Input(
                    placeholder=(
                        "Type a message... (/crew to toggle, /clear to reset, "
                        "/exit to quit)"
                    ),
                    id="user-input",
                )

    def action_clear_screen(self) -> None:
        self.action_clear()

    def _start_thinking(self) -> None:
        self.active_queries += 1
        if self.active_queries == 1:
            self._spinner_idx = 0
            if self._spinner_timer:
                self._spinner_timer.stop()
            self._spinner_timer = self.set_interval(0.15, self._tick_spinner)

    def _stop_thinking(self) -> None:
        self.active_queries = max(0, self.active_queries - 1)
        if self.active_queries == 0:
            if self._spinner_timer:
                self._spinner_timer.stop()
                self._spinner_timer = None
            self.query_one("#thinking-indicator", Static).update("")

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * (self._spinner_idx % 4)
        self._spinner_idx += 1
        if self.is_activating_crew:
            status = f"Activating CrewAI{dots}"
        else:
            if self.crew_mode and self.current_active_agent:
                label = f"[{self.current_active_agent}]"
            else:
                label = "[AGENT]" if self.crew_mode else ""
            status = f"{frame} {label} {self.status_label}{dots}"
        self.query_one("#thinking-indicator", Static).update(status)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        self.query_one("#user-input", Input).value = ""
        self.query_one("#user-input").focus()
        if user_text.startswith("/"):
            await self._handle_command(user_text.lower())
            return
        chat_view = self.query_one("#chat-view")
        chat_view.mount(MessageBubble(role="user", content=user_text))
        chat_view.scroll_end()
        self.run_worker(self._process_message_cycle(user_text, mount_user=False))

    async def _handle_command(self, command: str) -> None:
        logger.debug(f"Handling TUI command: {command}")
        if command == "/exit":
            self.exit()
        elif command == "/clear":
            if self.agent:
                self.agent.clear_history()
            self.query_one("#chat-view").query("*").remove()
        elif command == "/crew":
            self.crew_mode = not self.crew_mode
            status = "ENABLED" if self.crew_mode else "DISABLED"
            self._start_thinking()
            self.is_activating_crew = True
            try:
                await self._setup_agent(use_crew=self.crew_mode)
                chat_view = self.query_one("#chat-view")
                await chat_view.mount(
                    MessageBubble(role="agent", content=f"Multi-Agent Mode {status}")
                )
                chat_view.scroll_end()
            finally:
                self.is_activating_crew = False
                self._stop_thinking()

    async def _process_message_cycle(self, text: str, mount_user: bool = True) -> None:
        logger.debug("Entering _process_message_cycle")
        chat_view = self.query_one("#chat-view")
        if mount_user:
            await chat_view.mount(MessageBubble(role="user", content=text))
            chat_view.scroll_end()
            self.query_one("#user-input").focus()
        self.current_active_agent = ""
        self.status_label = "Thinking"
        self._start_thinking()
        start_time = time.perf_counter()
        try:
            if not self.agent:
                raise RuntimeError("Agent is not initialized")

            if isinstance(self.agent, CrewAgent):
                response = await self.agent.execute(text)
                ui_duration = time.perf_counter() - start_time
                agent_roles = response.usage.get("agents") if response.usage else None
                await chat_view.mount(
                    MessageBubble(
                        role="agent",
                        content=response.text,
                        model=response.model,
                        duration=ui_duration,
                        agents=agent_roles,
                    )
                )
                chat_view.scroll_end()
            else:
                bubble = None
                first_chunk = True
                accumulated_text = ""
                actual_model = None
                async for chunk in self.agent.execute_stream(text):
                    if chunk.model:
                        actual_model = chunk.model
                    if first_chunk:
                        accumulated_text += chunk.text
                        if not accumulated_text.strip():
                            continue
                        first_chunk = False
                        self.status_label = "Streaming"
                        bubble = MessageBubble(
                            role="agent", content=accumulated_text, streaming=True
                        )
                        await chat_view.mount(bubble)
                        chat_view.scroll_end()
                    else:
                        if bubble:
                            bubble.append_token(chunk.text)
                            chat_view.scroll_end()

                if bubble:
                    ui_duration = time.perf_counter() - start_time
                    await bubble.finalize(model=actual_model, duration=ui_duration)
                    chat_view.scroll_end()
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            await chat_view.mount(MessageBubble(role="agent", content=f"Error: {e}"))
            chat_view.scroll_end()
        finally:
            self._stop_thinking()
        logger.debug("Exiting _process_message_cycle")


def main():
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
