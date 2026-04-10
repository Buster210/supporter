from __future__ import annotations
import asyncio
import time
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import Input, Label, Static
from . import ChatAgent, CrewAgent, get_provider
from .logger import init_logger, logger

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
    ):
        super().__init__()
        self.role = role
        self.content = content
        self.model = model
        self.duration = duration
        self.agents = agents
        self.add_class("right" if role == "user" else "left")

    def compose(self) -> ComposeResult:
        is_user = self.role == "user"
        border_color = "green" if is_user else "blue"
        yield Static(
            self.content, classes=f"bubble {border_color}", expand=False, markup=False
        )
        if not is_user and (self.model or self.duration is not None):
            meta_parts = []
            if self.agents:
                meta_parts.append(", ".join(self.agents))
            model_info = self.model
            if self.duration is not None:
                model_info += f" in {self.duration:.2f}s"
            if self.agents:
                meta_text = f"({', '.join(self.agents)} via {model_info})"
            else:
                meta_text = f"({model_info})"
            yield Label(meta_text, classes="message-meta")


class SupporterApp(App):
    CSS = f"\n    Screen {{\n        background: {THEME['background']};\n        width: 100%;\n        height: 100%;\n        padding: 0;\n        margin: 0;\n    }}\n\n\n    #main-container {{\n        width: 100%;\n        height: 100%;\n        layout: vertical;\n        background: transparent;\n        overflow-y: scroll;\n        scrollbar-size: 0 0;\n    }}\n\n    SupporterHeader {{\n        height: 5;\n        content-align: center middle;\n        background: transparent;\n        margin-top: 1;\n        color: {THEME['header_teal']};\n        text-style: bold;\n    }}\n\n    #chat-view {{\n        width: 100%;\n        height: 1fr;\n        padding: 0;\n        margin: 0;\n        background: transparent;\n        layout: vertical;\n        overflow-y: scroll;\n        scrollbar-size: 0 0;\n    }}\n\n    MessageBubble {{\n        width: 1fr;\n        height: auto;\n        margin: 0;\n        padding: 0;\n        overflow: hidden;\n    }}\n\n    MessageBubble.left  {{ align-horizontal: left; }}\n    MessageBubble.right {{ align-horizontal: right; }}\n\n    .bubble {{\n        width: auto;\n        max-width: 60%;\n        height: auto;\n        padding: 0 1;\n        margin: 0;\n        background: {THEME['bubble_bg']};\n        border: round #444;\n    }}\n\n    .bubble.green {{\n        border: round {THEME['magenta']};\n        color: {THEME['magenta']};\n    }}\n\n    .bubble.blue {{\n        border: round {THEME['header_teal']};\n        color: {THEME['header_teal']};\n    }}\n\n    .message-meta {{\n        color: {THEME['yellow']};\n        text-style: italic;\n        margin: 0 0 0 1;\n        width: auto;\n    }}\n\n    #input-area {{\n        height: 3;\n        border: solid {THEME['magenta']};\n        margin: 0;\n        padding: 0 1;\n        background: transparent;\n    }}\n\n    #prompt-row {{\n        height: 100%;\n    }}\n\n    #prompt-row {{\n        height: 1;\n        align: left middle;\n    }}\n\n    #prompt-symbol {{\n        color: {THEME['green']};\n        text-style: bold;\n        width: 2;\n    }}\n\n    Input {{\n        background: transparent !important;\n        border: none !important;\n        padding: 0 !important;\n        margin: 0 !important;\n        height: 1 !important;\n        color: {THEME['green']};\n    }}\n\n    #user-input {{\n        width: 1fr;\n    }}\n\n    #thinking-indicator {{\n        color: {THEME['yellow']};\n        text-style: italic;\n        margin-left: 2;\n        height: 1;\n    }}\n    "
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.is_thinking = False
        self._spinner_timer = None
        self._spinner_idx: int = 0
        self.crew_mode = True
        self.is_activating_crew = False
        self.current_active_agent = ""

    def _on_agent_active(self, agent_role: str) -> None:
        self.current_active_agent = agent_role
        self.call_from_thread(self._tick_spinner)

    async def on_mount(self) -> None:
        init_logger()
        logger.info("Starting Supporter TUI")
        try:
            await self._setup_agent(use_crew=True)
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Failed to initialize agent: {e}"
            logger.error(msg)
            self.notify(msg, severity="error")

    async def _setup_agent(self, use_crew: bool = False) -> None:

        def _initialize():
            provider = get_provider()
            if use_crew:
                agent = CrewAgent(status_callback=self._on_agent_active)
                logger.info("CrewAgent successfully initialized")
                return agent
            else:
                agent = ChatAgent(
                    provider,
                    {
                        "system_instruction": "You are a helpful assistant. Be concise and professional. You can use Google Search and Code Execution when needed.",
                        "use_search": True,
                        "use_code_execution": True,
                    },
                )
                logger.info("ChatAgent successfully initialized")
                return agent

        self.agent = await asyncio.to_thread(_initialize)

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader()
            with ChatContainer(id="chat-view"):
                pass
            yield Static("", id="thinking-indicator", markup=False)
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label("❯", id="prompt-symbol")
                yield Input(
                    placeholder="Type a message... (/crew to toggle, /clear to reset, /exit to quit)",
                    id="user-input",
                )

    def action_clear_screen(self) -> None:
        self.action_clear()

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
            status = f"{frame} {label} Thinking{dots}"
        self.query_one("#thinking-indicator", Static).update(status)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text or self.is_thinking:
            return
        self.query_one("#user-input", Input).value = ""
        if user_text.startswith("/"):
            await self._handle_command(user_text.lower())
            return
        await self._process_message_cycle(user_text)

    async def _handle_command(self, command: str) -> None:
        if command == "/exit":
            self.exit()
        elif command == "/clear":
            if self.agent:
                self.agent.clear_history()
            self.query_one("#chat-view").query("*").remove()
        elif command == "/crew":
            self.crew_mode = not self.crew_mode
            status = "ENABLED" if self.crew_mode else "DISABLED"
            self.is_thinking = True
            self.is_activating_crew = True
            self._spinner_idx = 0
            self._spinner_timer = self.set_interval(0.15, self._tick_spinner)
            try:
                await self._setup_agent(use_crew=self.crew_mode)
                chat_view = self.query_one("#chat-view")
                await chat_view.mount(
                    MessageBubble(role="agent", content=f"Multi-Agent Mode {status}")
                )
                chat_view.scroll_end(animate=False)
            finally:
                self.is_thinking = False
                self.is_activating_crew = False
                if self._spinner_timer:
                    self._spinner_timer.stop()
                    self._spinner_timer = None
                self.query_one("#thinking-indicator", Static).update("")

    async def _process_message_cycle(self, text: str) -> None:
        chat_view = self.query_one("#chat-view")
        await chat_view.mount(MessageBubble(role="user", content=text))
        chat_view.scroll_end(animate=False)
        self.current_active_agent = ""
        self.is_thinking = True
        self._spinner_idx = 0
        self._spinner_timer = self.set_interval(0.15, self._tick_spinner)
        start_time = time.perf_counter()
        try:
            if not self.agent:
                raise RuntimeError("Agent is not initialized")
            response = await self.agent.execute(text)
            duration = time.perf_counter() - start_time
            await chat_view.mount(
                MessageBubble(
                    role="agent",
                    content=response["text"],
                    model=response.get("model"),
                    duration=response.get("duration"),
                    agents=response.get("agents"),
                )
            )
            chat_view.scroll_end(animate=False)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            await chat_view.mount(MessageBubble(role="agent", content=f"Error: {e}"))
            chat_view.scroll_end(animate=False)
        finally:
            self.is_thinking = False
            if self._spinner_timer:
                self._spinner_timer.stop()
                self._spinner_timer = None
            self.query_one("#thinking-indicator", Static).update("")


def main():
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
