from __future__ import annotations

import time
from datetime import datetime as datetime_

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import MouseScrollDown, MouseScrollUp
from textual.widgets import Input, Label, Static

from . import ChatAgent, get_provider
from .logger import init_logger, logger

THEME = {
    "background": "#212121",
    "bubble_bg": "#2a2a2a",
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

        r1, g1, b1 = CRISTAL_STOPS[seg]
        r2, g2, b2 = CRISTAL_STOPS[seg + 1]

        r = int(r1 + (r2 - r1) * local_t)
        g = int(g1 + (g2 - g1) * local_t)
        b = int(b1 + (b2 - b1) * local_t)

        rich_text.append(ch, style=f"bold rgb({r},{g},{b})")

    return rich_text


class SupporterHeader(Static):
    _ART = (
        r" █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ " + "\n"
        r" ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ " + "\n"
        r" ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
    )

    def render(self) -> Text:
        return _apply_cristal_gradient(self._ART)


class ChatContainer(Vertical):
    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self.scroll_down()
        event.prevent_default()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self.scroll_up()
        event.prevent_default()


class MessageBubble(Vertical):
    def __init__(
        self,
        role: str,
        content: str,
        model: str | None = None,
        duration: float | None = None,
    ):
        super().__init__()
        self.role = role
        self.content = content
        self.model = model
        self.duration = duration
        self.add_class("right" if role == "user" else "left")

    def compose(self) -> ComposeResult:
        is_user = self.role == "user"
        border_color = "green" if is_user else "blue"
        yield Static(self.content, classes=f"bubble {border_color}", expand=False)

        if not is_user and (self.model or self.duration is not None):
            meta_text = f"({self.model}"
            if self.duration is not None:
                meta_text += f" in {self.duration:.2f}s"
            meta_text += ")"
            yield Label(meta_text, classes="message-meta")


class SupporterApp(App):
    CSS = f"""
    Screen {{
        background: {THEME["background"]};
        width: 100%;
        height: 100%;
        padding: 0;
        margin: 0;
    }}


    SupporterHeader {{
        height: 3;
        content-align: center middle;
        background: transparent;
        margin-top: 0;
        color: {THEME["header_teal"]};
        text-style: bold;
        margin-bottom: 0;
    }}

    #chat-view {{
        width: 100%;
        height: 1fr;
        padding: 0;
        margin: 0;
        background: transparent;
        layout: vertical;
        overflow-y: hidden;
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
        max-width: 95%;
        height: auto;
        padding: 0 1;
        margin: 0;
        background: {THEME["bubble_bg"]};
        border: solid #444;
    }}

    .bubble.green {{
        border: solid {THEME["magenta"]};
        color: {THEME["magenta"]};
    }}

    .bubble.blue {{
        border: solid {THEME["header_teal"]};
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
    """

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

    async def on_mount(self) -> None:
        init_logger()
        logger.info("Starting Supporter TUI")

        try:
            await self._setup_agent()
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Failed to initialize agent: {e}"
            logger.error(msg)
            self.notify(msg, severity="error")

    async def _setup_agent(self) -> None:
        provider = get_provider()

        tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_current_time",
                        "description": "Get the current system time",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {},
                            "required": [],
                        },
                    }
                ]
            }
        ]

        def get_current_time() -> dict:
            return {"time": datetime_.now().strftime("%I:%M:%S %p")}

        registry = {"get_current_time": get_current_time}

        self.agent = ChatAgent(
            provider,
            {
                "tools": tools,
                "registry": registry,
                "system_instruction": "You are a helpful assistant. Be concise and professional.",
            },
        )
        logger.info("Agent successfully initialized")

    def compose(self) -> ComposeResult:
        yield SupporterHeader()
        with ChatContainer(id="chat-view"):
            pass
        yield Static("", id="thinking-indicator")
        with Vertical(id="input-area"), Horizontal(id="prompt-row"):
            yield Label("❯", id="prompt-symbol")

            yield Input(
                placeholder="Type a message... (/exit to quit, /clear to reset)",
                id="user-input",
            )

    def action_clear_screen(self) -> None:
        self.action_clear()

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * (self._spinner_idx % 4)
        self._spinner_idx += 1
        self.query_one("#thinking-indicator", Static).update(f"{frame} Thinking{dots}")

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

    async def _process_message_cycle(self, text: str) -> None:
        chat_view = self.query_one("#chat-view")

        await chat_view.mount(MessageBubble(role="user", content=text))
        chat_view.scroll_end(animate=False)

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
                    duration=duration,
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
