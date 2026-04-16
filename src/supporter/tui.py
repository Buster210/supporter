from __future__ import annotations

import asyncio
import re
import time
from typing import Any, ClassVar

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.style import Style
from rich.styled import Styled
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.timer import Timer
from textual.widgets import Input, Label, Static

from . import ChatAgent, CrewAgent, get_provider
from .config import config
from .llm_types import DEFAULT_SYSTEM_INSTRUCTION
from .logger import init_logger, logger

THEME = {
    "background": "#121212",
    "bubble_bg": "#1e1e1e",
    "header_teal": "#00ffcc",
    "magenta": "#ff06b5",
    "green": "#00ff00",
    "blue": "#0080ff",
    "yellow": "#ffeb3b",
    "meta_gray": "#666666",
}

CRYSTAL_GRADIENT_STOPS: list[tuple[int, int, int]] = [
    (0, 255, 255),  # Cyan
    (0, 255, 180),  # Teal
    (0, 180, 255),  # Sky Blue
    (100, 200, 255),  # Soft Blue
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _apply_crystal_gradient(text: str) -> Text:
    rich_text = Text(justify="center")
    char_count = len(text)
    num_stops = len(CRYSTAL_GRADIENT_STOPS) - 1

    for i, char in enumerate(text):
        progress = i / max(char_count - 1, 1)
        segment = min(int(progress * num_stops), num_stops - 1)
        local_progress = progress * num_stops - segment

        start_rgb = CRYSTAL_GRADIENT_STOPS[segment]
        end_rgb = CRYSTAL_GRADIENT_STOPS[segment + 1]

        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * local_progress)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * local_progress)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * local_progress)

        rich_text.append(char, style=f"bold rgb({r},{g},{b})")
    return rich_text


class SupporterHeader(Static):
    _ART = (
        " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ "
        + "\n ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ "
        + "\n ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
    )

    def render(self) -> Text:
        return _apply_crystal_gradient(self._ART)


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
        self.thoughts = ""
        self.thoughts_expanded = True
        self._message_view: Static | None = None
        self._meta_label: Label | None = None
        self.add_class("right" if role == "user" else "left")

    def compose(self) -> ComposeResult:
        is_user = self.role == "user"
        border_class = "user-border" if is_user else "agent-border"

        bubble = Static("", classes=f"bubble {border_class}", expand=False)
        self._message_view = bubble

        if self.thoughts or self.content:
            self._update_ui_content()

        yield bubble

        meta = Label("", classes="message-meta")
        self._meta_label = meta

        if is_user or self.streaming or (not self.model and self.duration is None):
            meta.display = False
        else:
            meta.update(self._get_meta_text())
        yield meta

    def _get_meta_text(self) -> str:
        model_info = self.model or "Unknown"
        if self.duration is not None:
            model_info += f" in {self.duration:.2f}s"
        if self.agents:
            return f"({', '.join(self.agents)} via {model_info})"
        return f"({model_info})"

    def _preprocess_markdown(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        block_pats = r"(?:[*+-]\s|\d+\.\s|#+|>\s)"
        return re.sub(r"([^\n])\n(" + block_pats + ")", r"\1\n\n\2", text)

    def _should_use_markdown(self, text: str) -> bool:
        syntax_markers = [
            r"\n[*+-]\s",
            r"^[*+-]\s",  # Unordered list
            r"\n\d+\.\s",
            r"^\d+\.\s",  # Ordered list
            r"#+\s",  # Headers
            r"\*\*.*?\*\*",  # Bold
            r"\*.*?\*",  # Italic
            r"`.*?`",  # Inline code
            r"\[.*?\]\(.*?\)",  # Links
            r"\n>\s",
            r"^>\s",  # Quotes
        ]
        return any(re.search(m, text, re.MULTILINE) for m in syntax_markers)

    def _update_ui_content(self) -> None:
        if not self._message_view:
            return

        from rich.console import RenderableType

        elements: list[RenderableType] = []

        if self.thoughts:
            expanded_symbol = "▼" if self.thoughts_expanded else "▶"
            is_thinking = self.streaming and not self.content
            label_text = "Thinking..." if is_thinking else "Thoughts"

            thought_header = Text(
                f"{expanded_symbol} {label_text}",
                style=Style(color=THEME["yellow"]),
            )
            elements.append(thought_header)

            if self.thoughts_expanded:
                elements.append(Text(""))
                clean_thoughts = self._preprocess_markdown(self.thoughts)
                elements.append(
                    Styled(
                        RichMarkdown(clean_thoughts),
                        Style(italic=True, color=THEME["meta_gray"]),
                    )
                )

            if self.content:
                elements.append(Text(""))

        if self.content:
            clean_content = self._preprocess_markdown(self.content)
            if self._should_use_markdown(clean_content):
                elements.append(RichMarkdown(clean_content))
            else:
                elements.append(Text(clean_content))

        if elements:
            self._message_view.update(Group(*elements))

    def append_token(self, token: str, is_thought: bool = False) -> None:
        if is_thought:
            self.thoughts += token
        else:
            if self.thoughts and not self.content:
                if self.thoughts_expanded and self._message_view:
                    self._message_view.styles.min_width = self._message_view.size.width
                self.thoughts_expanded = False
            self.content += token

        self._update_ui_content()

    def finalize(
        self,
        model: str | None = None,
        duration: float | None = None,
        agents: list[str] | None = None,
    ) -> None:
        self.model = model or self.model
        self.duration = duration or self.duration
        self.agents = agents or self.agents
        self.streaming = False

        if self._meta_label:
            self._meta_label.update(self._get_meta_text())
            self._meta_label.display = True

        if self._message_view:
            final_width = self._message_view.size.width
            if final_width > 0:
                self._message_view.styles.width = final_width
            self._message_view.styles.min_width = None

    def on_click(self, event: Click) -> None:
        if not self.thoughts or self.role == "user":
            return
        self.thoughts_expanded = not self.thoughts_expanded
        self._update_ui_content()


class SupporterApp(App[None]):
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

    .bubble.user-border {{
        border: round {THEME["magenta"]};
        color: {THEME["magenta"]};
    }}

    .bubble.agent-border {{
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

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: ChatAgent | CrewAgent | None = None
        self.active_queries = 0
        self._spinner_timer: Timer | None = None
        self._spinner_idx: int = 0
        self.crew_mode = False
        self.live_mode = True
        self.is_activating_mode = False
        self.status_label = "Thinking"
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self.current_active_agent: str = ""

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
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

        if (
            self.agent
            and hasattr(self.agent, "provider")
            and hasattr(self.agent.provider, "close")
        ):
            await self.agent.provider.close()

    async def _setup_agent(
        self, use_crew: bool = False, use_live: bool = False
    ) -> None:
        provider = get_provider(live=use_live)
        if use_crew:
            self.agent = CrewAgent(
                provider=provider, status_callback=self._on_agent_active
            )
            logger.info("Switched to multi-agent crew mode")
            return

        self.agent = ChatAgent(
            provider,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            use_search=True,
            use_code_execution=True,
        )
        logger.info(f"Switched to standard chat agent (Live: {use_live})")

        if use_live and hasattr(provider, "_ensure_session"):
            task = asyncio.create_task(provider._ensure_session())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader()
            with ChatContainer(id="chat-view"):
                pass
            yield Static("", id="thinking-indicator", markup=False)
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
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
            indicator = self.query_one("#thinking-indicator", Static)
            indicator.update("")
            indicator.refresh()

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * (self._spinner_idx % 4)
        self._spinner_idx += 1

        if self.is_activating_mode:
            status = f"Activating Mode{dots}"
        else:
            label = ""
            if self.crew_mode and self.current_active_agent:
                label = f"[{self.current_active_agent}]"
            elif self.live_mode:
                label = "[LIVE]"
            elif self.crew_mode:
                label = "[AGENT]"

            status = f"{frame} {label} {self.status_label}{dots}"
        self.query_one("#thinking-indicator", Static).update(status)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        input_widget.focus()

        if user_text.startswith("/"):
            await self._handle_command(user_text.lower())
            return

        chat_view = self.query_one("#chat-view", Vertical)
        chat_view.mount(MessageBubble(role="user", content=user_text))
        chat_view.scroll_end()

        self.run_worker(self._process_message_cycle(user_text, mount_user=False))

    async def _handle_command(self, command: str) -> None:
        from collections.abc import Callable

        command_map: dict[str, Callable[[], Any]] = {
            "/exit": self.exit,
            "/clear": self.action_clear_screen,
            "/crew": lambda: self._toggle_mode(crew=True),
            "/live": lambda: self._toggle_mode(live=True),
        }

        handler = command_map.get(command)
        if handler:
            result = handler()
            if asyncio.iscoroutine(result):
                await result

    async def _toggle_mode(self, crew: bool = False, live: bool = False) -> None:
        if crew:
            self.crew_mode = not self.crew_mode
            if self.crew_mode:
                self.live_mode = False
        if live:
            self.live_mode = not self.live_mode
            if self.live_mode:
                self.crew_mode = False

        status_msg = "ENABLED" if (self.crew_mode or self.live_mode) else "DISABLED"
        mode_label = (
            "Multi-Agent Crew"
            if crew
            else (
                f"Single Agent Live "
                f"({config.gemini_live_model if self.live_mode else 'Standard'})"
            )
        )

        self._start_thinking()
        self.is_activating_mode = True
        try:
            await self._setup_agent(use_crew=self.crew_mode, use_live=self.live_mode)
            chat_view = self.query_one("#chat-view", Vertical)
            await chat_view.mount(
                MessageBubble(role="agent", content=f"{mode_label} {status_msg}")
            )
            chat_view.scroll_end()
        finally:
            self.is_activating_mode = False
            self._stop_thinking()

    async def _process_message_cycle(self, text: str, mount_user: bool = True) -> None:
        chat_view = self.query_one("#chat-view", Vertical)
        if mount_user:
            await chat_view.mount(MessageBubble(role="user", content=text))
            chat_view.scroll_end()
            self.query_one("#user-input").focus()

        self.current_active_agent = ""
        self.status_label = "Thinking"
        self._start_thinking()
        start_time = time.perf_counter()

        try:
            agent = self.agent
            if not agent:
                raise RuntimeError("Agent is not initialized")

            if isinstance(agent, CrewAgent):
                await self._process_crew_execution(text, chat_view, start_time)
            else:
                await self._process_streaming_execution(
                    text, chat_view, start_time, agent
                )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            await chat_view.mount(MessageBubble(role="agent", content=f"Error: {e}"))
            chat_view.scroll_end()
        finally:
            self._stop_thinking()

    async def _process_crew_execution(
        self, text: str, chat_view: Vertical, start_time: float
    ) -> None:
        if not isinstance(self.agent, CrewAgent):
            return

        response = await self.agent.execute(text)
        agent_roles = response.usage.get("agents") if response.usage else None

        await chat_view.mount(
            MessageBubble(
                role="agent",
                content=response.text,
                model=response.model,
                duration=time.perf_counter() - start_time,
                agents=agent_roles,
            )
        )
        chat_view.scroll_end()

    async def _process_streaming_execution(
        self, text: str, chat_view: Vertical, start_time: float, agent: ChatAgent
    ) -> None:
        bubble = None
        is_first_chunk = True
        actual_model = None

        async for chunk in agent.execute_stream(text):
            if chunk.is_last:
                self.status_label = "Thinking"
                self._tick_spinner()

            if chunk.model:
                actual_model = chunk.model

            if is_first_chunk:
                if not chunk.text.strip() and not chunk.is_thought:
                    continue

                is_first_chunk = False
                self.status_label = "Streaming"
                bubble = MessageBubble(role="agent", content="", streaming=True)

                if chunk.is_thought:
                    bubble.thoughts = chunk.text
                else:
                    bubble.content = chunk.text

                await chat_view.mount(bubble)
                chat_view.scroll_end()
            else:
                if bubble:
                    bubble.append_token(chunk.text, is_thought=chunk.is_thought)
                    chat_view.scroll_end()

        if bubble:
            bubble.finalize(
                model=actual_model, duration=time.perf_counter() - start_time
            )
            chat_view.scroll_end()


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
