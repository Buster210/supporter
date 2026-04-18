from __future__ import annotations

import asyncio
import re
import time
from typing import Any, ClassVar

from rich.markdown import Markdown as RichMarkdown
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
    "meta_gray": "#999999",
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


class SectionHeader(Static):
    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def update_label(self, label: str, is_collapsed: bool, is_collapsible: bool):
        self.label = label
        self.set_class(is_collapsed, "collapsed")
        self.set_class(is_collapsible, "collapsible")

        hint = ""
        if is_collapsible:
            hint = f" [{THEME['meta_gray']} italic](Click to expand/collapse)[/]"
        self.update(f"{self.label}{hint}")

    def on_click(self, event: Click) -> None:
        parent = self.parent
        while parent and not isinstance(parent, MessageBubble):
            parent = parent.parent
        if parent and parent.collapsible:
            event.stop()
            parent.toggle_section(self.id)


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
        self.tool_calls: list[dict[str, Any]] = []
        self.collapsed = False
        self.collapsible = False
        self.thoughts_collapsed = False
        self.tools_collapsed = False
        self._message_view: Static | None = None
        self._meta_label: Label | None = None
        self.add_class("right" if role == "user" else "left")

    def compose(self) -> ComposeResult:
        is_user = self.role == "user"
        border_class = "user-border" if is_user else "agent-border"

        with Vertical(classes=f"bubble {border_class}", id="bubble-container"):
            self._thought_header = SectionHeader(
                "", classes="section-header", id="thoughts-header"
            )
            self._thought_view = Static(
                "", classes="section-content", id="thoughts-content"
            )
            self._thought_header.display = False
            self._thought_view.display = False

            self._tool_header = SectionHeader(
                "", classes="section-header", id="tools-header"
            )
            self._tool_view = Static("", classes="section-content", id="tools-content")
            self._tool_header.display = False
            self._tool_view.display = False

            self._message_view = Static(self.content, id="main-content")

            yield self._thought_header
            yield self._thought_view
            yield self._tool_header
            yield self._tool_view
            yield self._message_view

            if not is_user:
                self._update_ui_content()

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
        return text.strip()

    def _should_use_markdown(self, text: str) -> bool:
        syntax_markers = [
            r"\n[*+-]\s",
            r"[*+-]\s",
            r"\n\d+\.\s",
            r"\d+\.\s",
            r"#+\s",
            r"\*\*.*?\*\*",
            r"\*.*?\*",
            r"`.*?`",
            r"\[.*?\]\(.*?\)",
            r"\n>\s",
            r">\s",
        ]
        return any(re.search(m, text, re.MULTILINE) for m in syntax_markers)

    def toggle_section(self, section_id: str | None) -> None:
        if section_id == "thoughts-header":
            self.thoughts_collapsed = not self.thoughts_collapsed
        elif section_id == "tools-header":
            self.tools_collapsed = not self.tools_collapsed
        self._update_ui_content()

    def _update_ui_content(self) -> None:
        if not self._message_view:
            return

        self.set_class(self.collapsed, "collapsed-bubble")

        if self.collapsed:
            self._thought_header.display = False
            self._thought_view.display = False
            self._tool_header.display = False
            self._tool_view.display = False

            summary = self.content.split("\n")[0][:50]
            if len(self.content) > 50 or "\n" in self.content:
                summary += "..."

            hint = f"[{THEME['meta_gray']} italic](Click to expand/collapse)[/] "
            self._message_view.update(f"{hint}{summary}")
            self._message_view.display = True
            if self._meta_label:
                self._meta_label.display = False
            return

        if self.thoughts and not self.collapsed:
            self._thought_header.display = True
            self._thought_view.display = (
                not self.thoughts_collapsed if self.collapsible else True
            )

            is_thinking = self.streaming and not self.content
            label_text = "Thinking" if is_thinking else "Thoughts"
            self._thought_header.update_label(
                label_text, self.thoughts_collapsed, self.collapsible
            )

            if not self.collapsible or not self.thoughts_collapsed:
                clean_thoughts = self._preprocess_markdown(self.thoughts)
                self._thought_view.update(RichMarkdown(clean_thoughts))
        else:
            self._thought_header.display = False
            self._thought_view.display = False

        if self.tool_calls and not self.collapsed:
            self._tool_header.display = True
            self._tool_view.display = (
                not self.tools_collapsed if self.collapsible else True
            )
            self._tool_header.update_label(
                "Tools Used", self.tools_collapsed, self.collapsible
            )
            self._tool_header.set_class(bool(self.thoughts), "section-gap")

            if not self.collapsible or not self.tools_collapsed:
                tool_lines = []
                for tc in self.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    arg_str = ""
                    if args:
                        # Format args nicely: key=val, key2=val2
                        items = []
                        for k, v in args.items():
                            val = str(v)
                            if len(val) > 40:
                                val = val[:37] + "..."
                            items.append(f"{k}={val}")
                        arg_str = f"({', '.join(items)})"
                    tool_lines.append(f"• {name}{arg_str}")

                self._tool_view.update("\n".join(tool_lines))
        else:
            self._tool_header.display = False
            self._tool_view.display = False

        if self.content:
            clean_content = self._preprocess_markdown(self.content)

            display_text = clean_content
            is_active_turn = False
            if hasattr(self.app, "active_turn"):
                active = self.app.active_turn
                if self == active.user_bubble or self in active.agent_bubbles:
                    is_active_turn = True

            if self.role == "user" and not self.streaming and not is_active_turn:
                hint = f"[{THEME['meta_gray']} italic](Click to expand/collapse)[/] "
                display_text = f"{hint}{clean_content}"

            if self._should_use_markdown(clean_content):
                self._message_view.update(RichMarkdown(clean_content))

            else:
                self._message_view.update(display_text)
            self._message_view.display = True
            self._message_view.set_class(
                bool(self.thoughts or self.tool_calls), "section-gap"
            )
        else:
            self._message_view.display = False

    def append_token(self, token: str, is_thought: bool = False) -> None:
        if is_thought:
            self.thoughts += token
        else:
            self.content += token

        self._update_ui_content()

    def add_tool_call(
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> None:
        call_entry = {"name": tool_name, "args": tool_args or {}}
        if call_entry not in self.tool_calls:
            self.tool_calls.append(call_entry)
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

        self.agents = agents or self.agents
        self.streaming = False

        if self._meta_label:
            self._meta_label.update(self._get_meta_text())
            self._meta_label.display = not self.collapsed

        if self._message_view:
            final_width = self._message_view.size.width
            if final_width > 0:
                self._message_view.styles.width = final_width

    def on_click(self, event: Click) -> None:
        if self.collapsed:
            parent = self.parent
            if isinstance(parent, ChatTurn):
                parent.toggle_collapse()
            else:
                self.collapsed = False
                self._update_ui_content()
                if self._meta_label:
                    self._meta_label.display = True
            event.stop()
            return


class ChatTurn(Vertical):
    def __init__(self, user_bubble: MessageBubble):
        super().__init__(classes="chat-turn")
        self.user_bubble = user_bubble
        self.agent_bubbles: list[MessageBubble] = []
        self.collapsed = False

    def compose(self) -> ComposeResult:
        yield self.user_bubble

    def collapse(self) -> None:
        if not self.collapsed:
            self.toggle_collapse()

    def toggle_collapse(self) -> None:
        if hasattr(self.app, "active_turn") and self == self.app.active_turn:
            return

        self.collapsed = not self.collapsed
        self.user_bubble.collapsed = self.collapsed
        self.user_bubble._update_ui_content()
        for bubble in self.agent_bubbles:
            bubble.collapsed = self.collapsed
            bubble.display = not self.collapsed
            if self.collapsed:
                bubble.collapsible = True
                bubble.thoughts_collapsed = True
                bubble.tools_collapsed = True
            if not self.collapsed:
                bubble._update_ui_content()

    async def mount_bubble(self, bubble: MessageBubble) -> None:
        bubble.collapsed = self.collapsed
        if self.collapsed:
            bubble.thoughts_collapsed = True
            bubble.tools_collapsed = True
        self.agent_bubbles.append(bubble)
        await self.mount(bubble)
        if self.collapsed:
            bubble.display = False
        else:
            bubble.display = True
            bubble._update_ui_content()

    def on_click(self, event: Click) -> None:
        if hasattr(self.app, "active_turn") and self == self.app.active_turn:
            return
        self.toggle_collapse()
        event.stop()


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

    .chat-turn {{
        width: 100%;
        height: auto;
        margin-top: 1;
        padding: 0;
        layout: vertical;
    }}

    MessageBubble.left  {{ align-horizontal: left; }}
    MessageBubble.right {{ align-horizontal: right; }}

    .bubble {{
        width: 100%;
        height: auto;
        padding: 0 2;
        margin: 0;
        background: transparent;
        border: none;
        layout: vertical;
    }}

    .bubble.user-border {{
        color: {THEME["magenta"]};
        content-align-horizontal: right;
    }}

    .bubble.user-border #main-content {{
        text-align: right;
    }}

    .bubble.agent-border {{
        color: {THEME["header_teal"]};
        content-align-horizontal: left;
    }}

    .section-header {{
        color: {THEME["yellow"]};
        text-style: bold;
        width: 1fr;
    }}

    .section-header.collapsed {{
        text-style: dim bold;
    }}

    .section-header.collapsible {{
    }}

    .section-header.collapsible:hover {{
        background: {THEME["bubble_bg"]};
    }}

    .section-content {{
        color: {THEME["meta_gray"]};
        text-style: italic;
        margin-left: 2;
        margin-top: 1;
        width: 100%;
    }}

    #main-content {{
        width: 100%;
    }}

    .collapsed-bubble #main-content {{
        color: {THEME["meta_gray"]};
        text-style: dim;
    }}

    .section-gap {{
        margin-top: 1;
    }}

    .message-meta {{
        color: {THEME["yellow"]};
        text-style: dim italic;
        margin: 0;
        width: 1fr;
    }}

    .right .message-meta {{
        text-align: right;
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

    #mode-indicator {{
        color: {THEME["yellow"]};
        text-style: bold;
        width: auto;
        margin-right: 1;
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
            indicator.display = False
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
                label = f"[{self.current_active_agent}] "

            status = f"{frame} {label}{self.status_label}{dots}"

        indicator = self.query_one("#thinking-indicator", Static)
        indicator.update(status)
        indicator.display = True

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

            mode_text = (
                "CREW" if self.crew_mode else ("LIVE" if self.live_mode else "SINGLE")
            )
            indicator = self.query_one("#mode-indicator", Label)
            indicator.markup = False
            indicator.update(f"[{mode_text}]")

            target = (
                self.active_turn
                if hasattr(self, "active_turn")
                else self.query_one("#chat-view", Vertical)
            )
            await target.mount(
                MessageBubble(role="agent", content=f"{mode_label} {status_msg}")
            )
        finally:
            self.is_activating_mode = False
            self._stop_thinking()

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
        if not isinstance(self.agent, CrewAgent):
            return

        response = await self.agent.execute(text)
        agent_roles = response.usage.get("agents") if response.usage else None

        bubble = MessageBubble(
            role="agent",
            content=response.text,
            model=response.model,
            duration=time.perf_counter() - start_time,
            agents=agent_roles,
        )

        if isinstance(target, ChatTurn):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)

    async def _process_streaming_execution(
        self, text: str, target: Vertical, start_time: float, agent: ChatAgent
    ) -> None:
        bubble = None
        is_first_chunk = True
        actual_model = None

        async for chunk in agent.execute_stream(text):
            if chunk.is_tool_call:
                if "google_search" in (chunk.tool_name or "").lower():
                    self.status_label = "Searching"
                else:
                    self.status_label = f"Using {chunk.tool_name}"
                self._tick_spinner()

                if is_first_chunk:
                    is_first_chunk = False
                    bubble = MessageBubble(role="agent", content="", streaming=True)
                    if isinstance(target, ChatTurn):
                        await target.mount_bubble(bubble)
                    else:
                        await target.mount(bubble)

                if bubble:
                    bubble.add_tool_call(
                        chunk.tool_name or "unknown_tool", chunk.tool_args
                    )
                continue

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

                if isinstance(target, ChatTurn):
                    await target.mount_bubble(bubble)
                else:
                    await target.mount(bubble)
            else:
                if bubble:
                    if self.status_label == "Searching" or "Using" in self.status_label:
                        self.status_label = "Streaming"
                    bubble.append_token(chunk.text, is_thought=chunk.is_thought)

        if bubble:
            bubble.finalize(
                model=actual_model, duration=time.perf_counter() - start_time
            )


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
