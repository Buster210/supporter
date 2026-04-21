from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from .theme import THEME, apply_crystal_gradient


class ConfirmationModal(ModalScreen[bool]):
    def __init__(self, path: str, content: str):
        super().__init__()
        self.path = path
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            filename = Path(self.path).name
            yield Label(f"Write {filename}?", id="modal-header")
            with ScrollableContainer(id="modal-content"):
                yield Static(
                    Syntax(
                        self.content,
                        "diff",
                        theme="monokai",
                        background_color="default",
                        word_wrap=True,
                    ),
                    expand=True,
                )
            with Horizontal(id="modal-buttons"):
                yield Button("Allow", id="allow")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        max_line = max((len(line) for line in self.content.splitlines()), default=40)
        padding = 6
        width = min(int((max_line + padding) * 1.3), int(self.app.size.width * 0.9))
        self.query_one("#modal-container").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class SupporterHeader(Static):
    _ART = (
        " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ \n"
        " ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ \n"
        " ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
    )

    def render(self) -> Text:
        return apply_crystal_gradient(self._ART)


class ChatContainer(Vertical):
    SCROLL_AMOUNT = 5

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        for _ in range(self.SCROLL_AMOUNT):
            self.scroll_down()
        event.prevent_default()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        for _ in range(self.SCROLL_AMOUNT):
            self.scroll_up()
        event.prevent_default()


class SectionHeader(Static):
    def __init__(self, label: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label = label

    def update_label(
        self, label: str, is_collapsed: bool, is_collapsible: bool
    ) -> None:
        self.label = label
        self.set_class(is_collapsed, "collapsed")
        self.set_class(is_collapsible, "collapsible")

        hint = (
            f" [{THEME['meta_gray']} italic](Click to expand/collapse)[/]"
            if is_collapsible
            else ""
        )
        self.update(f"{self.label}{hint}")

    def on_click(self, event: Click) -> None:
        parent = self._find_bubble_parent()
        if parent and parent.collapsible:
            event.stop()
            parent.toggle_section(self.id)

    def _find_bubble_parent(self) -> MessageBubble | None:
        parent = self.parent
        while parent and not isinstance(parent, MessageBubble):
            parent = parent.parent
        return cast(MessageBubble, parent) if parent else None


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

            self._meta_label = Label("", classes="message-meta")
            if is_user or self.streaming or (not self.model and self.duration is None):
                self._meta_label.display = False
            else:
                self._meta_label.update(self._get_meta_text())
            yield self._meta_label

    def _get_meta_text(self) -> str:
        model_info = self.model or "Unknown"
        if self.duration is not None:
            model_info += f" in {self.duration:.2f}s"
        if self.agents:
            return f"({', '.join(self.agents)} via {model_info})"
        return f"({model_info})"

    def _should_use_markdown(self, text: str) -> bool:
        syntax_markers = [
            r"[*+-]\s",
            r"\d+\.\s",
            r"#+\s",
            r"\*\*.*?\*\*",
            r"\*.*?\*",
            r"`.*?`",
            r"\[.*?\]\(.*?\)",
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
            self._render_collapsed()
            return

        self._render_expanded()

    def _render_collapsed(self) -> None:
        if not self._message_view:
            return

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

    def _render_expanded(self) -> None:
        self._render_thoughts()
        self._render_tools()
        self._render_main_content()

    def _render_thoughts(self) -> None:
        if not self.thoughts:
            self._thought_header.display = False
            self._thought_view.display = False
            return

        self._thought_header.display = True
        self._thought_view.display = (
            not self.thoughts_collapsed if self.collapsible else True
        )

        is_thinking = self.streaming and not self.content
        label = "Thinking" if is_thinking else "Thoughts"
        self._thought_header.update_label(
            label, self.thoughts_collapsed, self.collapsible
        )

        if not self.collapsible or not self.thoughts_collapsed:
            self._thought_view.update(RichMarkdown(self.thoughts.strip()))

    def _render_tools(self) -> None:
        if not self.tool_calls:
            self._tool_header.display = False
            self._tool_view.display = False
            return

        self._tool_header.display = True
        self._tool_view.display = not self.tools_collapsed if self.collapsible else True
        self._tool_header.update_label(
            "Tools Used", self.tools_collapsed, self.collapsible
        )
        self._tool_header.set_class(bool(self.thoughts), "section-gap")

        if not self.collapsible or not self.tools_collapsed:
            tool_lines = []
            for tc in self.tool_calls:
                name, args = tc["name"], tc["args"]
                arg_str = ""
                if args:
                    items = [
                        f"{k}={str(v)[:37]}..." if len(str(v)) > 40 else f"{k}={v}"
                        for k, v in args.items()
                    ]
                    arg_str = f"({', '.join(items)})"
                tool_lines.append(f"• {name}{arg_str}")
            self._tool_view.update("\n".join(tool_lines))

    def _render_main_content(self) -> None:
        if not self.content or not self._message_view:
            if self._message_view:
                self._message_view.display = False
            return

        clean_content = self.content.strip()
        display_text = clean_content

        is_active = False
        if hasattr(self.app, "active_turn"):
            active = self.app.active_turn
            is_active = self == active.user_bubble or self in active.agent_bubbles

        if self.role == "user" and not self.streaming and not is_active:
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

        if self._meta_label:
            self._meta_label.update(self._get_meta_text())
            self._meta_label.display = not self.collapsed

        if self._message_view and self._message_view.size.width > 0:
            self._message_view.styles.width = self._message_view.size.width

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


class ChatTurn(Vertical):
    def __init__(self, user_bubble: MessageBubble):
        super().__init__(classes="chat-turn")
        self.user_bubble = user_bubble
        self.agent_bubbles: list[MessageBubble] = []
        self.collapsed = False

    def compose(self) -> ComposeResult:
        yield self.user_bubble

    def toggle_collapse(self) -> None:
        if self.collapsed:
            self.expand_turn()
        else:
            self.collapse_turn()

    def collapse_turn(self) -> None:
        if hasattr(self.app, "active_turn") and self == self.app.active_turn:
            return
        if self.collapsed:
            return

        self.collapsed = True
        self.set_class(True, "collapsed")
        self.user_bubble.collapsed = True
        self.user_bubble._update_ui_content()
        for bubble in self.agent_bubbles:
            bubble.collapsed = True

    def expand_turn(self) -> None:
        if not self.collapsed:
            return

        self.collapsed = False
        self.set_class(False, "collapsed")
        self.user_bubble.collapsed = False
        self.user_bubble._update_ui_content()
        for bubble in self.agent_bubbles:
            bubble.collapsed = False
            bubble._update_ui_content()

    async def mount_bubble(self, bubble: MessageBubble) -> None:
        bubble.collapsed = self.collapsed
        self.agent_bubbles.append(bubble)
        await self.mount(bubble)
        if not self.collapsed:
            bubble._update_ui_content()

    def on_click(self, event: Click) -> None:
        if hasattr(self.app, "active_turn") and self.app.active_turn is self:
            event.stop()
            return
        self.toggle_collapse()
        event.stop()
