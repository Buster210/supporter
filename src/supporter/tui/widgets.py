from __future__ import annotations

import re
from typing import Any

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.widgets import Label, Static

from .theme import THEME, apply_crystal_gradient


class SupporterHeader(Static):
    _ART = (
        " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ "
        + "\n ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ "
        + "\n ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
    )

    def render(self) -> Text:
        return apply_crystal_gradient(self._ART)


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
    def __init__(self, label: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.label = label

    def update_label(
        self, label: str, is_collapsed: bool, is_collapsible: bool
    ) -> None:
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
        if isinstance(parent, MessageBubble) and parent.collapsible:
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
