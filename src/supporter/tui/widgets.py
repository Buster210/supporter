from __future__ import annotations

import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..logger import logger
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
        lines = self.content.splitlines()
        max_line = max((len(line) for line in lines), default=40)
        width = min(int((max_line + 6) * 1.3), int(self.app.size.width * 0.9))
        self.query_one("#modal-container").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class BashConfirmationModal(ModalScreen[bool]):
    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Authorize Bash Execution?", id="modal-header")
            with ScrollableContainer(id="modal-content"):
                cmd_str = " ".join(self.command)
                yield Label(
                    f"[bold cyan]Working Dir:[/] {self.cwd}", classes="modal-meta"
                )
                yield Static(
                    Syntax(
                        cmd_str,
                        "bash",
                        theme="monokai",
                        background_color="default",
                        word_wrap=True,
                    ),
                    expand=True,
                )
            with Horizontal(id="modal-buttons"):
                yield Button("Allow", id="allow", variant="success")
                yield Button("Deny", id="cancel", variant="error")

    def on_mount(self) -> None:
        self.query_one("#modal-container").styles.width = min(
            80, int(self.app.size.width * 0.9)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


SUPPORTER_ART = (
    " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ \n"
    " ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ \n"
    " ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
)


class SupporterHeader(Static):
    def render(self) -> Text:
        return apply_crystal_gradient(SUPPORTER_ART)


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
    class ToggleRequest(Message):
        def __init__(self, header: SectionHeader) -> None:
            self.header = header
            super().__init__()

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
        bubble = self._find_bubble_parent()
        if not (bubble and bubble.collapsible):
            self.post_message(self.ToggleRequest(self))
            return

        event.stop()
        bubble.toggle_section(self.id)

    def _find_bubble_parent(self) -> MessageBubble | None:
        current = self.parent
        while current and not isinstance(current, MessageBubble):
            current = current.parent
        return cast(MessageBubble, current)


class MessageBubble(Vertical):
    collapsed = reactive(False)
    is_active = reactive(False)

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
        self.elements: list[dict[str, Any]] = []
        if self.thoughts:
            self.elements.append(
                {"type": "thought", "content": self.thoughts, "collapsed": False}
            )
        if self.content:
            self.elements.append(
                {"type": "content", "content": self.content, "collapsed": False}
            )

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
            with Vertical(id="elements-container") as container:
                self._elements_container = container
                for i, el in enumerate(self.elements):
                    yield from self._create_widgets_for_element(i, el)

            self._message_view = Static(self.content, id="main-content")
            self._message_view.display = False
            yield self._message_view

            self._meta_label = Label("", classes="message-meta")
            if is_user or self.streaming or (not self.model and self.duration is None):
                self._meta_label.display = False
            else:
                self._meta_label.update(self._get_meta_text())
            yield self._meta_label

    def on_mount(self) -> None:
        self._update_ui_content()

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

    def on_section_header_toggle_request(
        self, event: SectionHeader.ToggleRequest
    ) -> None:
        self.toggle_section(event.header.id)

    def toggle_section(self, section_id: str | None) -> None:
        try:
            container = self.query_one("#elements-container")
            idx = -1
            if section_id:
                try:
                    header = self.query_one(f"#{section_id}")
                    idx = container.children.index(header)
                except Exception:
                    logger.debug("Header not found for section toggle")

            if idx == -1:
                return

            current_child_idx = 0
            for el in self.elements:
                if el["type"] in ("thought", "tool_calls"):
                    if current_child_idx == idx:
                        el["collapsed"] = not el["collapsed"]
                        break
                    current_child_idx += 2
                else:
                    current_child_idx += 1
        except (ValueError, Exception):
            logger.debug("Failed to toggle section")

        self._update_ui_content()

    def watch_collapsed(self, value: bool) -> None:
        if self._meta_label:
            self._meta_label.display = not value
        self._update_ui_content()

    def watch_is_active(self, value: bool) -> None:
        self._update_ui_content()

    def _update_ui_content(self) -> None:
        if not self.is_attached or not hasattr(self, "_message_view"):
            return

        self.set_class(self.collapsed, "collapsed-bubble")
        if self.collapsed:
            self._render_collapsed()
        else:
            self._render_expanded()

    def _render_collapsed(self) -> None:
        if not self._message_view:
            return

        self._elements_container.display = False

        summary = self.content.split("\n")[0][:50]
        if len(self.content) > 50 or "\n" in self.content:
            summary += "..."

        hint = f"[{THEME['meta_gray']} italic](Click to expand/collapse)[/] "
        self._message_view.update(f"{hint}{summary}")
        self._message_view.display = True
        if self._meta_label:
            self._meta_label.display = False

    def _render_expanded(self) -> None:
        try:
            container = self.query_one("#elements-container")
            container.display = True
        except Exception:
            return

        if self._message_view:
            self._message_view.display = False
        self._sync_elements()

    def _sync_elements(self) -> None:
        try:
            container = self.query_one("#elements-container")
        except Exception:
            return

        if not container.is_attached:
            return

        current_widgets = container.query("*")

        expected_widget_count = 0
        for el in self.elements:
            expected_widget_count += 2 if el["type"] in ("thought", "tool_calls") else 1

        if len(current_widgets) < expected_widget_count:
            current_element_idx = 0
            w_idx = 0
            while w_idx < len(current_widgets) and current_element_idx < len(
                self.elements
            ):
                el = self.elements[current_element_idx]
                w_idx += 2 if el["type"] in ("thought", "tool_calls") else 1
                current_element_idx += 1

            new_widgets = []
            for i in range(current_element_idx, len(self.elements)):
                el_widgets = self._create_widgets_for_element(i, self.elements[i])
                new_widgets.extend(el_widgets)

            if new_widgets:
                container.mount(*new_widgets)

            current_widgets = container.query("*")
        elif len(current_widgets) > expected_widget_count:
            container.remove_children()
            new_widgets = []
            for i, el in enumerate(self.elements):
                new_widgets.extend(self._create_widgets_for_element(i, el))
            container.mount(*new_widgets)
            current_widgets = container.query("*")

        w_idx = 0
        for i, el in enumerate(self.elements):
            if el["type"] in ("thought", "tool_calls"):
                if w_idx + 1 >= len(current_widgets):
                    break
                header = cast(SectionHeader, current_widgets[w_idx])
                view = cast(Static, current_widgets[w_idx + 1])

                if el["type"] == "thought":
                    is_thinking = (
                        self.streaming
                        and i == len(self.elements) - 1
                        and not self.content
                    )
                    label = "Thinking" if is_thinking else "Thoughts"
                    header.update_label(label, el["collapsed"], self.collapsible)
                    view.update(RichMarkdown(el["content"]))
                else:
                    header.update_label("Tools Used", el["collapsed"], self.collapsible)
                    view.update(self._format_tool_calls(el["calls"]))
                w_idx += 2
            else:
                if w_idx >= len(current_widgets):
                    break
                view = cast(Static, current_widgets[w_idx])
                content = el["content"]
                if self._should_use_markdown(content):
                    view.update(RichMarkdown(content))
                else:
                    view.update(content)
                w_idx += 1

    def _create_widgets_for_element(self, idx: int, el: dict[str, Any]) -> list[Static]:
        if el["type"] == "thought":
            is_thinking = self.streaming and not self.content
            label = "Thinking" if is_thinking else "Thoughts"
            header = SectionHeader("", classes="section-header")
            header.update_label(label, el["collapsed"], self.collapsible)
            header.set_class(idx > 0, "section-gap")
            view = Static(
                RichMarkdown(el["content"].strip()), classes="section-content"
            )
            view.display = not el["collapsed"] if self.collapsible else True
            return [header, view]
        if el["type"] == "tool_calls":
            header = SectionHeader("", classes="section-header")
            header.update_label("Tools Used", el["collapsed"], self.collapsible)
            view = Static(
                self._format_tool_calls(el["calls"]), classes="section-content"
            )
            view.display = not el["collapsed"] if self.collapsible else True
            header.set_class(idx > 0, "section-gap")
            return [header, view]

        content = el["content"].strip()
        if self._should_use_markdown(content):
            view = Static(RichMarkdown(content), classes="main-content")
        else:
            view = Static(content, classes="main-content")
        view.set_class(idx > 0, "section-gap")
        return [view]

    def _format_tool_calls(self, calls: list[dict[str, Any]]) -> str:
        lines = []
        for tc in calls:
            name, args = tc["name"], tc["args"]
            arg_str = ""
            if args:
                items = [
                    f"{k}={str(v)[:37]}..." if len(str(v)) > 40 else f"{k}={v}"
                    for k, v in args.items()
                ]
                arg_str = f"({', '.join(items)})"
            lines.append(f"• {name}{arg_str}")
        return "\n".join(lines)

    def append_token(self, token: str, is_thought: bool = False) -> None:
        if is_thought:
            self.thoughts += token
            etype = "thought"
        else:
            self.content += token
            etype = "content"

        if not self.elements or self.elements[-1]["type"] != etype:
            self.elements.append({"type": etype, "content": token, "collapsed": False})
        else:
            self.elements[-1]["content"] += token

        self._update_ui_content()

    def add_tool_call(
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> None:
        entry = {"name": tool_name, "args": tool_args or {}}
        if entry not in self.tool_calls:
            self.tool_calls.append(entry)

            if not self.elements or self.elements[-1]["type"] != "tool_calls":
                self.elements.append(
                    {"type": "tool_calls", "calls": [entry], "collapsed": False}
                )
            else:
                self.elements[-1]["calls"].append(entry)

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

        self._update_ui_content()

    def on_click(self, event: Click) -> None:
        if not self.collapsed:
            return

        if isinstance(self.parent, ChatTurn):
            self.parent.toggle_collapse()
        else:
            self.collapsed = False
            self._update_ui_content()
            if self._meta_label:
                self._meta_label.display = True

        event.stop()


class ChatTurn(Vertical):
    collapsed = reactive(False)
    manually_expanded = reactive(False)

    def __init__(self, user_bubble: MessageBubble):
        super().__init__(classes="chat-turn")
        self.user_bubble = user_bubble
        self.agent_bubbles: list[MessageBubble] = []

    def watch_collapsed(self, value: bool) -> None:
        self.set_class(value, "collapsed")
        self.user_bubble.collapsed = value
        for bubble in self.agent_bubbles:
            bubble.collapsed = value

    def on_mount(self) -> None:
        self.watch(self.app, "active_turn", self._on_active_turn_change)
        self._on_active_turn_change(self.app.active_turn)  # type: ignore

    def _on_active_turn_change(self, active_turn: ChatTurn | None) -> None:
        is_active = active_turn is self
        self.user_bubble.is_active = is_active
        for bubble in self.agent_bubbles:
            bubble.is_active = is_active

        if is_active:
            self.collapsed = False

    def auto_collapse(self) -> None:
        if not self.manually_expanded:
            self.collapsed = True

    def watch_is_active(self, value: bool) -> None:
        self.user_bubble.is_active = value
        for bubble in self.agent_bubbles:
            bubble.is_active = value

    def compose(self) -> ComposeResult:
        yield self.user_bubble

    def toggle_collapse(self) -> None:
        if self.collapsed:
            self.manually_expanded = True
            self.collapsed = False
        else:
            self.manually_expanded = False
            self.collapsed = True

    def expand_turn(self) -> None:
        self.app.active_turn = self  # type: ignore

    async def mount_bubble(self, bubble: MessageBubble) -> None:
        bubble.collapsed = self.collapsed
        bubble.is_active = self.app.active_turn is self  # type: ignore
        self.agent_bubbles.append(bubble)
        await self.mount(bubble)

    def on_click(self, event: Click) -> None:
        self.toggle_collapse()
        event.stop()


class ThinkingIndicator(Static):
    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)
    crew_mode = reactive(False)
    current_active_agent = reactive("")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._tick)

        for prop in [
            "status_label",
            "active_queries",
            "is_activating_mode",
            "crew_mode",
            "current_active_agent",
        ]:
            self.watch(self.app, prop, self._sync_app_prop)

        self._sync_app_prop(None)

    def _sync_app_prop(self, _: Any) -> None:
        self.status_label = self.app.status_label  # type: ignore
        self.active_queries = self.app.active_queries  # type: ignore
        self.is_activating_mode = self.app.is_activating_mode  # type: ignore
        self.crew_mode = self.app.crew_mode  # type: ignore
        self.current_active_agent = self.app.current_active_agent  # type: ignore

    def _tick(self) -> None:
        self._update_display(None)

    def _update_display(self, _: Any) -> None:
        from .theme import SPINNER_FRAMES

        if self.active_queries == 0 and not self.is_activating_mode:
            self.update("")
            self.display = False
            return

        idx = getattr(self, "_spinner_idx", 0)
        self._spinner_idx = idx + 1

        frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
        dots = "." * (idx % 4)

        if self.is_activating_mode:
            status = f"Activating Mode{dots}"
        else:
            role = self.current_active_agent
            prefix = f"[{role}] " if self.crew_mode and role else ""
            status = f"{frame} {prefix}{self.status_label}{dots}"

        self.update(status)
        self.display = True


class QueuedMessagesDisplay(Vertical):
    def update_queue(self, messages: list[str]) -> None:
        self.query("*").remove()
        if not messages:
            self.display = False
            return

        self.mount(Label("Queued:", classes="queue-header"))
        for msg in messages:
            self.mount(Label(msg, classes="queue-badge"))

        self.display = True


class ToastManager:
    def __init__(self, timeout: float = 5.0) -> None:
        self.active_toasts: OrderedDict[str, str] = OrderedDict()
        self.last_toast_time: float = 0
        self.timeout = timeout

    def notify(self, app: Any, message: str, type: str = "system") -> None:
        now = time.time()

        if now - self.last_toast_time > self.timeout:
            self.active_toasts.clear()

        if type in self.active_toasts:
            del self.active_toasts[type]

        self.active_toasts[type] = message
        self.active_toasts.move_to_end(type, last=False)

        self._clear_ui(app)
        self.last_toast_time = now

        content = "\n".join(self.active_toasts.values())
        app.notify(content, timeout=self.timeout)

    def clear(self, app: Any) -> None:
        self.active_toasts.clear()
        self._clear_ui(app)

    def _clear_ui(self, app: Any) -> None:
        if hasattr(app, "clear_notifications"):
            app.clear_notifications()
            return

        if hasattr(app, "screen") and app.screen:
            app.screen.query("Toast, Notification, .textual-notification").remove()
