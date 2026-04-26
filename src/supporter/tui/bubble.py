from __future__ import annotations

import contextlib
import re
from typing import Any, cast

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Label, Static

from ..config import (
    COLLAPSED_SUMMARY_LEN,
    MARKDOWN_SYNTAX_MARKERS,
    THEME,
    TOOL_ARG_MAX_LEN,
    TOOL_ARG_TRUNC_LEN,
)


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
        bubble.toggle_section(self)

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
        streaming: bool = False,
    ):
        super().__init__()
        self.role = role
        self.content = content
        self.model = model
        self.duration = duration
        self.streaming = streaming
        self.thoughts = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.elements: list[dict[str, Any]] = []

        if self.content:
            self.elements.append(
                {"type": "content", "content": self.content, "collapsed": False}
            )

        self.collapsible = True
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
        return f"({model_info})"

    def _should_use_markdown(self, text: str) -> bool:
        return any(re.search(m, text, re.MULTILINE) for m in MARKDOWN_SYNTAX_MARKERS)

    def on_section_header_toggle_request(
        self, event: SectionHeader.ToggleRequest
    ) -> None:
        self.toggle_section(event.header)

    def toggle_section(self, section: SectionHeader) -> None:
        with contextlib.suppress(ValueError, Exception):
            container = self.query_one("#elements-container")
            idx = container.children.index(section)
            if idx == -1:
                return

            current_child_idx = 0
            for el in self.elements:
                if el["type"] in ("thought", "tool_calls"):
                    if current_child_idx == idx:
                        el["collapsed"] = not el["collapsed"]
                        el["manually_interacted"] = True
                        break
                    current_child_idx += 2
                else:
                    current_child_idx += 1
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
        summary = self.content.split("\n")[0][:COLLAPSED_SUMMARY_LEN]
        if len(self.content) > COLLAPSED_SUMMARY_LEN or "\n" in self.content:
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
        self._ensure_correct_widget_count(container)
        self._refresh_widget_content(container)

    def _ensure_correct_widget_count(self, container: Any) -> None:
        current_widgets = container.query("*")
        expected_count = sum(
            2 if el["type"] in ("thought", "tool_calls") else 1 for el in self.elements
        )
        if len(current_widgets) < expected_count:
            self._mount_missing_widgets(container, current_widgets)
        elif len(current_widgets) > expected_count:
            self._rebuild_widgets(container)

    def _mount_missing_widgets(self, container: Any, current_widgets: Any) -> None:
        element_idx = 0
        w_idx = 0
        while w_idx < len(current_widgets) and element_idx < len(self.elements):
            el = self.elements[element_idx]
            w_idx += 2 if el["type"] in ("thought", "tool_calls") else 1
            element_idx += 1
        new_widgets = []
        for i in range(element_idx, len(self.elements)):
            new_widgets.extend(self._create_widgets_for_element(i, self.elements[i]))
        if new_widgets:
            container.mount(*new_widgets)

    def _rebuild_widgets(self, container: Any) -> None:
        container.remove_children()
        new_widgets = []
        for i, el in enumerate(self.elements):
            new_widgets.extend(self._create_widgets_for_element(i, el))
        container.mount(*new_widgets)

    def _refresh_widget_content(self, container: Any) -> None:
        current_widgets = container.query("*")
        w_idx = 0
        for i, el in enumerate(self.elements):
            if el["type"] in ("thought", "tool_calls"):
                if w_idx + 1 >= len(current_widgets):
                    break
                self._update_section_widget(
                    current_widgets[w_idx], current_widgets[w_idx + 1], el, i
                )
                w_idx += 2
            else:
                if w_idx >= len(current_widgets):
                    break
                self._update_content_widget(current_widgets[w_idx], el)
                w_idx += 1

    def _update_section_widget(
        self, header: Any, view: Any, el: dict[str, Any], idx: int
    ) -> None:
        header = cast(SectionHeader, header)
        view = cast(Static, view)
        if el["type"] == "thought":
            is_thinking = (
                self.streaming and idx == len(self.elements) - 1 and not self.content
            )
            label = "Thinking" if is_thinking else "Thoughts"
            header.update_label(label, el["collapsed"], self.collapsible)
            view.update(RichMarkdown(el["content"]))
            view.display = not el["collapsed"] if self.collapsible else True
        else:
            header.update_label("Tools Used", el["collapsed"], self.collapsible)
            view.update(self._format_tool_calls(el["calls"]))
            view.display = not el["collapsed"] if self.collapsible else True

    def _update_content_widget(self, view: Any, el: dict[str, Any]) -> None:
        view = cast(Static, view)
        content = el["content"]
        if self._should_use_markdown(content):
            view.update(RichMarkdown(content))
        else:
            view.update(content)

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
                    (
                        f"{k}={str(v)[:TOOL_ARG_TRUNC_LEN]}..."
                        if len(str(v)) > TOOL_ARG_MAX_LEN
                        else f"{k}={v}"
                    )
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
            if (
                self.elements
                and self.elements[-1]["type"] in ("thought", "tool_calls")
                and not self.elements[-1].get("manually_interacted")
            ):
                self.elements[-1]["collapsed"] = True
            self.elements.append(
                {
                    "type": etype,
                    "content": token,
                    "collapsed": False,
                    "manually_interacted": False,
                }
            )
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
                if (
                    self.elements
                    and self.elements[-1]["type"] == "thought"
                    and not self.elements[-1].get("manually_interacted")
                ):
                    self.elements[-1]["collapsed"] = True
                self.elements.append(
                    {
                        "type": "tool_calls",
                        "calls": [entry],
                        "collapsed": False,
                        "manually_interacted": False,
                    }
                )
            else:
                self.elements[-1]["calls"].append(entry)
            self._update_ui_content()

    def finalize(
        self,
        model: str | None = None,
        duration: float | None = None,
    ) -> None:
        self.model = model or self.model
        self.duration = duration or self.duration
        self.streaming = False
        if (
            self.elements
            and self.elements[-1]["type"] in ("thought", "tool_calls")
            and not self.elements[-1].get("manually_interacted")
        ):
            self.elements[-1]["collapsed"] = True
        if self._meta_label:
            self._meta_label.update(self._get_meta_text())
            self._meta_label.display = not self.collapsed
        if self._message_view and self._message_view.size.width > 0:
            self._message_view.styles.width = self._message_view.size.width
        self._update_ui_content()

    def on_click(self, event: Click) -> None:
        if not self.collapsed:
            return
        from .chat import ChatTurn

        if isinstance(self.parent, ChatTurn):
            self.parent.toggle_collapse()
        else:
            self.collapsed = False
            self._update_ui_content()
            if self._meta_label:
                self._meta_label.display = True
        event.stop()
