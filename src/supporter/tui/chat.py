from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.widgets import Label, Static

from ..config import SCROLL_STEP, SPINNER_FRAMES
from .bubble import MessageBubble
from .utils import apply_crystal_gradient

SUPPORTER_ART = (
    " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ \n"
    " ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ \n"
    " ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
)


class SupporterHeader(Static):
    def render(self) -> Text:
        return apply_crystal_gradient(SUPPORTER_ART)


class ChatContainer(Vertical):
    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        for _ in range(SCROLL_STEP):
            self.scroll_down()
        event.prevent_default()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        for _ in range(SCROLL_STEP):
            self.scroll_up()
        event.prevent_default()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        self._was_at_bottom = new_value >= self.max_scroll_y - 4
        self._update_scroll_btn()

    def watch_virtual_size(self, old_value: object, new_value: object) -> None:
        if getattr(self, "_was_at_bottom", True):
            self.scroll_end(animate=False)
        self._update_scroll_btn()

    def _update_scroll_btn(self) -> None:
        try:
            if not hasattr(self, "_scroll_wrapper"):
                self._scroll_wrapper = self.app.query_one("#scroll-btn-wrapper")

            wrapper = self._scroll_wrapper
            at_bottom = self.scroll_y >= self.max_scroll_y - 4
            if not at_bottom:
                if wrapper.has_class("hidden"):
                    wrapper.remove_class("hidden")
            else:
                if not wrapper.has_class("hidden"):
                    wrapper.add_class("hidden")
        except Exception:  # noqa: S110
            pass


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
        if self.collapsed:
            self.expand_turn()
        self.toggle_collapse()
        event.stop()


class ThinkingIndicator(Static):
    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._tick)
        for prop in [
            "status_label",
            "active_queries",
            "is_activating_mode",
        ]:
            self.watch(self.app, prop, self._sync_app_prop)
        self._sync_app_prop(None)

    def _sync_app_prop(self, _: Any) -> None:
        self.status_label = self.app.status_label  # type: ignore
        self.active_queries = self.app.active_queries  # type: ignore
        self.is_activating_mode = self.app.is_activating_mode  # type: ignore

    def _tick(self) -> None:
        self._update_display(None)

    def _update_display(self, _: Any) -> None:

        if self.active_queries == 0 and not self.is_activating_mode:
            self.update("")
            self.display = False
            return
        idx = getattr(self, "_spinner_idx", 0)
        self._spinner_idx = idx + 1
        frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
        dots = "." * (idx % 4)
        status = (
            f"Activating Mode{dots}"
            if self.is_activating_mode
            else f"{frame} {self.status_label}{dots}"
        )
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
