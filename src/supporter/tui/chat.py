from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from . import SupporterApp
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Label, Static

from .bubble import MessageBubble
from .constants import SPINNER_FRAMES
from .utils import apply_crystal_gradient

_AT_BOTTOM_THRESHOLD = 4

_SUPPORTER_ART = (
    " █▀▀ █ █ █▀█ █▀█ █▀█ █▀█ ▀█▀ █▀▀ █▀█ \n"
    " ▀▀█ █ █ █▀▀ █▀▀ █ █ █▀▄  █  █▀▀ █▀▄ \n"
    " ▀▀▀ ▀▀▀ ▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀▀▀ ▀ ▀ "
)
_SUPPORTER_ART_RENDERED: Text | None = None


class SupporterHeader(Static):
    def render(self) -> Text:
        global _SUPPORTER_ART_RENDERED
        if _SUPPORTER_ART_RENDERED is None:
            _SUPPORTER_ART_RENDERED = apply_crystal_gradient(_SUPPORTER_ART)
        return _SUPPORTER_ART_RENDERED


class WelcomeBanner(Static):
    message = reactive("")

    def render(self) -> str:
        return self.message

    def watch_message(self, message: str) -> None:
        self.set_class(not message, "hidden")


class ChatContainer(Vertical):
    def watch_scroll_y(self, _old_value: float, new_value: float) -> None:
        self._was_at_bottom = new_value >= self.max_scroll_y - _AT_BOTTOM_THRESHOLD
        self._update_scroll_btn()

    def watch_virtual_size(self, _old_value: object, new_value: object) -> None:
        if getattr(self, "_was_at_bottom", True):
            self.scroll_end(animate=False)
        self._update_scroll_btn()

    def _update_scroll_btn(self) -> None:
        from textual.css.query import NoMatches

        try:
            if not hasattr(self, "_scroll_wrapper"):
                self._scroll_wrapper = self.app.query_one("#scroll-btn-wrapper")
        except NoMatches:
            return

        wrapper = self._scroll_wrapper
        at_bottom = (
            self.max_scroll_y <= 0
            or getattr(self, "_was_at_bottom", True)
            or self.scroll_y >= self.max_scroll_y - _AT_BOTTOM_THRESHOLD
        )
        if not at_bottom:
            if wrapper.has_class("hidden"):
                wrapper.remove_class("hidden")
        elif not wrapper.has_class("hidden"):
            wrapper.add_class("hidden")


class ChatTurn(Vertical):
    collapsed = reactive(False)
    manually_expanded = reactive(False)

    def __init__(self, user_bubble: MessageBubble):
        super().__init__(classes="chat-turn")
        self.user_bubble = user_bubble
        self.agent_bubbles: list[MessageBubble] = []
        self.turn_start_time = time.perf_counter()

    def watch_collapsed(self, value: bool) -> None:
        self.set_class(value, "collapsed")
        self.user_bubble.collapsed = value
        for bubble in self.agent_bubbles:
            bubble.collapsed = value

    def on_mount(self) -> None:
        self.watch(self.app, "active_turn", self._on_active_turn_change)
        self._on_active_turn_change(cast("SupporterApp", self.app).active_turn)

    def _on_active_turn_change(self, active_turn: ChatTurn | None) -> None:
        is_active = active_turn is self
        self.user_bubble.is_active = is_active
        for bubble in self.agent_bubbles:
            bubble.is_active = is_active
        if is_active:
            self.collapsed = False

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

    def auto_collapse(self) -> None:
        """Collapse the turn unless user has manually expanded it."""
        if not self.manually_expanded:
            self.collapsed = True

    def expand_turn(self) -> None:
        cast("SupporterApp", self.app).active_turn = self

    async def mount_bubble(self, bubble: MessageBubble) -> None:
        bubble.collapsed = self.collapsed
        bubble.is_active = cast("SupporterApp", self.app).active_turn is self
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

    _timer: Timer | None

    def on_mount(self) -> None:
        self._timer = None
        for prop in [
            "status_label",
            "active_queries",
            "is_activating_mode",
        ]:
            self.watch(self.app, prop, self._sync_app_prop)
        self._sync_app_prop(None)

    def _sync_app_prop(self, _: Any) -> None:
        app = cast("SupporterApp", self.app)
        self.status_label = app.status_label
        self.active_queries = app.active_queries
        self.is_activating_mode = app.is_activating_mode
        self._update_timer_state()

    def _update_timer_state(self) -> None:
        should_run = self.active_queries > 0 or self.is_activating_mode
        if should_run and self._timer is None:
            self._timer = self.set_interval(1 / 20, self._tick)
            self._update_display(None)
        elif not should_run and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._update_display(None)

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
