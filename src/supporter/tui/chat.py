from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from . import SupporterApp
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click, MouseScrollUp
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Label, Static

from .bubble import MessageBubble
from .constants import SPINNER_FRAMES
from .utils import apply_crystal_gradient

# Re-arm auto-follow once the viewport is within this many rows of the bottom.
# Kept tight so the user has to be essentially at the bottom to resume following.
_BOTTOM_TOLERANCE_ROWS = 1

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
    # Auto-follow the newest content. Disarmed ONLY by an explicit upward user
    # gesture (mouse wheel up, PageUp, Home) and re-armed only when the user
    # scrolls back DOWN to the bottom. Re-arm is direction-aware on purpose:
    # geometry alone is a trap — on short content the bottom tolerance band
    # covers the whole (tiny) scroll range, so re-arming on proximity would
    # instantly cancel a wheel-up and let the next streamed chunk yank the
    # viewport back to the bottom (the reported "stuck at bottom" bug). Reflow
    # clamps (a turn auto-collapsing, markdown re-rendering) move scroll_y
    # DOWNWARD (new < old), so requiring new > old also keeps reflow from ever
    # re-arming spuriously while leaving a disarmed follow disarmed.
    _follow = True

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        # MUST call super(): Widget.watch_scroll_y runs _refresh_scroll() (the
        # repaint that moves content to the new offset) and tracks the scrollbar.
        # Overriding without it froze the view — scroll_y changed but the content
        # never moved, so the wheel could never reach the top (the reported bug).
        super().watch_scroll_y(old_value, new_value)
        at_bottom = new_value >= self.max_scroll_y - _BOTTOM_TOLERANCE_ROWS
        if new_value > old_value and at_bottom:
            self._follow = True
        self._update_scroll_btn()

    def watch_virtual_size(self, _old_value: object, _new_value: object) -> None:
        if self._follow:
            # immediate=True pins to the bottom in this same frame so the view
            # never trails the growing reply (deferred-only lands one render
            # behind, and the render throttle widens that gap into a visible
            # "stuck mid-reply"); the deferred call then settles onto the true
            # bottom once layout works out the final max_scroll_y.
            self.scroll_end(animate=False, immediate=True)
            self.scroll_end(animate=False)
        self._update_scroll_btn()

    # User upward gestures (wheel/PageUp/Home) disarm auto-follow so streamed
    # content stops yanking the viewport back to the bottom. No origin healing:
    # native Textual reaches the true top on its own (verified empirically — the
    # old "negative-origin" workaround was the actual cause of the unreachable
    # top, collapsing every turn per wheel tick and jumping the viewport).
    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        # Ignore when there is no scroll range (content fits, gesture is a no-op
        # and later growth must still pin to bottom) or a horizontal modifier.
        if not (event.ctrl or event.shift) and self.max_scroll_y > 0:
            self._follow = False
        super()._on_mouse_scroll_up(event)

    def scroll_page_up(self, *args: Any, **kwargs: Any) -> None:
        self._follow = False
        super().scroll_page_up(*args, **kwargs)

    def scroll_home(self, *args: Any, **kwargs: Any) -> None:
        self._follow = False
        super().scroll_home(*args, **kwargs)

    def follow_end(self) -> None:
        """Scroll to the bottom only while auto-follow is armed.

        Use for background updates (delegation progress, system messages) that
        must not steal the viewport when the user has scrolled up to read.
        """
        if self._follow:
            self.scroll_end(animate=False)

    def jump_to_bottom(self) -> None:
        """Force the viewport to the bottom and resume auto-follow."""
        self._follow = True
        self.scroll_end(animate=False)

    def _update_scroll_btn(self) -> None:
        from textual.css.query import NoMatches

        try:
            if not hasattr(self, "_scroll_wrapper"):
                self._scroll_wrapper = self.app.query_one("#scroll-btn-wrapper")
        except NoMatches:
            return
        at_bottom = self.max_scroll_y <= 0 or (
            self.scroll_y >= self.max_scroll_y - _BOTTOM_TOLERANCE_ROWS
        )
        self._scroll_wrapper.set_class(at_bottom, "hidden")


class ChatTurn(Vertical):
    collapsed = reactive(False)
    manually_expanded = reactive(False)

    def __init__(self, user_bubble: MessageBubble):
        super().__init__(classes="chat-turn")
        self.user_bubble = user_bubble
        self.agent_bubbles: list[MessageBubble] = []
        self.turn_start_time = time.perf_counter()
        self._delegation_job_id: str | None = None

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
        # Only the latest bubble in a turn keeps its meta line — suppress the
        # previous ones so a multi-step task shows metadata just once, at the end.
        for prev in self.agent_bubbles:
            prev.hide_meta()
        bubble.collapsed = self.collapsed
        bubble.is_active = cast("SupporterApp", self.app).active_turn is self
        self.agent_bubbles.append(bubble)
        await self.mount(bubble)

    def on_click(self, event: Click) -> None:
        # toggle_collapse() must run BEFORE expand_turn(): expand_turn() sets
        # active_turn, whose watcher synchronously sets collapsed=False — if that
        # ran first, toggle_collapse() would see an already-expanded turn and
        # re-collapse it, so a click on a collapsed turn never expanded it.
        was_collapsed = self.collapsed
        self.toggle_collapse()
        if was_collapsed:
            self.expand_turn()
        event.stop()


class ThinkingIndicator(Static):
    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)

    _spinner_timer: Timer | None
    _dots_timer: Timer | None

    def on_mount(self) -> None:
        self._spinner_timer = None
        self._dots_timer = None
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
        if should_run:
            if self._spinner_timer is None:
                self._spinner_timer = self.set_interval(1 / 40, self._spinner_tick)
            if self._dots_timer is None:
                self._dots_timer = self.set_interval(1 / 10, self._dots_tick)
            self._update_display(None)
        else:
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
                self._spinner_timer = None
            if self._dots_timer is not None:
                self._dots_timer.stop()
                self._dots_timer = None
            self._update_display(None)

    def _spinner_tick(self) -> None:
        self._update_display("spinner")

    def _dots_tick(self) -> None:
        self._update_display("dots")

    def _update_display(self, advance: str | None) -> None:
        if self.active_queries == 0 and not self.is_activating_mode:
            self.update("")
            self.display = False
            return
        if advance == "spinner":
            self._spinner_idx = getattr(self, "_spinner_idx", 0) + 1
        elif advance == "dots":
            self._dots_idx = getattr(self, "_dots_idx", 0) + 1
        spinner_idx = getattr(self, "_spinner_idx", 0)
        dots_idx = getattr(self, "_dots_idx", 0)
        frame = SPINNER_FRAMES[spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * (dots_idx % 4)
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
