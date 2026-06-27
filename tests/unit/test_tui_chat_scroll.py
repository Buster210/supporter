from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.events import MouseScrollUp
from textual.widgets import Button, Static

from supporter.tui.chat import ChatContainer


class _ScrollApp(App[None]):
    CSS = """
    #chat-view { height: 1fr; overflow-y: scroll; }
    #scroll-btn-wrapper { dock: bottom; height: auto; }
    #scroll-btn-wrapper.hidden { display: none; }
    """

    def compose(self) -> ComposeResult:
        with ChatContainer(id="chat-view"):
            pass
        with Horizontal(id="scroll-btn-wrapper", classes="hidden"):
            yield Button("bottom", id="scroll-bottom-btn")

    async def add_lines(self, count: int) -> None:
        view = self.query_one("#chat-view", ChatContainer)
        await view.mount_all(Static(f"line {i}") for i in range(count))


def _wheel_up(
    view: ChatContainer, *, ctrl: bool = False, shift: bool = False
) -> MouseScrollUp:
    return MouseScrollUp(
        widget=view,
        x=0,
        y=0,
        delta_x=0,
        delta_y=-1,
        button=0,
        shift=shift,
        meta=False,
        ctrl=ctrl,
    )


async def _settle(pilot: object) -> None:
    # Auto-follow defers its scroll to after the next refresh, so geometry
    # needs a few message-queue drains to reach its final state.
    for _ in range(4):
        await pilot.pause()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_auto_follow_pins_to_bottom_by_default() -> None:
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)
        assert view.max_scroll_y > 0  # content exceeds viewport
        assert view._follow is True
        assert view.scroll_y >= view.max_scroll_y - 1


@pytest.mark.asyncio
async def test_wheel_up_disarms_follow_and_new_content_does_not_yank() -> None:
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)

        view.scroll_to(y=5, animate=False)  # programmatic move is not a gesture
        await pilot.pause()
        assert view._follow is True  # reflow-style clamp must not disarm

        view._on_mouse_scroll_up(_wheel_up(view))  # explicit user wheel-up
        await pilot.pause()
        assert view._follow is False

        # New content arriving must NOT drag the viewport down.
        await app.add_lines(20)
        await pilot.pause()
        assert view._follow is False
        assert view.scroll_y < view.max_scroll_y - 1


@pytest.mark.asyncio
async def test_wheel_up_on_short_content_is_not_re_armed_by_geometry() -> None:
    # Regression for the "stuck at the bottom" trap. When content is only just
    # taller than the viewport, max_scroll_y == 1 and the whole scroll range sits
    # inside the bottom tolerance band. The old geometry-only re-arm flipped
    # follow back on the instant a wheel-up landed in that band, so the gesture
    # was cancelled and the next streamed chunk yanked the viewport to the bottom
    # — the user could never read back. Re-arm is now direction-aware, so a
    # wheel-up here disarms and STAYS disarmed through subsequent growth.
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(11)
        await _settle(pilot)
        assert view.max_scroll_y == 1  # entire range is within the tolerance band

        view._on_mouse_scroll_up(_wheel_up(view))
        await pilot.pause()
        assert view._follow is False  # gesture must survive the geometry band

        await app.add_lines(40)  # streaming growth must not steal the viewport
        await _settle(pilot)
        assert view._follow is False
        assert view.scroll_y < view.max_scroll_y - 1  # not yanked to the bottom


@pytest.mark.asyncio
async def test_wheel_up_disarm_is_guarded() -> None:
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)

        view.scroll_to(y=5, animate=False)
        await pilot.pause()

        view._on_mouse_scroll_up(_wheel_up(view, ctrl=True))  # horizontal intent
        await pilot.pause()
        assert view._follow is True  # modifier wheel never disarms vertical follow


@pytest.mark.asyncio
async def test_wheel_up_on_unscrollable_content_keeps_follow() -> None:
    # Content that fits the viewport has no scroll range (max_scroll_y == 0); a
    # wheel-up there is a no-op gesture and must not disarm follow, or later
    # growth past the viewport would never pin to the bottom.
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(2)
        await _settle(pilot)
        assert view.max_scroll_y == 0

        view._on_mouse_scroll_up(_wheel_up(view))
        await pilot.pause()
        assert view._follow is True

        await app.add_lines(50)
        await _settle(pilot)
        assert view.scroll_y >= view.max_scroll_y - 1  # followed once scrollable


@pytest.mark.asyncio
async def test_pageup_disarms_and_jump_to_bottom_rearms() -> None:
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)

        view.scroll_page_up(animate=False)  # keyboard PageUp path
        await pilot.pause()
        assert view._follow is False

        view.jump_to_bottom()
        await _settle(pilot)
        assert view._follow is True

        await app.add_lines(20)
        await _settle(pilot)
        assert view.scroll_y >= view.max_scroll_y - 1


@pytest.mark.asyncio
async def test_reflow_clamp_does_not_disarm_follow() -> None:
    # Regression: a turn auto-collapsing / markdown re-rendering shrinks the
    # virtual size and clamps scroll_y downward. That is a programmatic move,
    # not a user gesture, and must leave auto-follow armed — otherwise streaming
    # strands the viewport mid-history ("stuck in the middle").
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)
        assert view._follow is True

        view.scroll_to(y=view.max_scroll_y - 10, animate=False)  # simulate reflow clamp
        await pilot.pause()
        assert view._follow is True

        await app.add_lines(20)
        await _settle(pilot)
        assert view._follow is True
        assert view.scroll_y >= view.max_scroll_y - 1


@pytest.mark.asyncio
async def test_follow_end_respects_user_scroll_position() -> None:
    app = _ScrollApp()
    async with app.run_test(size=(40, 10)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        await app.add_lines(50)
        await _settle(pilot)

        view.scroll_to(y=5, animate=False)
        await pilot.pause()
        view._on_mouse_scroll_up(_wheel_up(view))
        await pilot.pause()
        assert view._follow is False
        at = view.scroll_y

        view.follow_end()  # background update: must be a no-op while scrolled up
        await pilot.pause()
        assert view.scroll_y == at

        view.jump_to_bottom()  # explicit user intent: forces bottom + re-arms
        await _settle(pilot)
        assert view._follow is True
        assert view.scroll_y >= view.max_scroll_y - 1
