"""Regression for the reported bug: real mouse-wheel-up (not PageUp/Home) must
reach the absolute top of a long history. Drives MouseScrollUp events through
the real ChatContainer + ChatTurn + MessageBubble under the real stylesheet —
the exact gesture that previously stuck mid-history.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.events import MouseScrollUp
from textual.reactive import reactive
from textual.widgets import Button

from supporter.tui.bubble import MessageBubble
from supporter.tui.chat import ChatContainer, ChatTurn

_STYLES = Path(__file__).resolve().parents[2] / "src/supporter/tui/styles.tcss"


class _HistoryApp(App[None]):
    CSS_PATH = str(_STYLES)
    active_turn: reactive[ChatTurn | None] = reactive(None)

    def compose(self) -> ComposeResult:
        with ChatContainer(id="chat-view"):
            pass
        with Horizontal(id="scroll-btn-wrapper", classes="hidden"):
            yield Button("bottom", id="scroll-bottom-btn")

    async def add_turn(self, i: int) -> ChatTurn:
        view = self.query_one("#chat-view", ChatContainer)
        turn = ChatTurn(MessageBubble(role="user", content=f"USERMSG-{i}"))
        self.active_turn = turn
        await view.mount(turn)
        await turn.mount_bubble(
            MessageBubble(role="agent", content=f"AGENTREPLY-{i} " + "x" * 40)
        )
        view.jump_to_bottom()
        return turn


def _wheel_up(view: ChatContainer) -> MouseScrollUp:
    return MouseScrollUp(
        widget=view,
        x=0,
        y=0,
        delta_x=0,
        delta_y=-1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


async def _settle(pilot: object) -> None:
    for _ in range(6):
        await pilot.pause()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mouse_wheel_reaches_absolute_top() -> None:
    app = _HistoryApp()
    async with app.run_test(size=(50, 12)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        for i in range(60):
            await app.add_turn(i)
        await _settle(pilot)
        assert view.scroll_y >= view.max_scroll_y - 1  # starts at bottom

        # Drive real wheel-up ticks until movement stops — must land on y==0.
        prev = -1.0
        ticks = 0
        while view.scroll_y != prev and ticks < 600:
            prev = view.scroll_y
            view._on_mouse_scroll_up(_wheel_up(view))
            await pilot.pause()
            ticks += 1

        assert view._follow is False  # wheel-up disarmed follow
        assert view.scroll_y == 0, f"wheel stuck at y={view.scroll_y}, not the top"
        assert app.query(ChatTurn)[0].region.y >= 0  # earliest turn on-screen
