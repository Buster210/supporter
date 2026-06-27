"""Full-history regression: the whole conversation must be reachable start-to-end.

The existing scroll tests drive dumb ``Static`` lines, so they exercise the
follow/disarm state machine but never the real history pipeline (``ChatTurn`` +
``MessageBubble`` laid out under the app's real CSS). They also never assert the
two things the user actually reported: that scrolling to the TOP reveals the
EARLIEST turn, and that a live reply streaming in does NOT yank the viewport back
down while the user is reading up. This test closes that gap with the real
widgets under the real stylesheet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
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
        view.jump_to_bottom()  # production jumps to bottom on every new user turn
        return turn


async def _settle(pilot: object) -> None:
    for _ in range(6):
        await pilot.pause()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_full_history_reachable_and_stable_while_streaming() -> None:
    n = 60
    app = _HistoryApp()
    async with app.run_test(size=(50, 12)) as pilot:
        view = app.query_one("#chat-view", ChatContainer)
        for i in range(n):
            await app.add_turn(i)
        await _settle(pilot)

        turns = view.query(ChatTurn)
        # Whole history stays mounted — no cap / truncation hides old turns.
        assert len(turns) == n
        # Real bubbles produce a real scroll range under the real CSS.
        assert view.max_scroll_y > 0
        # Fresh conversation follows the bottom by default.
        assert view.scroll_y >= view.max_scroll_y - 1

        first_turn, last_turn = turns[0], turns[len(turns) - 1]
        vh = view.size.height  # viewport height in rows

        # At the bottom (default), the LATEST turn is on-screen and the EARLIEST
        # is scrolled far above the viewport — i.e. old history really is hidden
        # off the top until you scroll to it (this is the user's starting state).
        assert last_turn.region.y < vh  # newest visible
        assert first_turn.region.y < 0  # earliest is above the viewport, not shown

        # User scrolls to the very top. This must (a) disarm auto-follow and
        # (b) bring the EARLIEST turn into the document-space top while pushing the
        # LATEST turn off the bottom — proving we navigated a long history, not
        # that everything happened to fit.
        view.scroll_home(animate=False)
        await _settle(pilot)
        assert view.scroll_y == 0
        assert view._follow is False
        assert first_turn.region.y >= 0 and first_turn.region.y < vh  # now on-screen
        assert last_turn.region.y >= vh  # latest now pushed off the bottom

        # A live reply streams into the active (bottom) turn. While the user reads
        # the top, the growing virtual size must NOT yank the viewport down.
        max_before = view.max_scroll_y
        agent_bubble = last_turn.agent_bubbles[0]
        for _ in range(30):
            agent_bubble.append_token("more streamed text " * 3)
            await pilot.pause()
        await _settle(pilot)
        assert view.max_scroll_y > max_before  # content genuinely grew
        assert view._follow is False
        assert view.scroll_y == 0  # held at the top through the whole stream
        assert first_turn.region.y < vh  # earliest still on-screen, not yanked away
