"""Delegation UI mounts inside the triggering bubble, before its meta line."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from supporter.tui.bubble import MessageBubble


class _BubbleHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield MessageBubble(
            role="agent", content="Waiting for the plan", model="gemini", duration=8.93
        )


async def test_append_before_meta_positions_widget_before_meta() -> None:
    async with _BubbleHarness().run_test() as pilot:
        bubble = pilot.app.query_one(MessageBubble)
        assert bubble._meta_label is not None

        widget = Static("delegation table", classes="delegation-progress")
        assert bubble.append_before_meta(widget) is True

        parent = bubble._meta_label.parent
        assert parent is not None
        children = list(parent.children)
        assert children.index(widget) < children.index(bubble._meta_label)


def test_append_before_meta_returns_false_when_not_composed() -> None:
    bubble = MessageBubble(role="agent", content="x", model="m", duration=0.1)
    bubble._meta_label = None
    assert bubble.append_before_meta(Static("t")) is False


async def test_defer_meta_keeps_meta_hidden_until_revealed() -> None:
    async with _BubbleHarness().run_test() as pilot:
        bubble = pilot.app.query_one(MessageBubble)
        bubble.defer_meta = True
        bubble.finalize(model="gemini", duration=1.0)
        assert bubble._meta_label is not None
        assert bubble._meta_label.display is False  # deferred: stays hidden

        bubble.reveal_meta()
        assert bubble.defer_meta is False
        assert bubble._meta_label is not None
        assert bubble._meta_label.display is True  # now shown, after content
