"""Delegation UI mounts inside the triggering bubble, before its meta line."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from supporter.llm.types import Message, TextPart
from supporter.tui import SupporterApp
from supporter.tui.bubble import MessageBubble
from supporter.tui.chat import ChatContainer


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


@pytest.mark.asyncio
async def test_replay_skips_empty_delegation_only_model_message() -> None:
    """A delegation-only turn persists an empty model message; replaying it
    must not mount an empty (zero-element) agent bubble."""
    app = SupporterApp()
    store = MagicMock()
    store.load.return_value = [
        Message(role="user", parts=[TextPart(text="do the thing")]),
        Message(role="model", parts=[]),  # delegation-only: no text/tool parts
    ]
    app.agent = MagicMock()
    app.agent._store = store

    async with app.run_test(size=(80, 24)):
        await app._replay_history()
        chat_view = app.query_one("#chat-view", ChatContainer)
        agent_bubbles = [
            b for b in chat_view.query(MessageBubble) if b.role == "agent"
        ]
        assert agent_bubbles == []
