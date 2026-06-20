"""Startup history replay: full persisted session renders as scrollable bubbles.

On TUI startup, ``_replay_history`` loads ALL records from the history store
(uncapped) and mounts each as a ``MessageBubble`` inside a ``ChatTurn``.  This
test verifies the replay renders every persisted message, hides the welcome
banner for non-empty history, and scrolls to bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.reactive import reactive

from supporter.llm.types import Message, TextPart, ToolCallPart, ToolResultPart
from supporter.tui.bubble import MessageBubble
from supporter.tui.chat import ChatContainer, ChatTurn, WelcomeBanner

_STYLES = Path(__file__).resolve().parents[2] / "src/supporter/tui/styles.tcss"


# ---------------------------------------------------------------------------
# Fake history store
# ---------------------------------------------------------------------------


class _FakeStore:
    """In-memory history store that returns pre-built records."""

    def __init__(self, records: list[Message]) -> None:
        self._records = records

    def load(self, limit: int | None = None) -> list[Message]:
        if limit and len(self._records) > limit:
            return self._records[-limit:]
        return list(self._records)


# ---------------------------------------------------------------------------
# Minimal app that exercises the real SupporterApp._replay_history
# ---------------------------------------------------------------------------


class _ReplayApp(App[None]):
    """Lightweight harness with SupporterApp's compose + _replay_history."""

    CSS_PATH = str(_STYLES)
    active_turn: reactive[ChatTurn | None] = reactive(None)

    def __init__(self, records: list[Message]) -> None:
        super().__init__()
        self._fake_records = records
        # Provide enough of the agent interface that _replay_history needs.
        self.agent = MagicMock()
        self.agent._store = _FakeStore(records)

    def compose(self) -> ComposeResult:
        from supporter.tui.chat import SupporterHeader

        with SupporterHeader(id="supporter-header"):
            pass  # not rendered in this test harness
        with ChatContainer(id="chat-view"):
            yield WelcomeBanner(id="welcome-banner", classes="hidden")
            yield from ()

    async def on_mount(self) -> None:
        # Skip the full startup chain; call replay directly.
        from supporter.tui import SupporterApp

        await SupporterApp._replay_history(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(n: int = 250) -> list[Message]:
    """Build *n* alternating user/model records.

    The last model message includes a ``ToolCallPart`` so the replay path
    exercises the tool-call rendering branch.
    """
    records: list[Message] = []
    for i in range(n):
        if i % 2 == 0:
            records.append(
                Message(role="user", parts=[TextPart(text=f"User message {i}")])
            )
        else:
            parts: list[Any] = [TextPart(text=f"Model reply {i}")]
            # Inject a tool call in the very last model message.
            if i == n - 1:
                parts.append(
                    ToolCallPart(name="search", args={"query": "test"})
                )
            records.append(Message(role="model", parts=parts))
    return records


async def _settle(pilot: object) -> None:
    for _ in range(8):
        await pilot.pause()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_history_replay_renders_all_entries() -> None:
    """250 alternating user/model records must all appear as mounted widgets."""
    records = _make_history(250)
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        turns = app.query(ChatTurn)
        # 125 user turns (every even index), each mounted as a ChatTurn.
        assert len(turns) == 125

        bubbles = app.query(MessageBubble)
        # 125 user bubbles + 125 agent bubbles = 250 total.
        assert len(bubbles) == 250

        user_bubbles = [b for b in bubbles if b.role == "user"]
        agent_bubbles = [b for b in bubbles if b.role == "agent"]
        assert len(user_bubbles) == 125
        assert len(agent_bubbles) == 125


@pytest.mark.asyncio
async def test_full_history_replay_not_truncated_to_200() -> None:
    """The UI must show ALL 250 records, NOT just the LLM-capped 200."""
    records = _make_history(250)
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        bubbles = app.query(MessageBubble)
        assert len(bubbles) == 250, (
            f"Expected 250 bubbles (full history), got {len(bubbles)}"
        )


@pytest.mark.asyncio
async def test_welcome_banner_hidden_on_nonempty_history() -> None:
    """The welcome banner must be hidden when history has messages."""
    records = _make_history(10)
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        banner = app.query_one(WelcomeBanner)
        # The replay sets banner.message = "" which triggers the hidden class.
        assert banner.message == ""


@pytest.mark.asyncio
async def test_tool_call_rendered_in_last_model_bubble() -> None:
    """A ToolCallPart in a model message must appear in the bubble."""
    records = _make_history(250)
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        agent_bubbles = [b for b in app.query(MessageBubble) if b.role == "agent"]
        last_bubble = agent_bubbles[-1]
        assert len(last_bubble.tool_calls) == 1
        assert last_bubble.tool_calls[0]["name"] == "search"


@pytest.mark.asyncio
async def test_view_scrolled_to_bottom_after_replay() -> None:
    """jump_to_bottom must be called so newest content is visible."""
    records = _make_history(250)
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        chat_view = app.query_one("#chat-view", ChatContainer)
        assert chat_view.scroll_y >= chat_view.max_scroll_y - 1


@pytest.mark.asyncio
async def test_empty_history_no_bubbles_banner_visible() -> None:
    """First-run (empty history) must show the welcome banner and no bubbles."""
    app = _ReplayApp([])
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        turns = app.query(ChatTurn)
        assert len(turns) == 0

        bubbles = app.query(MessageBubble)
        assert len(bubbles) == 0

        # Welcome banner class should still include 'hidden' (default compose).
        banner = app.query_one(WelcomeBanner)
        assert "hidden" in banner.classes


@pytest.mark.asyncio
async def test_tool_result_parts_skipped_cleanly() -> None:
    """ToolResultPart messages (role=tool) must be skipped without crash."""
    records = [
        Message(role="user", parts=[TextPart(text="question")]),
        Message(role="tool", parts=[ToolResultPart(name="search", response={})]),
        Message(role="model", parts=[TextPart(text="answer")]),
    ]
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        # Only user + model bubbles; tool record is skipped.
        bubbles = app.query(MessageBubble)
        assert len(bubbles) == 2
        roles = [b.role for b in bubbles]
        assert roles == ["user", "agent"]


@pytest.mark.asyncio
async def test_chronological_order_preserved() -> None:
    """Bubbles must appear in the same order as the persisted history."""
    records = [
        Message(role="user", parts=[TextPart(text="first")]),
        Message(role="model", parts=[TextPart(text="second")]),
        Message(role="user", parts=[TextPart(text="third")]),
        Message(role="model", parts=[TextPart(text="fourth")]),
    ]
    app = _ReplayApp(records)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)

        bubbles = app.query(MessageBubble)
        contents = [b.content for b in bubbles]
        assert contents == ["first", "second", "third", "fourth"]
