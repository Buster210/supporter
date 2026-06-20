from __future__ import annotations

from typing import Any

from supporter.tui.bubble import MessageBubble
from supporter.tui.constants import (
    RENDER_COALESCE_INTERVAL,
    STREAM_RENDER_MAX_INTERVAL,
)


def _make_immediate_timer(bubble: MessageBubble) -> None:
    def _set_timer(_interval: float, callback: Any) -> None:
        callback()

    bubble.set_timer = _set_timer  # type: ignore[assignment,method-assign]


def test_stream_200_rapid_tokens() -> None:
    bubble = MessageBubble(role="agent", content="", streaming=True)
    _make_immediate_timer(bubble)

    for _ in range(200):
        bubble.append_token("x")

    assert bubble.content == "x" * 200
    assert bubble.elements
    assert bubble.elements[-1]["type"] == "content"


def test_thought_content_interleaving_stable_sections() -> None:
    bubble = MessageBubble(role="agent", content="", streaming=True)
    _make_immediate_timer(bubble)

    bubble.append_token("thinking-1", is_thought=True)
    bubble.append_token("answer-1")
    bubble.append_token("thinking-2", is_thought=True)
    bubble.append_token("answer-2")

    types = [el["type"] for el in bubble.elements]
    assert types == ["thought", "content", "thought", "content"]
    assert bubble.thoughts == "thinking-1thinking-2"
    assert bubble.content == "answer-1answer-2"


def test_tool_calls_during_streaming_adds_tool_section() -> None:
    bubble = MessageBubble(role="agent", content="", streaming=True)
    _make_immediate_timer(bubble)

    bubble.append_token("start")
    bubble.add_tool_call("read_file", {"path": "workspace/file.txt"})
    bubble.append_token("end")

    tool_sections = [el for el in bubble.elements if el["type"] == "tool_calls"]
    assert len(tool_sections) == 1
    assert tool_sections[0]["calls"][0]["name"] == "read_file"
    assert bubble.content == "startend"


def test_finalize_collapses_last_thought_or_tool_section() -> None:
    bubble = MessageBubble(role="agent", content="", streaming=True)
    _make_immediate_timer(bubble)

    bubble.append_token("reasoning", is_thought=True)
    bubble.finalize(model="gemini", duration=0.5)

    assert bubble.streaming is False
    assert bubble.elements[-1]["type"] == "thought"
    assert bubble.elements[-1]["collapsed"] is True
    assert bubble.model == "gemini"


def test_render_interval_throttles_with_size_and_caps() -> None:
    # A short bubble renders at the base cadence; as content accumulates the
    # coalesce interval stretches (bounded duty cycle) and saturates at the cap,
    # so a tall streaming message can never starve the event loop.
    bubble = MessageBubble(role="agent", content="", streaming=True)

    assert bubble._render_interval() == RENDER_COALESCE_INTERVAL

    bubble.content = "x" * 1800
    assert bubble._render_interval() > RENDER_COALESCE_INTERVAL

    bubble.content = "x" * 100_000
    assert bubble._render_interval() == STREAM_RENDER_MAX_INTERVAL


def test_finalize_empty_streaming_is_safe() -> None:
    bubble = MessageBubble(role="agent", content="", streaming=True)
    _make_immediate_timer(bubble)

    bubble.finalize()

    assert bubble.streaming is False
    assert bubble.elements == []
