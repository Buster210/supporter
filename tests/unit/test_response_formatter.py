"""Tests for G4: format_response, MessageBubble.replace_content, and
_maybe_format_bubble wiring in ChatMessageProcessor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from supporter import worker
from supporter.tui.bubble import MessageBubble

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str, options: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self._response)


class _FailingProvider:
    async def generate(self, prompt: str, options: Any) -> SimpleNamespace:
        raise RuntimeError("model unavailable")


def _patch_provider(provider: Any) -> Any:
    return patch("supporter.worker.get_provider", return_value=provider)


def _make_bubble(tokens: list[str]) -> MessageBubble:
    """Build a finalized pure-text bubble without mounting it."""
    bubble = MessageBubble(role="agent", content="", streaming=True)
    # Suppress timer so append_token doesn't try to schedule on an unmounted widget.
    bubble.set_timer = lambda _interval, callback: None  # type: ignore[method-assign,misc,assignment]
    for tok in tokens:
        bubble.append_token(tok)
    # Simulate finalize without UI (no _meta_label / _message_view attached).
    bubble.streaming = False
    return bubble


# ---------------------------------------------------------------------------
# format_response
# ---------------------------------------------------------------------------


class TestFormatResponse:
    @pytest.mark.asyncio
    async def test_returns_stripped_formatted_text(self) -> None:
        with _patch_provider(_FakeProvider("  # Hi\n")):
            result = await worker.format_response("# Hi  \n", "test-model")
        assert result == "# Hi"

    @pytest.mark.asyncio
    async def test_empty_provider_response_returns_empty(self) -> None:
        with _patch_provider(_FakeProvider("")):
            result = await worker.format_response("some text", "test-model")
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        with _patch_provider(_FakeProvider("")):
            result = await worker.format_response("", "test-model")
        assert result == ""

    @pytest.mark.asyncio
    async def test_provider_raises_returns_empty_never_raises(self) -> None:
        with _patch_provider(_FailingProvider()):
            result = await worker.format_response("hello", "test-model")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_response_returns_empty(self) -> None:
        with _patch_provider(_FakeProvider("   \n  ")):
            result = await worker.format_response("text", "test-model")
        assert result == ""


# ---------------------------------------------------------------------------
# MessageBubble.replace_content
# ---------------------------------------------------------------------------


class TestReplaceContent:
    def test_replaces_content_and_single_element(self) -> None:
        bubble = _make_bubble(["Hello ", "world"])
        bubble.replace_content("new formatted")

        assert bubble.content == "new formatted"
        content_els = [el for el in bubble.elements if el["type"] == "content"]
        assert len(content_els) == 1
        assert content_els[0]["content"] == "new formatted"

    def test_blank_new_text_is_noop(self) -> None:
        bubble = _make_bubble(["original text"])
        bubble.replace_content("   ")

        assert bubble.content == "original text"
        content_els = [el for el in bubble.elements if el["type"] == "content"]
        assert len(content_els) == 1
        assert content_els[0]["content"] == "original text"

    def test_empty_new_text_is_noop(self) -> None:
        bubble = _make_bubble(["original"])
        bubble.replace_content("")
        assert bubble.content == "original"

    def test_preserves_non_content_elements(self) -> None:
        bubble = MessageBubble(role="agent", content="", streaming=True)
        bubble.set_timer = lambda _i, _c: None  # type: ignore[method-assign,misc,assignment]
        bubble.append_token("thinking", is_thought=True)
        bubble.append_token("answer text")
        bubble.streaming = False

        bubble.replace_content("reformatted answer")

        types = [el["type"] for el in bubble.elements]
        assert "thought" in types
        assert types.count("content") == 1
        assert bubble.content == "reformatted answer"

    def test_recheck_markdown_flag_set(self) -> None:
        bubble = _make_bubble(["plain text"])
        bubble.replace_content("## Heading\n\n- item")
        content_el = next(el for el in bubble.elements if el["type"] == "content")
        assert content_el.get("_recheck_markdown") is True


# ---------------------------------------------------------------------------
# _maybe_format_bubble wiring
# ---------------------------------------------------------------------------


class TestMaybeFormatBubble:
    """Test _maybe_format_bubble via ChatMessageProcessor directly."""

    def _make_processor(self) -> Any:
        from supporter.tui.message_processor import ChatMessageProcessor

        class _FakeApp:
            status_label = "Idle"

        return ChatMessageProcessor(_FakeApp())

    @pytest.mark.asyncio
    async def test_no_model_never_calls_format_response(self) -> None:
        processor = self._make_processor()
        bubble = _make_bubble(["some text"])
        called = False

        async def _sentinel(text: str, model: str) -> str:
            nonlocal called
            called = True
            return "formatted"

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = None
            with patch("supporter.worker.format_response", side_effect=_sentinel):
                await processor._maybe_format_bubble(bubble)

        assert not called

    @pytest.mark.asyncio
    async def test_with_model_and_pure_text_calls_format_and_replaces(self) -> None:
        processor = self._make_processor()
        bubble = _make_bubble(["raw text"])

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = "fallback-model"
            with patch(
                "supporter.worker.format_response", new_callable=AsyncMock
            ) as mock_fmt:
                mock_fmt.return_value = "clean text"
                await processor._maybe_format_bubble(bubble)

        mock_fmt.assert_called_once_with("raw text", "fallback-model")
        assert bubble.content == "clean text"

    @pytest.mark.asyncio
    async def test_mixed_bubble_formatted_preserving_tool_calls(self) -> None:
        """Mixed bubbles are now formatted; tool_calls element survives."""
        processor = self._make_processor()
        bubble = _make_bubble(["some text"])
        # Inject a tool_calls element to simulate a mixed bubble.
        tool_el = {
            "type": "tool_calls",
            "calls": [{"name": "read_file", "args": {}}],
            "collapsed": False,
            "manually_interacted": False,
        }
        bubble.elements.append(tool_el)

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = "fallback-model"
            with patch(
                "supporter.worker.format_response", new_callable=AsyncMock
            ) as mock_fmt:
                mock_fmt.return_value = "clean text"
                await processor._maybe_format_bubble(bubble)

        mock_fmt.assert_called_once_with("some text", "fallback-model")
        assert bubble.content == "clean text"
        assert tool_el in bubble.elements

    @pytest.mark.asyncio
    async def test_empty_content_bubble_skipped(self) -> None:
        processor = self._make_processor()
        bubble = _make_bubble([])  # no tokens → content == ""
        called = False

        async def _sentinel(text: str, model: str) -> str:
            nonlocal called
            called = True
            return "x"

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = "fallback-model"
            with patch("supporter.worker.format_response", side_effect=_sentinel):
                await processor._maybe_format_bubble(bubble)

        assert not called

    @pytest.mark.asyncio
    async def test_format_response_same_text_no_replace(self) -> None:
        """If formatted == original, replace_content must not be called."""
        processor = self._make_processor()
        bubble = _make_bubble(["identical"])

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = "fallback-model"
            with patch(
                "supporter.worker.format_response", new_callable=AsyncMock
            ) as mock_fmt:
                mock_fmt.return_value = "identical"
                original_elements = list(bubble.elements)
                await processor._maybe_format_bubble(bubble)

        assert bubble.content == "identical"
        assert bubble.elements == original_elements

    @pytest.mark.asyncio
    async def test_format_response_exception_does_not_raise(self) -> None:
        """Formatter failure must never propagate — bubble stays raw."""
        processor = self._make_processor()
        bubble = _make_bubble(["raw"])

        async def _boom(text: str, model: str) -> str:
            raise RuntimeError("network error")

        with patch("supporter.config.config") as mock_cfg:
            mock_cfg.gemini_fallback_model = "fallback-model"
            with patch("supporter.worker.format_response", side_effect=_boom):
                # Must not raise.
                await processor._maybe_format_bubble(bubble)

        assert bubble.content == "raw"
