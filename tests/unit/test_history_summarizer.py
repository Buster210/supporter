"""Unit tests for history_summarizer module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.history_summarizer import render_turns, summarize_turns


def _make_content(role: str, text: str | None = None) -> Any:
    """Create a mock Content with optional text part."""
    from google.genai.types import Content, Part

    if text is not None:
        return Content(role=role, parts=[Part(text=text)])
    return Content(role=role, parts=[])


def _make_function_call_part(name: str, args: dict[str, Any] | None = None) -> Any:
    """Create a mock function call part.

    WHY: text is explicitly set to None so render_turns correctly treats this
    as a non-text part (a bare MagicMock's .text is a truthy MagicMock).
    """
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = name
    part.function_call.args = args or {}
    part.function_response = None
    return part


def _make_function_response_part(
    name: str, response: dict[str, Any] | None = None
) -> Any:
    """Create a mock function response part.

    WHY: text is explicitly set to None so render_turns correctly treats this
    as a non-text part (a bare MagicMock's .text is a truthy MagicMock).
    """
    part = MagicMock()
    part.text = None
    part.function_call = None
    part.function_response = MagicMock()
    part.function_response.name = name
    part.function_response.response = response or {}
    return part


class TestRenderTurns:
    """Tests for render_turns function."""

    def test_render_empty_turns_returns_empty_string(self) -> None:
        result = render_turns([])
        assert result == ""

    def test_render_user_turn(self) -> None:
        turns = [_make_content("user", "Hello")]
        result = render_turns(turns)
        assert result == "User: Hello"

    def test_render_model_turn(self) -> None:
        turns = [_make_content("model", "Hi there")]
        result = render_turns(turns)
        assert result == "Assistant: Hi there"

    def test_render_multiple_turns(self) -> None:
        turns = [
            _make_content("user", "Question"),
            _make_content("model", "Answer"),
        ]
        result = render_turns(turns)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "User: Question" in lines[0]
        assert "Assistant: Answer" in lines[1]

    def test_render_skips_unknown_roles(self) -> None:
        other_turn = MagicMock()
        other_turn.role = "other"
        other_turn.parts = []
        turns = [_make_content("user", "Test"), other_turn]
        result = render_turns(turns)
        assert result == "User: Test"

    def test_render_skips_textless_turns(self) -> None:
        turns = [_make_content("user", None), _make_content("model", None)]
        result = render_turns(turns)
        assert result == ""

    def test_render_function_call_part(self) -> None:
        turn = MagicMock()
        turn.role = "model"
        turn.parts = [_make_function_call_part("get_weather", {"city": "SF"})]
        result = render_turns([turn])
        assert "[tool_call: get_weather(city='SF')]" in result

    def test_render_function_response_part(self) -> None:
        turn = MagicMock()
        turn.role = "user"
        turn.parts = [_make_function_response_part("get_weather", {"temp": 72})]
        result = render_turns([turn])
        assert "[tool_response: get_weather(temp=72)]" in result

    def test_render_mixed_parts(self) -> None:
        turn = MagicMock()
        turn.role = "model"
        turn.parts = [
            _make_function_call_part("search", {"query": "x"}),
            MagicMock(text="Some text", function_call=None, function_response=None),
        ]
        result = render_turns([turn])
        assert "[tool_call: search(query='x')]" in result
        assert "Some text" in result


class TestSummarizeTurns:
    """Tests for summarize_turns function."""

    @pytest.mark.asyncio
    async def test_summarize_empty_turns_returns_empty(self) -> None:
        result = await summarize_turns([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_no_api_keys_raises(self) -> None:
        with patch("supporter.history_summarizer.config") as mock_config:
            mock_config.gemini_api_keys = []
            with pytest.raises(RuntimeError, match="No Gemini API keys"):
                await summarize_turns([_make_content("user", "Hello")])

    @pytest.mark.asyncio
    async def test_summarize_calls_gemini_provider(self) -> None:
        mock_result = MagicMock()
        mock_result.text = "Summary text"

        with (
            patch("supporter.history_summarizer.config") as mock_config,
            patch(
                "supporter.providers.gemini_provider.GeminiProvider"
            ) as mock_provider_cls,
        ):
            mock_config.gemini_api_keys = ["test-key"]
            mock_config.gemini_model = "test-model"
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(return_value=mock_result)
            mock_provider_cls.return_value = mock_provider

            result = await summarize_turns([_make_content("user", "Hello")])

            assert result == "Summary text"
            mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_uses_low_temperature(self) -> None:
        mock_result = MagicMock()
        mock_result.text = "Summary"

        with (
            patch("supporter.history_summarizer.config") as mock_config,
            patch(
                "supporter.providers.gemini_provider.GeminiProvider"
            ) as mock_provider_cls,
        ):
            mock_config.gemini_api_keys = ["test-key"]
            mock_config.gemini_model = "test-model"
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(return_value=mock_result)
            mock_provider_cls.return_value = mock_provider

            await summarize_turns([_make_content("user", "Test")])

            call_kwargs = mock_provider.generate.call_args.args[1]
            assert call_kwargs["temperature"] == 0.2
