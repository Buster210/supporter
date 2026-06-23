"""Unit tests for the OpenRouter provider — mocked httpx, no network."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.llm.types import GenOptions, Message, TextPart
from supporter.providers.openrouter_provider import OpenRouterProvider
from supporter.providers.registry import PROVIDER_FACTORIES
from supporter.types import LLMChunk, LLMResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROVIDER = OpenRouterProvider(api_key="test-key", model_name="test-model")


def _sse_lines(*chunks: dict[str, Any]) -> list[str]:
    """Build SSE lines from a sequence of chunk dicts, ending with [DONE]."""
    lines = []
    for c in chunks:
        lines.append(f"data: {json.dumps(c)}")
    lines.append("data: [DONE]")
    return lines


def _make_delta(content: str) -> dict[str, Any]:
    return {
        "choices": [{"delta": {"content": content}}],
        "model": "test-model",
    }


def _mock_response(json_data: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _mock_stream(lines: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()

    async def _aiter() -> AsyncIterator[str]:
        for line in lines:
            yield line

    resp.aiter_lines = _aiter
    return resp


# ---------------------------------------------------------------------------
# _messages_to_openai
# ---------------------------------------------------------------------------


def test_messages_to_openai_str_prompt() -> None:
    from supporter.providers.openrouter_provider import _messages_to_openai

    assert _messages_to_openai("hello") == [{"role": "user", "content": "hello"}]


def test_messages_to_openai_neutral_messages() -> None:
    from supporter.providers.openrouter_provider import _messages_to_openai

    msgs = [
        Message(role="user", parts=[TextPart(text="hi")]),
        Message(role="model", parts=[TextPart(text="yo")]),
    ]
    result = _messages_to_openai(msgs)
    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


async def test_generate_returns_llm_result() -> None:
    body = {"choices": [{"message": {"content": "pong"}}], "model": "test-model"}
    with patch("supporter.providers.openrouter_provider.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.post.return_value = _mock_response(body)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = client

        result = await PROVIDER.generate("ping")

    assert isinstance(result, LLMResult)
    assert result.text == "pong"
    assert result.model == "test-model"


async def test_generate_with_options() -> None:
    body = {"choices": [{"message": {"content": "ok"}}], "model": "m"}
    with patch("supporter.providers.openrouter_provider.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.post.return_value = _mock_response(body)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = client

        opts = GenOptions(temperature=0.5, max_output_tokens=100)
        result = await PROVIDER.generate("q", options=opts)

    assert result.text == "ok"
    payload = client.post.call_args[1]["json"]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


async def test_generate_stream_yields_chunks() -> None:
    lines = _sse_lines(_make_delta("hello "), _make_delta("world"))
    with patch("supporter.providers.openrouter_provider.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        # Create a proper context manager for client.stream()
        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=_mock_stream(lines))
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(return_value=stream_ctx)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = client

        chunks: list[LLMChunk] = []
        async for chunk in PROVIDER.generate_stream("hi"):
            chunks.append(chunk)

    # Two content chunks + final sentinel
    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["hello ", "world", ""]
    assert chunks[-1].is_last is True
    assert all(c.is_last is False for c in chunks[:-1])


async def test_generate_stream_handles_done_marker() -> None:
    lines = _sse_lines(_make_delta("a"))
    with patch("supporter.providers.openrouter_provider.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=_mock_stream(lines))
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(return_value=stream_ctx)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = client

        chunks = [c async for c in PROVIDER.generate_stream("x")]

    assert chunks[0].text == "a"
    assert chunks[-1].is_last is True


async def test_generate_stream_skips_non_data_lines() -> None:
    lines = ["", ": comment", "data: [DONE]"]
    with patch("supporter.providers.openrouter_provider.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        stream_ctx = AsyncMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=_mock_stream(lines))
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(return_value=stream_ctx)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        mock_cls.return_value = client

        chunks = [c async for c in PROVIDER.generate_stream("x")]

    # Only the final sentinel
    assert len(chunks) == 1
    assert chunks[0].is_last is True


# ---------------------------------------------------------------------------
# get_name()
# ---------------------------------------------------------------------------


def test_get_name() -> None:
    assert PROVIDER.get_name() == "test-model"


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_openrouter_factory_registered() -> None:
    assert "openrouter" in PROVIDER_FACTORIES


def test_openrouter_factory_rejects_live() -> None:
    with pytest.raises(ValueError, match="Live"):
        PROVIDER_FACTORIES["openrouter"](keys=["k"], model_name="m", live=True)


def test_openrouter_factory_rejects_missing_key() -> None:
    from supporter.providers.registry import _openrouter_factory

    # Patch at the location where the function uses it
    with patch("supporter.pool.config") as mock_config:
        mock_config.openrouter_api_key = None
        mock_config.openrouter_model = "test"
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            _openrouter_factory(keys=[])
