from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.providers.gemini_live_provider import (
    GeminiLiveProvider,
    _format_grounding_sources,
    _is_native_audio,
)


@pytest.fixture(autouse=True)
def mock_genai_client() -> Any:
    with patch("google.genai.Client") as mock:
        yield mock


@pytest.fixture
def api_keys() -> Any:
    return ["key1", "key2"]


@pytest.fixture
def provider(api_keys: Any) -> Any:
    return GeminiLiveProvider(api_keys=api_keys)


GEMINI_3_FLASH = "gemini-3.1-flash"
GEMMA_4_IT = "gemma-4-31b-it"


def test_is_native_audio_matches_case_insensitive() -> None:
    assert _is_native_audio("gemini-live-NATIVE-AUDIO-preview") is True
    assert _is_native_audio("gemini-3.1-flash-live") is False


def test_format_grounding_sources_empty_returns_empty_string() -> None:
    grounding = MagicMock()
    grounding.grounding_chunks = []
    assert _format_grounding_sources(grounding) == ""


def test_format_grounding_sources_skips_chunks_missing_web_or_uri() -> None:
    grounding = MagicMock()
    chunk_no_web = MagicMock()
    chunk_no_web.web = None
    chunk_empty_uri = MagicMock()
    chunk_empty_uri.web.uri = ""
    grounding.grounding_chunks = [chunk_no_web, chunk_empty_uri]
    assert _format_grounding_sources(grounding) == ""


def test_format_grounding_sources_title_fallback() -> None:
    grounding = MagicMock()
    chunk = MagicMock()
    chunk.web.uri = "https://example.com/page"
    chunk.web.title = None
    grounding.grounding_chunks = [chunk]
    result = _format_grounding_sources(grounding)
    assert "- Search Result: https://example.com/page" in result


@pytest.mark.asyncio
async def test_auto_register_search() -> None:
    p = GeminiLiveProvider(api_keys=["key"], model_name=GEMINI_3_FLASH)
    assert "google_search" in p.registry


def test_resolve_search_gemma() -> None:
    p = GeminiLiveProvider(api_keys=["key"], model_name=GEMMA_4_IT)
    assert any(hasattr(tool, "google_search") for tool in p._resolve_tools())


def test_resolve_search_no_duplicates() -> None:
    p = GeminiLiveProvider(api_keys=["key"], model_name=GEMMA_4_IT)
    assert sum(1 for t in p._resolve_tools() if hasattr(t, "google_search")) == 1


@pytest.mark.asyncio
async def test_ensure_session_with_rotation(api_keys: Any) -> None:
    p = GeminiLiveProvider(api_keys=api_keys)
    mock_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(
        side_effect=[Exception("429 Quota exhausted"), mock_session]
    )
    with patch.object(
        p.client.aio.live, "connect", return_value=mock_mgr
    ) as mock_connect:
        session = await p._ensure_session()
        assert session == mock_session
        assert p._current_key_index == 1
        assert mock_connect.call_count == 2


@pytest.mark.asyncio
async def test_ensure_session_fatal_error(api_keys: Any) -> None:
    p = GeminiLiveProvider(api_keys=api_keys)
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(side_effect=Exception("Fatal Auth Error"))
    with (
        patch.object(p.client.aio.live, "connect", return_value=mock_mgr),
        pytest.raises(Exception, match="Fatal Auth Error"),
    ):
        await p._ensure_session()


@pytest.mark.asyncio
async def test_handle_tool_call_variations(provider: Any) -> None:
    mock_session = AsyncMock()
    provider.registry["sync_tool"] = lambda x: f"hello {x}"

    async def async_tool(y: Any) -> Any:
        return {"val": y}

    provider.registry["async_tool"] = async_tool

    def error_tool() -> None:
        raise ValueError("Tool crash")

    provider.registry["error_tool"] = error_tool
    tool_call = MagicMock()
    tc1 = MagicMock()
    tc1.name = "sync_tool"
    tc1.args = {"x": "world"}
    tc1.id = "1"
    tc2 = MagicMock()
    tc2.name = "async_tool"
    tc2.args = {"y": 42}
    tc2.id = "2"
    tc3 = MagicMock()
    tc3.name = "error_tool"
    tc3.args = {}
    tc3.id = "3"
    tc4 = MagicMock()
    tc4.name = "missing"
    tc4.args = {}
    tc4.id = "4"
    tool_call.function_calls = [tc1, tc2, tc3, tc4]
    await provider._handle_tool_call(mock_session, tool_call)
    _, kwargs = mock_session.send_tool_response.call_args
    responses = kwargs["function_responses"]
    assert len(responses) == 4
    assert responses[0].response == {"result": "hello world"}


@pytest.mark.asyncio
async def test_generate_with_thoughts_and_grounding(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session
    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = None
    r1.go_away = None
    part1 = MagicMock()
    part1.text = "Thinking"
    part1.thought = True
    r1.server_content.model_turn.parts = [part1]
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = None
    r1.server_content.turn_complete = False
    r1.server_content.generation_complete = False

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.grounding_metadata = MagicMock()
    ot = MagicMock()
    ot.text = "Done"
    r2.server_content.output_transcription = ot
    r2.server_content.turn_complete = True
    r2.server_content.generation_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    result = await provider.generate("test")
    assert result.text == "Done"
    assert result.thoughts == "Thinking"


@pytest.mark.asyncio
async def test_generate_stream_with_tool_call(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session
    r1 = MagicMock()
    r1.tool_call = MagicMock()
    r1.session_resumption_update = None
    r1.go_away = None
    fc = MagicMock()
    fc.name = "my_tool"
    fc.args = {"a": 1}
    fc.id = "call_1"
    r1.tool_call.function_calls = [fc]
    r1.server_content = None

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    part2 = MagicMock()
    part2.text = "T"
    part2.thought = True
    r2.server_content.model_turn.parts = [part2]
    ot2 = MagicMock()
    ot2.text = "Stream"
    r2.server_content.output_transcription = ot2
    r2.server_content.turn_complete = True
    r2.server_content.generation_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    provider.registry["my_tool"] = AsyncMock(return_value="ok")
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert any(c.is_tool_call for c in chunks)
    assert any(c.text == "Stream" for c in chunks)


@pytest.mark.asyncio
async def test_generate_exception_handling(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    async def mock_receive_fail() -> AsyncGenerator[Any, Any]:
        yield MagicMock(server_content=None)
        raise Exception("Crash")

    mock_session.receive = mock_receive_fail
    result = await provider.generate("test")
    assert result.text == ""
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert chunks[-1].is_last is True


@pytest.mark.asyncio
async def test_close(provider: Any) -> None:
    mock_mgr = AsyncMock()
    provider._session_manager = mock_mgr
    await provider.close()
    mock_mgr.__aexit__.assert_called_once()
    assert provider._session is None


def test_get_name(provider: Any) -> None:
    assert "Live" in provider.get_name()


def test_native_audio_output_transcription() -> None:
    p = GeminiLiveProvider(
        api_keys=["key"], model_name="gemini-live-native-audio-preview"
    )
    cfg = p._get_session_config()
    assert cfg.output_audio_transcription is not None


def test_non_native_audio_no_transcription() -> None:
    p = GeminiLiveProvider(api_keys=["key"], model_name="gemini-3.1-flash-live-preview")
    cfg = p._get_session_config()
    assert not getattr(cfg, "output_audio_transcription", None)


def test_thinking_config_included_when_include_thoughts_true() -> None:
    p = GeminiLiveProvider(api_keys=["key"], include_thoughts=True)
    cfg = p._get_session_config()
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.include_thoughts is True


def test_thinking_config_absent_when_include_thoughts_false() -> None:
    p = GeminiLiveProvider(api_keys=["key"], include_thoughts=False)
    cfg = p._get_session_config()
    assert not getattr(cfg, "thinking_config", None)


def test_thinking_config_fallback_to_medium(monkeypatch: Any) -> None:
    from supporter.config import config as supporter_config

    monkeypatch.setattr(supporter_config, "live_thinking_level", "NONEXISTENT_LEVEL")
    p = GeminiLiveProvider(api_keys=["key"], include_thoughts=True)
    cfg = p._get_session_config()
    from google.genai import types

    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM


@pytest.mark.asyncio
async def test_warmup_success(provider: Any) -> None:
    mock_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=mock_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider.warmup()
    assert provider._session == mock_session


@pytest.mark.asyncio
async def test_warmup_failure_logged(provider: Any) -> None:
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(side_effect=Exception("boom"))
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider.warmup()


@pytest.mark.asyncio
async def test_ensure_session_exhausts_all_keys_raises_last_error() -> None:
    p = GeminiLiveProvider(api_keys=["k1", "k2"])
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(side_effect=Exception("quota exhausted"))
    with (
        patch.object(p.client.aio.live, "connect", return_value=mock_mgr),
        pytest.raises(Exception, match="quota exhausted"),
    ):
        await p._ensure_session()


@pytest.mark.asyncio
async def test_drain_session_breaks_on_turn_complete(provider: Any) -> None:
    mock_session = AsyncMock()
    r = MagicMock()
    r.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, None]:
        yield r

    mock_session.receive = mock_receive
    await provider._drain_session(mock_session)


@pytest.mark.asyncio
async def test_handle_tool_call_empty_function_calls(provider: Any) -> None:
    mock_session = AsyncMock()
    tool_call = MagicMock()
    tool_call.function_calls = []
    await provider._handle_tool_call(mock_session, tool_call)
    mock_session.send_tool_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_tool_call_branch(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r1 = MagicMock()
    r1.tool_call = MagicMock()
    fc = MagicMock()
    fc.name = "noop"
    fc.args = {}
    fc.id = "x1"
    r1.tool_call.function_calls = [fc]
    r1.session_resumption_update = None
    r1.go_away = None
    r1.server_content = None

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.output_transcription = None
    r2.server_content.grounding_metadata = None
    r2.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    provider.registry["noop"] = lambda: "ok"
    result = await provider.generate("hi")
    mock_session.send_tool_response.assert_called_once()
    assert result.text == ""


@pytest.mark.asyncio
async def test_generate_session_resumption_update(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = MagicMock()
    r1.session_resumption_update.new_handle = "handle-xyz"
    r1.go_away = None
    r1.server_content = None

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.output_transcription = None
    r2.server_content.grounding_metadata = None
    r2.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    await provider.generate("hi")
    assert provider._session_handle == "handle-xyz"


@pytest.mark.asyncio
async def test_generate_go_away_branch(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.server_content = None

    r_final = MagicMock()
    r_final.tool_call = None
    r_final.session_resumption_update = None
    r_final.go_away = None
    r_final.server_content.model_turn = None
    r_final.server_content.output_transcription = None
    r_final.server_content.grounding_metadata = None
    r_final.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r_go
        yield r_final

    mock_session.receive = mock_receive

    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        result = await provider.generate("hi")
    assert result.text == ""


@pytest.mark.asyncio
async def test_generate_no_server_content_skipped(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r_empty = MagicMock()
    r_empty.tool_call = None
    r_empty.session_resumption_update = None
    r_empty.go_away = None
    r_empty.server_content = None

    r_final = MagicMock()
    r_final.tool_call = None
    r_final.session_resumption_update = None
    r_final.go_away = None
    r_final.server_content.model_turn = None
    r_final.server_content.output_transcription = None
    r_final.server_content.grounding_metadata = None
    r_final.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r_empty
        yield r_final

    mock_session.receive = mock_receive
    result = await provider.generate("hi")
    assert result.text == ""


@pytest.mark.asyncio
async def test_generate_model_turn_non_thought_text(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = None
    r1.go_away = None
    part = MagicMock()
    part.text = "Hello"
    part.thought = False
    r1.server_content.model_turn.parts = [part]
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = None
    r1.server_content.turn_complete = False

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.output_transcription = None
    r2.server_content.grounding_metadata = None
    r2.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    result = await provider.generate("hi")
    assert "Hello" in result.text


@pytest.mark.asyncio
async def test_generate_stream_go_away_branch(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.server_content = None

    r_final = MagicMock()
    r_final.tool_call = None
    r_final.session_resumption_update = None
    r_final.go_away = None
    r_final.server_content.model_turn = None
    r_final.server_content.output_transcription = None
    r_final.server_content.grounding_metadata = None
    r_final.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r_go
        yield r_final

    mock_session.receive = mock_receive

    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        chunks = []
        async for chunk in provider.generate_stream("hi"):
            chunks.append(chunk)
    assert chunks[-1].is_last is True


@pytest.mark.asyncio
async def test_generate_stream_model_turn_non_thought_text(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = None
    r1.go_away = None
    part = MagicMock()
    part.text = "World"
    part.thought = False
    r1.server_content.model_turn.parts = [part]
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = None
    r1.server_content.turn_complete = False

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.output_transcription = None
    r2.server_content.grounding_metadata = None
    r2.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    chunks = []
    async for chunk in provider.generate_stream("hi"):
        chunks.append(chunk)
    assert any(c.text == "World" for c in chunks)


@pytest.mark.asyncio
async def test_generate_stream_grounding_sources_chunk(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    grounding = MagicMock()
    web_chunk = MagicMock()
    web_chunk.web.uri = "https://example.com"
    web_chunk.web.title = "Example"
    grounding.grounding_chunks = [web_chunk]

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = None
    r1.go_away = None
    r1.server_content.model_turn = None
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = grounding
    r1.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1

    mock_session.receive = mock_receive
    chunks = []
    async for chunk in provider.generate_stream("hi"):
        chunks.append(chunk)
    assert chunks[-1].is_last is True
    assert "SOURCES FOUND:" in chunks[-2].text
    assert chunks[-2].raw is grounding


@pytest.mark.asyncio
async def test_generate_appends_grounding_sources_and_sets_raw(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    grounding = MagicMock()
    web_chunk = MagicMock()
    web_chunk.web.uri = "https://example.com/result"
    web_chunk.web.title = "Result Title"
    grounding.grounding_chunks = [web_chunk]

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = None
    r1.go_away = None
    r1.server_content.model_turn = None
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = grounding
    r1.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1

    mock_session.receive = mock_receive
    result = await provider.generate("hi")
    assert "SOURCES FOUND:" in result.text
    assert "https://example.com/result" in result.text
    assert "Result Title" in result.text
    assert result.raw is grounding


@pytest.mark.asyncio
async def test_generate_stream_emits_tool_call_chunk(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    fc = MagicMock()
    fc.name = "stub_tool"
    fc.args = {"key": "val"}
    fc.id = "tc_1"

    r1 = MagicMock()
    r1.tool_call = MagicMock()
    r1.tool_call.function_calls = [fc]
    r1.session_resumption_update = None
    r1.go_away = None
    r1.server_content = None

    r2 = MagicMock()
    r2.tool_call = None
    r2.session_resumption_update = None
    r2.go_away = None
    r2.server_content.model_turn = None
    r2.server_content.output_transcription = None
    r2.server_content.grounding_metadata = None
    r2.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r1
        yield r2

    mock_session.receive = mock_receive
    provider.registry["stub_tool"] = lambda key: {"result": key}

    chunks = []
    async for chunk in provider.generate_stream("hi"):
        chunks.append(chunk)

    tool_chunks = [c for c in chunks if c.is_tool_call]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_name == "stub_tool"
    assert tool_chunks[0].tool_args == {"key": "val"}
    assert chunks[-1].is_last is True
