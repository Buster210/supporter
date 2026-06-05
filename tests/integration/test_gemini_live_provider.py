import asyncio
import time
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
async def test_generate_go_away_defers_reconnect_preserving_content(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session
    provider._session_manager = AsyncMock()

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.go_away.time_left = "5s"
    r_go.server_content = None

    r_content = MagicMock()
    r_content.tool_call = None
    r_content.session_resumption_update = None
    r_content.go_away = None
    part = MagicMock()
    part.text = "after-go-away"
    part.thought = False
    r_content.server_content.model_turn.parts = [part]
    r_content.server_content.output_transcription = None
    r_content.server_content.grounding_metadata = None
    r_content.server_content.turn_complete = False

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
        yield r_content
        yield r_final

    mock_session.receive = mock_receive

    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        result = await provider.generate("hi")
        assert "after-go-away" in result.text
        assert provider._reconnect_pending is False
        assert provider._prewarm_task is not None
        await provider._consume_prewarm()
    assert provider._session is new_session
    assert provider._prewarm_task is None


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
async def test_generate_stream_go_away_defers_reconnect_preserving_content(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session
    provider._session_manager = AsyncMock()

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.go_away.time_left = "5s"
    r_go.server_content = None

    r_content = MagicMock()
    r_content.tool_call = None
    r_content.session_resumption_update = None
    r_content.go_away = None
    part = MagicMock()
    part.text = "after-go-away"
    part.thought = False
    r_content.server_content.model_turn.parts = [part]
    r_content.server_content.output_transcription = None
    r_content.server_content.grounding_metadata = None
    r_content.server_content.turn_complete = False

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
        yield r_content
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
        assert any(c.text == "after-go-away" for c in chunks)
        assert provider._reconnect_pending is False
        assert provider._prewarm_task is not None
        await provider._consume_prewarm()
    assert provider._session is new_session
    assert provider._prewarm_task is None


@pytest.mark.asyncio
async def test_close_resets_last_turn_complete(provider: Any) -> None:
    provider._last_turn_complete = False
    provider._session_manager = AsyncMock()
    await provider.close()
    assert provider._last_turn_complete is True


@pytest.mark.asyncio
async def test_close_cancels_pending_prewarm(provider: Any) -> None:
    async def never_finish() -> None:
        await asyncio.sleep(3600)

    provider._prewarm_task = asyncio.create_task(never_finish())
    await provider.close()
    assert provider._prewarm_task is None


@pytest.mark.asyncio
async def test_close_retrieves_failed_prewarm_exception(provider: Any) -> None:
    async def boom() -> None:
        raise RuntimeError("prewarm exploded")

    task = asyncio.create_task(boom())
    await asyncio.sleep(0)
    provider._prewarm_task = task
    await provider.close()
    assert provider._prewarm_task is None
    assert task.exception() is not None


@pytest.mark.asyncio
async def test_consume_prewarm_swallows_failure(provider: Any) -> None:
    async def boom() -> None:
        raise RuntimeError("prewarm exploded")

    provider._prewarm_task = asyncio.create_task(boom())
    await provider._consume_prewarm()
    assert provider._prewarm_task is None


@pytest.mark.asyncio
async def test_prepare_turn_stashes_history_from_options(provider: Any) -> None:
    provider._session = AsyncMock()
    history = [MagicMock()]
    await provider._prepare_turn("hi", {"history": history})
    assert provider._history == history
    assert provider._history is not history


@pytest.mark.asyncio
async def test_prepare_turn_reconnects_when_pending(provider: Any) -> None:
    old_session = AsyncMock()
    old_mgr = AsyncMock()
    provider._session = old_session
    provider._session_manager = old_mgr
    provider._reconnect_pending = True

    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        session = await provider._prepare_turn("hi")
    old_mgr.__aexit__.assert_called_once()
    assert session is new_session
    assert provider._reconnect_pending is False


@pytest.mark.asyncio
async def test_ensure_session_flags_replay_when_handleless(provider: Any) -> None:
    provider._session_handle = None
    provider._history = [MagicMock(), MagicMock()]
    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider._ensure_session()
    assert provider._needs_replay is True
    new_session.send_client_content.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_session_no_replay_flag_when_handle_present(provider: Any) -> None:
    provider._session_handle = "existing-handle"
    provider._history = [MagicMock()]
    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider._ensure_session()
    assert provider._needs_replay is False


def _turn(role: str, text: str) -> Any:
    from google.genai.types import Content, Part

    return Content(role=role, parts=[Part(text=text)])


def _texts(turns: Any) -> list[str]:
    return ["".join(p.text or "" for p in t.parts) for t in turns]


@pytest.mark.asyncio
async def test_send_user_turn_replays_history_with_prompt(provider: Any) -> None:
    provider._needs_replay = True
    provider._history = [_turn("user", "my number is 42"), _turn("model", "noted, 42")]
    session = AsyncMock()
    await provider._send_user_turn(session, "what is my number?")

    session.send_realtime_input.assert_not_called()
    session.send_client_content.assert_called_once()
    _, kwargs = session.send_client_content.call_args
    assert kwargs["turn_complete"] is True
    turns = kwargs["turns"]
    assert all(t.role == "user" for t in turns)
    assert _texts(turns) == [
        "my number is 42",
        "(Assistant earlier said: noted, 42)",
        "what is my number?",
    ]
    assert provider._needs_replay is False


@pytest.mark.asyncio
async def test_send_user_turn_normal_turn_uses_realtime(provider: Any) -> None:
    provider._needs_replay = False
    provider._history = [_turn("user", "earlier")]
    session = AsyncMock()
    await provider._send_user_turn(session, "hello")
    session.send_client_content.assert_not_called()
    session.send_realtime_input.assert_awaited_once_with(text="hello")


@pytest.mark.asyncio
async def test_send_user_turn_no_history_uses_realtime(provider: Any) -> None:
    provider._needs_replay = True
    provider._history = []
    session = AsyncMock()
    await provider._send_user_turn(session, "hello")
    session.send_client_content.assert_not_called()
    session.send_realtime_input.assert_awaited_once_with(text="hello")
    assert provider._needs_replay is False


@pytest.mark.asyncio
async def test_replay_turns_relabels_model_and_drops_nontext(provider: Any) -> None:
    from google.genai.types import Content, Part

    nontext = Content(role="user", parts=[Part(function_response=None)])
    provider._history = [
        _turn("user", "hi"),
        _turn("model", "hello"),
        nontext,
    ]
    turns = provider._replay_turns()
    assert all(t.role == "user" for t in turns)
    assert _texts(turns) == ["hi", "(Assistant earlier said: hello)"]


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


STALE_HANDLE_ERROR = Exception("1008: BidiGenerateContent session not found")
INVALID_HANDLE_ERROR = Exception("1007 None. Invalid session handle")


def _mgr_yielding(*results: Any) -> Any:
    mgr = MagicMock()
    mgr.__aenter__ = AsyncMock(side_effect=list(results))
    mgr.__aexit__ = AsyncMock(return_value=None)
    return mgr


@pytest.mark.asyncio
async def test_stale_handle_dropped_and_history_replayed(provider: Any) -> None:
    provider._session_handle = "stale-handle"
    provider._history = [MagicMock(), MagicMock()]
    new_session = AsyncMock()
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        session = await provider._ensure_session()
    assert session is new_session
    assert provider._session_handle is None
    assert provider._needs_replay is True
    new_session.send_client_content.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_handle_dropped_and_recovered(provider: Any) -> None:
    provider._session_handle = "malformed-handle"
    provider._history = [MagicMock()]
    new_session = AsyncMock()
    mgr = _mgr_yielding(INVALID_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        session = await provider._ensure_session()
    assert session is new_session
    assert provider._session_handle is None
    assert provider._needs_replay is True
    assert provider._current_key_index == 0


@pytest.mark.asyncio
async def test_non_handle_1007_does_not_drop_handle() -> None:
    with patch("google.genai.Client"):
        p = GeminiLiveProvider(api_keys=["only-key"])
    p._session_handle = "good-handle"
    p._history = [MagicMock()]
    modality_error = Exception("1007 None. TEXT is not supported by the model")
    mgr = _mgr_yielding(modality_error)
    with (
        patch.object(p.client.aio.live, "connect", return_value=mgr),
        pytest.raises(Exception, match="TEXT is not supported"),
    ):
        await p._ensure_session()
    assert p._session_handle == "good-handle"
    assert mgr.__aenter__.call_count == 1


@pytest.mark.asyncio
async def test_stale_handle_recovery_does_not_burn_a_key(provider: Any) -> None:
    provider._session_handle = "stale-handle"
    provider._history = [MagicMock()]
    assert provider._current_key_index == 0
    new_session = AsyncMock()
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        await provider._ensure_session()
    assert provider._current_key_index == 0


@pytest.mark.asyncio
async def test_stale_handle_recovers_with_single_api_key() -> None:
    with patch("google.genai.Client"):
        p = GeminiLiveProvider(api_keys=["only-key"])
    p._session_handle = "stale-handle"
    p._history = [MagicMock()]
    new_session = AsyncMock()
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(p.client.aio.live, "connect", return_value=mgr):
        session = await p._ensure_session()
    assert session is new_session
    assert p._session_handle is None
    assert p._needs_replay is True


@pytest.mark.asyncio
async def test_stale_handle_dropped_only_once_no_infinite_loop(provider: Any) -> None:
    provider._session_handle = "stale-handle"
    provider._history = [MagicMock()]
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, STALE_HANDLE_ERROR, STALE_HANDLE_ERROR)
    with (
        patch.object(provider.client.aio.live, "connect", return_value=mgr),
        pytest.raises(Exception, match="session not found"),
    ):
        await provider._ensure_session()
    assert provider._session_handle is None
    assert mgr.__aenter__.call_count == 3


@pytest.mark.asyncio
async def test_stale_handle_without_history_reconnects_clean(provider: Any) -> None:
    provider._session_handle = "stale-handle"
    provider._history = []
    new_session = AsyncMock()
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        session = await provider._ensure_session()
    assert session is new_session
    assert provider._session_handle is None
    new_session.send_client_content.assert_not_called()


@pytest.mark.asyncio
async def test_non_stale_error_preserves_handle_and_rotates(provider: Any) -> None:
    provider._session_handle = "valid-handle"
    new_session = AsyncMock()
    mgr = _mgr_yielding(Exception("429 Quota exhausted"), new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        session = await provider._ensure_session()
    assert session is new_session
    assert provider._session_handle == "valid-handle"
    assert provider._current_key_index == 1


@pytest.mark.asyncio
async def test_fatal_error_preserves_handle(provider: Any) -> None:
    provider._session_handle = "valid-handle"
    mgr = _mgr_yielding(Exception("Fatal Auth Error"))
    with (
        patch.object(provider.client.aio.live, "connect", return_value=mgr),
        pytest.raises(Exception, match="Fatal Auth Error"),
    ):
        await provider._ensure_session()
    assert provider._session_handle == "valid-handle"


@pytest.mark.asyncio
async def test_replay_failure_falls_back_to_realtime(provider: Any) -> None:
    provider._needs_replay = True
    provider._history = [_turn("user", "earlier")]
    session = AsyncMock()
    session.send_client_content = AsyncMock(side_effect=RuntimeError("send died"))
    await provider._send_user_turn(session, "hello")
    session.send_client_content.assert_awaited_once()
    session.send_realtime_input.assert_awaited_once_with(text="hello")
    assert provider._needs_replay is False


@pytest.mark.asyncio
async def test_handle_not_stored_when_not_resumable(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r1 = MagicMock()
    r1.tool_call = None
    r1.session_resumption_update = MagicMock()
    r1.session_resumption_update.resumable = False
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
    assert provider._session_handle is None


def _completed_turn_response() -> Any:
    r = MagicMock()
    r.tool_call = None
    r.session_resumption_update = None
    r.go_away = None
    r.server_content.model_turn = None
    r.server_content.output_transcription = None
    r.server_content.grounding_metadata = None
    r.server_content.turn_complete = True
    return r


@pytest.mark.asyncio
async def test_go_away_then_stale_reconnect_recovers_end_to_end(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session
    provider._session_manager = AsyncMock()
    provider._session_handle = "stale-handle"
    turn1_history = [_turn("user", "turn one")]
    provider._history = list(turn1_history)

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.go_away.time_left = "5s"
    r_go.server_content = None

    async def turn1_receive() -> AsyncGenerator[Any, Any]:
        yield r_go
        yield _completed_turn_response()

    mock_session.receive = turn1_receive

    new_session = AsyncMock()

    async def turn2_receive() -> AsyncGenerator[Any, Any]:
        yield _completed_turn_response()

    new_session.receive = turn2_receive

    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        await provider.generate("turn one")
        assert provider._prewarm_task is not None
        await provider._consume_prewarm()
        assert provider._session is new_session
        assert provider._session_handle is None
        assert provider._needs_replay is True
        new_session.send_client_content.assert_not_called()

        fresh_history = [*turn1_history, _turn("model", "ok"), _turn("user", "interim")]
        await provider.generate("turn two", {"history": fresh_history})

    new_session.send_client_content.assert_called_once()
    new_session.send_realtime_input.assert_not_called()
    _, kwargs = new_session.send_client_content.call_args
    assert kwargs["turn_complete"] is True
    turns = kwargs["turns"]
    assert all(t.role == "user" for t in turns)
    assert _texts(turns) == [
        "turn one",
        "(Assistant earlier said: ok)",
        "interim",
        "turn two",
    ]
    assert provider._needs_replay is False


@pytest.mark.asyncio
async def test_generate_transport_drop_sets_reconnect_pending(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    async def mock_receive_crash() -> AsyncGenerator[Any, Any]:
        yield MagicMock(server_content=None)
        raise ConnectionError("network changed")

    mock_session.receive = mock_receive_crash
    result = await provider.generate("hi")
    assert result.text == ""
    assert provider._last_turn_complete is True
    assert provider._prewarm_task is not None


@pytest.mark.asyncio
async def test_generate_stream_transport_drop_sets_reconnect_pending(
    provider: Any,
) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    async def mock_receive_crash() -> AsyncGenerator[Any, Any]:
        yield MagicMock(server_content=None)
        raise ConnectionError("network lost")

    mock_session.receive = mock_receive_crash
    chunks = []
    async for chunk in provider.generate_stream("hi"):
        chunks.append(chunk)
    assert chunks[-1].is_last is True
    assert provider._last_turn_complete is True
    assert provider._prewarm_task is not None


@pytest.mark.asyncio
async def test_generate_transport_drop_triggers_reconnect_on_next_turn(
    provider: Any,
) -> None:
    dead_session = AsyncMock()
    provider._session = dead_session
    provider._session_manager = AsyncMock()
    provider._session_handle = "still-valid-handle"

    async def mock_receive_crash() -> AsyncGenerator[Any, Any]:
        raise ConnectionError("connection lost")
        yield

    dead_session.receive = mock_receive_crash

    new_session = AsyncMock()

    async def mock_receive_ok() -> AsyncGenerator[Any, Any]:
        r = MagicMock()
        r.tool_call = None
        r.session_resumption_update = None
        r.go_away = None
        r.server_content.model_turn = None
        r.server_content.output_transcription = None
        r.server_content.grounding_metadata = None
        r.server_content.turn_complete = True
        yield r

    new_session.receive = mock_receive_ok

    mgr = _mgr_yielding(new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        await provider.generate("before crash")
        assert provider._prewarm_task is not None
        await provider._consume_prewarm()

        await provider.generate("after crash")

    dead_session.send_client_content.assert_not_called()
    assert provider._session is new_session
    assert provider._reconnect_pending is False
    assert provider._session_handle == "still-valid-handle"


def test_replay_turns_includes_tool_call_summary(provider: Any) -> None:
    from google.genai.types import Content, FunctionCall, Part

    fc_content = Content(
        role="model",
        parts=[
            Part(function_call=FunctionCall(name="search_web", args={"q": "hello"}))
        ],
    )
    provider._history = [_turn("user", "hi"), fc_content, _turn("model", "found it")]
    turns = provider._replay_turns()
    texts = _texts(turns)
    assert any("[called search_web" in t for t in texts)
    assert texts[2] == "(Assistant earlier said: found it)"


def test_replay_turns_includes_function_response_summary(provider: Any) -> None:
    from google.genai.types import Content, FunctionResponse, Part

    fr_content = Content(
        role="user",
        parts=[
            Part(
                function_response=FunctionResponse(
                    name="search_web",
                    response={"count": 3},
                )
            )
        ],
    )
    provider._history = [fr_content]
    turns = provider._replay_turns()
    texts = _texts(turns)
    assert len(texts) == 1
    assert "[tool search_web -> " in texts[0]


def test_replay_turns_truncates_long_tool_summary(provider: Any) -> None:
    from google.genai.types import Content, FunctionCall, Part

    big_args = {"data": "x" * 500}
    fc_content = Content(
        role="model",
        parts=[Part(function_call=FunctionCall(name="big_tool", args=big_args))],
    )
    provider._history = [fc_content]
    turns = provider._replay_turns()
    texts = _texts(turns)
    assert len(texts[0]) < 400


@pytest.mark.asyncio
async def test_inject_image_stores_to_recent_images(provider: Any) -> None:
    provider._session = None
    await provider._inject_image(b"png_data", "image/png")
    assert len(provider._recent_images) == 1
    assert provider._recent_images[0] == (b"png_data", "image/png")


@pytest.mark.asyncio
async def test_send_user_turn_reinjects_images_after_replay(
    provider: Any,
) -> None:
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    provider._recent_images.append((b"img1", "image/png"))
    provider._recent_images.append((b"img2", "image/jpeg"))
    session = AsyncMock()
    await provider._send_user_turn(session, "go")
    session.send_client_content.assert_called_once()
    assert session.send_realtime_input.call_count == 2
    media_calls = [
        c for c in session.send_realtime_input.call_args_list if c.kwargs.get("media")
    ]
    assert len(media_calls) == 2


@pytest.mark.asyncio
async def test_reinject_image_failure_does_not_abort_turn(provider: Any) -> None:
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    provider._recent_images.append((b"bad", "image/png"))
    session = AsyncMock()
    session.send_realtime_input = AsyncMock(side_effect=RuntimeError("send failed"))
    await provider._send_user_turn(session, "go")
    session.send_client_content.assert_called_once()


@pytest.mark.asyncio
async def test_replay_image_count_zero_skips_reinjection(
    provider: Any, monkeypatch: Any
) -> None:
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    provider._recent_images.append((b"img", "image/png"))
    session = AsyncMock()

    from supporter.config import config as live_cfg

    monkeypatch.setattr(live_cfg, "replay_image_count", 0)

    await provider._send_user_turn(session, "go")
    session.send_client_content.assert_called_once()
    session.send_realtime_input.assert_not_called()


@pytest.mark.asyncio
async def test_recovery_observer_fires_reconnecting_on_prewarm(provider: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    provider._reconnect_pending = True
    provider._schedule_prewarm()
    assert any(e[0] == "reconnecting" for e in events)


@pytest.mark.asyncio
async def test_recovery_observer_fires_context_partial_on_replay_fallback(
    provider: Any,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    session = AsyncMock()
    session.send_client_content = AsyncMock(side_effect=RuntimeError("send died"))
    await provider._send_user_turn(session, "go")
    assert any(e[0] == "context_partial" for e in events)


@pytest.mark.asyncio
async def test_recovery_observer_fires_replaying(provider: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    session = AsyncMock()
    await provider._send_user_turn(session, "go")
    assert any(e[0] == "replaying" for e in events)


@pytest.mark.asyncio
async def test_observer_exception_does_not_break_turn(provider: Any) -> None:
    def bad_observer(event: str, data: dict[str, Any]) -> None:
        raise RuntimeError("observer exploded")

    provider.recovery_observer = bad_observer
    provider._needs_replay = True
    provider._history = [_turn("user", "ctx")]
    session = AsyncMock()
    await provider._send_user_turn(session, "go")
    session.send_client_content.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_backoff_caps_attempts(provider: Any) -> None:
    provider._reconnect_attempts = 100
    provider._reconnect_attempts_max = 2
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    await provider._reconnect()
    assert any(e[0] == "reconnect_giving_up" for e in events)
    assert provider._prewarm_task is None


@pytest.mark.asyncio
async def test_recovery_observer_fires_handle_dropped(provider: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    provider._session_handle = "stale"
    provider._history = [MagicMock()]
    new_session = AsyncMock()
    mgr = _mgr_yielding(STALE_HANDLE_ERROR, new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mgr):
        await provider._ensure_session()
    assert any(e[0] == "handle_dropped" for e in events)


def test_to_seconds_handles_none() -> None:
    from supporter.providers.gemini_live_provider import _to_seconds

    assert _to_seconds(None) is None


def test_to_seconds_handles_float() -> None:
    from supporter.providers.gemini_live_provider import _to_seconds

    assert _to_seconds(5.0) == 5.0


def test_to_seconds_handles_timedelta() -> None:
    from datetime import timedelta

    from supporter.providers.gemini_live_provider import _to_seconds

    assert _to_seconds(timedelta(seconds=10)) == 10.0


def test_to_seconds_handles_int() -> None:
    from supporter.providers.gemini_live_provider import _to_seconds

    assert _to_seconds(7) == 7.0


@pytest.mark.asyncio
async def test_go_away_sets_prewarm_deadline_with_margin(provider: Any) -> None:
    mock_session = AsyncMock()
    provider._session = mock_session

    r_go = MagicMock()
    r_go.tool_call = None
    r_go.session_resumption_update = None
    r_go.go_away = MagicMock()
    r_go.go_away.time_left = 10.0
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
    provider._session_manager = MagicMock()
    provider._session_manager.__aexit__ = AsyncMock(return_value=None)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider.generate("hi")
        assert provider._go_away_deadline is not None
        assert provider._go_away_deadline > 0
        await provider._consume_prewarm()

    assert provider._go_away_deadline is None


@pytest.mark.asyncio
async def test_keepalive_cancelled_on_close(provider: Any) -> None:
    async def never_finish() -> None:
        await asyncio.sleep(3600)

    provider._keepalive_task = asyncio.create_task(never_finish())
    await provider.close()
    assert provider._keepalive_task is None


@pytest.mark.asyncio
async def test_keepalive_skips_probe_during_active_turn(provider: Any) -> None:
    provider._session = AsyncMock()
    provider._turn_lock = asyncio.Lock()
    provider._keepalive_task = None
    provider._go_away_deadline = None
    await provider._turn_lock.acquire()
    try:
        provider._recent_images.clear()
        task = asyncio.create_task(provider._keepalive_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        import contextlib

        with contextlib.suppress(asyncio.CancelledError):
            await task
    finally:
        provider._turn_lock.release()
    provider._session.send_realtime_input.assert_not_called()


@pytest.mark.asyncio
async def test_empty_resume_policy_trust_does_not_force_replay(
    provider: Any, monkeypatch: Any
) -> None:
    provider._session_handle = "valid-handle"
    provider._history = [MagicMock()]
    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)

    from supporter.config import config as live_cfg

    monkeypatch.setattr(live_cfg, "empty_resume_policy", "trust")
    monkeypatch.setattr(live_cfg, "keepalive_enabled", False)

    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider._ensure_session()
    assert provider._needs_replay is False
    assert provider._handle_resumed_pending is True


@pytest.mark.asyncio
async def test_empty_resume_policy_replay_forces_replay_on_reconnect(
    provider: Any, monkeypatch: Any
) -> None:
    provider._session_handle = "valid-handle"
    provider._history = [MagicMock()]
    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)

    from supporter.config import config as live_cfg

    monkeypatch.setattr(live_cfg, "empty_resume_policy", "replay")
    monkeypatch.setattr(live_cfg, "keepalive_enabled", False)

    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider._ensure_session()
    assert provider._needs_replay is True
    assert provider._handle_resumed_pending is False


@pytest.mark.asyncio
async def test_empty_resume_suspected_emits_observer_event(provider: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    provider.recovery_observer = lambda e, d: events.append((e, d))
    provider._handle_resumed_pending = True
    mock_session = AsyncMock()
    provider._session = mock_session

    r_final = MagicMock()
    r_final.tool_call = None
    r_final.session_resumption_update = None
    r_final.go_away = None
    r_final.server_content.model_turn = None
    r_final.server_content.output_transcription = None
    r_final.server_content.grounding_metadata = None
    r_final.server_content.turn_complete = True

    async def mock_receive() -> AsyncGenerator[Any, Any]:
        yield r_final

    mock_session.receive = mock_receive
    provider._handle_resumed_pending = True
    await provider.generate("hi")
    assert any(e[0] == "empty_resume_suspected" for e in events)
    assert provider._handle_resumed_pending is False


@pytest.mark.asyncio
async def test_ensure_session_clears_go_away_deadline_on_success(
    provider: Any,
) -> None:
    provider._go_away_deadline = time.monotonic() - 5
    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        await provider._ensure_session()
    assert provider._go_away_deadline is None


@pytest.mark.asyncio
async def test_keepalive_no_storm_after_reconnect(provider: Any) -> None:
    provider._session = AsyncMock()
    provider._session_manager = AsyncMock()
    provider._go_away_deadline = time.monotonic() - 10

    prewarm_call_count = 0
    original_schedule = provider._schedule_prewarm

    def counting_schedule_prewarm() -> None:
        nonlocal prewarm_call_count
        prewarm_call_count += 1
        original_schedule()

    provider._schedule_prewarm = counting_schedule_prewarm

    new_session = AsyncMock()
    mock_mgr = MagicMock()
    mock_mgr.__aenter__ = AsyncMock(return_value=new_session)
    with patch.object(provider.client.aio.live, "connect", return_value=mock_mgr):
        provider._reconnect_pending = True
        await provider._prepare_turn("hi")

    assert provider._go_away_deadline is None
    assert provider._reconnect_pending is False

    provider._go_away_deadline = None
    prewarm_call_count = 0

    provider._session = new_session
    task = asyncio.create_task(provider._keepalive_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert prewarm_call_count == 0
