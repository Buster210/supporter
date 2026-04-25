from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.providers.gemini_live_provider import GeminiLiveProvider


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


@pytest.mark.asyncio
async def test_auto_register_search() -> None:
    p = GeminiLiveProvider(api_keys=["key"], model_name="gemini-3.1-flash")
    assert "google_search" in p.registry


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
    r1.server_content.model_turn.parts = [MagicMock(text="Thinking", thought=True)]
    r1.server_content.output_transcription = None
    r1.server_content.grounding_metadata = None
    r1.server_content.turn_complete = False
    r2 = MagicMock()
    r2.tool_call = None
    r2.server_content.model_turn = None
    r2.server_content.grounding_metadata = MagicMock()
    r2.server_content.output_transcription.text = "Done"
    r2.server_content.turn_complete = True

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
    fc = MagicMock()
    fc.name = "my_tool"
    fc.args = {"a": 1}
    fc.id = "call_1"
    r1.tool_call.function_calls = [fc]
    r1.server_content = None
    r2 = MagicMock()
    r2.tool_call = None
    r2.server_content.model_turn.parts = [MagicMock(text="T", thought=True)]
    r2.server_content.output_transcription.text = "Stream"
    r2.server_content.turn_complete = True

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
