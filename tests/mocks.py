from collections import deque
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def mock_client() -> MagicMock:
    return create_mock_genai_client()


def get_mock_gemini_response(text: str = "Mocked Response") -> MagicMock:
    usage_metadata = MagicMock()
    usage_metadata.prompt_token_count = 10
    usage_metadata.candidates_token_count = 20
    usage_metadata.total_token_count = 30
    response = MagicMock()
    response.text = text
    response.usage_metadata = usage_metadata
    response.id = "mock-interaction-id"
    candidate = MagicMock()
    candidate.content.parts = [MagicMock(text=text)]
    response.candidates = [candidate]
    return response


def build_mock_stream_part(
    text: str | None = None, *, is_thought: bool = False, function_call: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(text=text, thought=is_thought, function_call=function_call)


def build_mock_stream_chunk(
    *,
    parts: list[Any] | None = None,
    include_candidate: bool = True,
    include_content: bool = True,
) -> SimpleNamespace:
    if not include_candidate:
        return SimpleNamespace(candidates=[])
    content = SimpleNamespace(parts=parts) if include_content else None
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class MockLiveSession:
    def __init__(self) -> None:
        self.sent_messages: list[Any] = []
        self.received_messages: deque[Any] = deque()
        self.receive = MagicMock()
        self.send_realtime_input = MagicMock(side_effect=lambda *args, **kwargs: None)
        self.send_tool_response = MagicMock(side_effect=lambda *args, **kwargs: None)

    async def __aenter__(self) -> MockLiveSession:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def create_mock_genai_client(**kwargs: Any) -> MagicMock:
    client = MagicMock()
    client.models = MagicMock()
    client.interactions = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.interactions = MagicMock()
    client.aio.live = MagicMock()
    res = get_mock_gemini_response("Mocked Response")
    client._stream_chunks = [
        build_mock_stream_chunk(
            parts=[build_mock_stream_part(text="Chunk 1", is_thought=False)]
        ),
        build_mock_stream_chunk(
            parts=[build_mock_stream_part(text="Chunk 2", is_thought=False)]
        ),
    ]

    async def mock_generate(**kwargs: Any) -> MagicMock:
        return res

    async def mock_interaction(**kwargs: Any) -> Any:
        return None

    async def mock_stream(**kwargs: Any) -> AsyncIterator[Any]:

        async def internal_gen() -> AsyncIterator[Any]:
            for stream_chunk in client._stream_chunks:
                yield stream_chunk

        return internal_gen()

    def mock_stream_sync(**kwargs: Any) -> Iterator[Any]:
        yield from client._stream_chunks

    client.models.generate_content = MagicMock(side_effect=mock_generate)
    client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)
    client.interactions.create = MagicMock(side_effect=mock_interaction)
    client.aio.interactions.create = AsyncMock(side_effect=mock_interaction)
    client.models.generate_content_stream = MagicMock(side_effect=mock_stream_sync)
    client.aio.models.generate_content_stream = AsyncMock(side_effect=mock_stream)
    mock_session = MockLiveSession()
    client.aio.live.connect = MagicMock(return_value=mock_session)
    return client
