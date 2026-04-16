from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock


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


def create_mock_genai_client(**kwargs: Any) -> MagicMock:

    client = MagicMock()
    client.models = MagicMock()
    client.interactions = MagicMock()

    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.interactions = MagicMock()

    res = get_mock_gemini_response("Mocked Response")

    async def mock_generate(**kwargs: Any) -> MagicMock:
        return res

    async def mock_interaction(**kwargs: Any) -> None:
        return None

    async def mock_stream(**kwargs: Any) -> AsyncIterator[Any]:
        async def internal_gen() -> AsyncIterator[Any]:
            # Mimic the SDK chunk structure: chunk.candidates[0].content.parts[0].text
            content_chunk_1 = MagicMock()
            content_chunk_1.candidates = [MagicMock()]
            content_chunk_1.candidates[0].content.parts = [
                MagicMock(text="Chunk 1", thought=False)
            ]
            yield content_chunk_1

            content_chunk_2 = MagicMock()
            content_chunk_2.candidates = [MagicMock()]
            content_chunk_2.candidates[0].content.parts = [
                MagicMock(text="Chunk 2", thought=False)
            ]
            yield content_chunk_2

        return internal_gen()

    def mock_stream_sync(**kwargs: Any) -> Iterator[Any]:
        content_chunk_1 = MagicMock()
        content_chunk_1.candidates = [MagicMock()]
        content_chunk_1.candidates[0].content.parts = [
            MagicMock(text="Chunk 1", thought=False)
        ]
        yield content_chunk_1

        content_chunk_2 = MagicMock()
        content_chunk_2.candidates = [MagicMock()]
        content_chunk_2.candidates[0].content.parts = [
            MagicMock(text="Chunk 2", thought=False)
        ]
        yield content_chunk_2

    client.models.generate_content = AsyncMock(side_effect=mock_generate)
    client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)

    client.interactions.create = AsyncMock(side_effect=mock_interaction)
    client.aio.interactions.create = AsyncMock(side_effect=mock_interaction)

    client.models.generate_content_stream = MagicMock(side_effect=mock_stream_sync)
    client.aio.models.generate_content_stream = AsyncMock(side_effect=mock_stream)

    return client
