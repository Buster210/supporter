from unittest.mock import AsyncMock, MagicMock

from supporter.index import LLMChunk


def get_mock_gemini_response(text: str = "Mocked Response") -> MagicMock:
    usage_metadata = MagicMock()
    usage_metadata.prompt_token_count = 10
    usage_metadata.candidates_token_count = 20
    usage_metadata.total_token_count = 30

    response = MagicMock()
    response.text = text
    response.usage_metadata = usage_metadata
    response.id = "mock-interaction-id"

    # Candidates
    candidate = MagicMock()
    candidate.content.parts = [MagicMock(text=text)]
    response.candidates = [candidate]

    return response


def create_mock_genai_client(**kwargs):
    # Create standard structure
    client = MagicMock()
    client.models = MagicMock()
    client.interactions = MagicMock()

    # Create aio structure
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.interactions = MagicMock()

    # Mock responses
    res = get_mock_gemini_response("Mocked Response")

    # Mock methods
    async def mock_generate(**kwargs):
        return res

    async def mock_interaction(**kwargs):
        return None  # Force fallback to models.generate_content

    async def mock_stream(**kwargs):
        yield LLMChunk(text="Chunk 1", is_last=False)
        yield LLMChunk(text="Chunk 2", is_last=True)

    client.models.generate_content = AsyncMock(side_effect=mock_generate)
    client.aio.models.generate_content = AsyncMock(side_effect=mock_generate)

    client.interactions.create = AsyncMock(side_effect=mock_interaction)
    client.aio.interactions.create = AsyncMock(side_effect=mock_interaction)

    client.models.generate_content_stream = MagicMock(side_effect=mock_stream)
    client.aio.models.generate_content_stream = MagicMock(side_effect=mock_stream)

    return client
