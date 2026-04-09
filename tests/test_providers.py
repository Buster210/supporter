import pytest

from supporter.index import get_provider


@pytest.mark.asyncio
async def test_gemini_provider_generate(mock_genai_client):
    provider = get_provider("gemini")
    assert provider.get_name() == "gemini-3.1-flash-lite-preview"

    result = await provider.generate("test")
    assert result.text == "Mocked Response"
    assert result.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_provider_options_propagation(mock_genai_client):
    provider = get_provider("gemini")

    await provider.generate(
        "test", {"temperature": 0.1, "top_p": 0.9, "max_output_tokens": 100}
    )

    # Verify the last call
    client_instance = mock_genai_client.return_value

    # Check which one was called
    if client_instance.aio.interactions.create.called:
        call_args = client_instance.aio.interactions.create.call_args
    else:
        # Check both the sync and async mocks as a fallback
        call_args = (
            client_instance.aio.models.generate_content.call_args
            or client_instance.models.generate_content.call_args
        )

    config = call_args[1]["config"]
    assert config.temperature == 0.1
    assert config.top_p == 0.9
    assert config.max_output_tokens == 100
