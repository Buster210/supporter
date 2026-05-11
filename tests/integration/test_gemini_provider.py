import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import Content, Part

import supporter.index as index
from supporter.config import load_config
from supporter.index import get_provider
from supporter.providers.gemini_provider import GeminiProvider
from tests.mocks import build_mock_stream_chunk, build_mock_stream_part


@pytest.mark.asyncio
async def test_gemini_provider_generate(mock_genai_client: Any) -> None:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",  # pragma: allowlist secret
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "GEMINI_FALLBACK_MODEL": "",
            "GOOGLE_API_KEY": "",
        },
        clear=True,
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider("gemini")
        assert "gemini" in provider.get_name().lower()
        result = await provider.generate("test")
        assert result.text == "Mocked Response"
        assert result.usage["total_tokens"] == 30


@pytest.mark.asyncio
async def test_provider_streaming(mock_genai_client: Any) -> None:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",  # pragma: allowlist secret
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "GEMINI_FALLBACK_MODEL": "",
            "GOOGLE_API_KEY": "",
        },
        clear=True,
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider("gemini")
        chunks = []
        async for chunk in provider.generate_stream("Say 'Test Success'"):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].text == "Chunk 1"
        assert chunks[1].text == "Chunk 2"
        assert chunks[1].is_last is False
        assert "".join(c.text for c in chunks) == "Chunk 1Chunk 2"


@pytest.mark.asyncio
async def test_provider_streaming_thought_chunks(mock_genai_client: Any) -> None:
    provider = GeminiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model_name="gemini-test-model",
    )
    cast(Any, provider).client._stream_chunks = [
        build_mock_stream_chunk(
            parts=[build_mock_stream_part(text="Internal reasoning", is_thought=True)]
        )
    ]
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].text == "Internal reasoning"
    assert chunks[0].is_thought is True
    assert chunks[0].is_tool_call is False


@pytest.mark.asyncio
async def test_provider_streaming_function_call_chunks(mock_genai_client: Any) -> None:
    provider = GeminiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model_name="gemini-test-model",
    )
    cast(Any, provider).client._stream_chunks = [
        build_mock_stream_chunk(
            parts=[
                build_mock_stream_part(
                    function_call=SimpleNamespace(
                        name="lookup_weather", args={"city": "Delhi"}
                    )
                )
            ]
        )
    ]
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].is_tool_call is True
    assert chunks[0].tool_name == "lookup_weather"
    assert chunks[0].tool_args == {"city": "Delhi"}


@pytest.mark.asyncio
async def test_provider_streaming_handles_empty_candidates(
    mock_genai_client: Any,
) -> None:
    provider = GeminiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model_name="gemini-test-model",
    )
    cast(Any, provider).client._stream_chunks = [
        build_mock_stream_chunk(include_candidate=False),
        build_mock_stream_chunk(
            parts=[build_mock_stream_part(text="Chunk after skip")]
        ),
    ]
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].text == "Chunk after skip"


@pytest.mark.asyncio
async def test_provider_streaming_handles_missing_parts(mock_genai_client: Any) -> None:
    provider = GeminiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model_name="gemini-test-model",
    )
    cast(Any, provider).client._stream_chunks = [
        build_mock_stream_chunk(include_content=False),
        build_mock_stream_chunk(parts=[]),
        build_mock_stream_chunk(parts=[build_mock_stream_part(text="Recovered chunk")]),
    ]
    chunks = []
    async for chunk in provider.generate_stream("test"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].text == "Recovered chunk"


@pytest.mark.asyncio
async def test_provider_options_propagation(mock_genai_client: Any) -> None:
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "test-key",  # pragma: allowlist secret
            "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            "GEMINI_FALLBACK_MODEL": "",
            "GOOGLE_API_KEY": "",
        },
        clear=True,
    ):
        index.clear_providers()
        index.config = load_config()
        provider = get_provider("gemini")
        await provider.generate(
            "test", {"temperature": 0.1, "top_p": 0.9, "max_output_tokens": 100}
        )
        client_instance = mock_genai_client.return_value
        if client_instance.aio.interactions.create.called:
            call_args = client_instance.aio.interactions.create.call_args
        else:
            call_args = (
                client_instance.aio.models.generate_content.call_args
                or client_instance.models.generate_content.call_args
            )
        if call_args:
            config = call_args[1]["config"]
            assert config.temperature == 0.1
            assert config.max_output_tokens == 100


@pytest.mark.asyncio
async def test_provider_generate_interaction_resume_falls_back_to_standard_generation(
    mock_genai_client: Any,
) -> None:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    client_instance = mock_genai_client.return_value
    client_instance.aio.interactions.create.side_effect = RuntimeError("resume failed")
    result = await provider.generate("test", {"interaction_id": "interaction-123"})
    client_instance.aio.interactions.create.assert_awaited_once()
    assert (
        client_instance.aio.interactions.create.call_args.kwargs[
            "previous_interaction_id"
        ]
        == "interaction-123"
    )
    client_instance.aio.models.generate_content.assert_awaited_once()
    assert result.text == "Mocked Response"


@pytest.mark.asyncio
async def test_provider_generate_uses_top_level_automatic_function_calling_history(
    mock_genai_client: Any,
) -> Any:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    client_instance = mock_genai_client.return_value
    mock_history = [{"role": "model", "parts": ["tool result"]}]
    response = MagicMock()
    response.text = "Mocked Response"
    response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=20, total_token_count=30
    )
    response.id = "mock-interaction-id"
    candidate = MagicMock()
    candidate.content.parts = [MagicMock(text="Mocked Response", thought=False)]
    response.candidates = [candidate]
    response.automatic_function_calling_history = mock_history
    response.response = None
    response.history = None

    async def mock_generate(**kwargs: Any) -> Any:
        return response

    client_instance.aio.models.generate_content.side_effect = mock_generate
    result = await provider.generate("test")
    assert cast(Any, result.automatic_function_calling_history) == mock_history


@pytest.mark.asyncio
async def test_provider_generate_uses_nested_response_history_fallback(
    mock_genai_client: Any,
) -> Any:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    client_instance = mock_genai_client.return_value
    mock_history = [{"role": "model", "parts": ["nested history"]}]
    response = MagicMock()
    response.text = "Mocked Response"
    response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=20, total_token_count=30
    )
    response.id = "mock-interaction-id"
    candidate = MagicMock()
    candidate.content.parts = [MagicMock(text="Mocked Response", thought=False)]
    response.candidates = [candidate]
    response.automatic_function_calling_history = None
    response.response = MagicMock(automatic_function_calling_history=mock_history)
    response.history = None

    async def mock_generate(**kwargs: Any) -> Any:
        return response

    client_instance.aio.models.generate_content.side_effect = mock_generate
    result = await provider.generate("test", {"interaction_id": "interaction-123"})
    assert cast(Any, result.automatic_function_calling_history) == mock_history


@pytest.mark.asyncio
async def test_provider_generate_uses_result_history_as_final_fallback(
    mock_genai_client: Any,
) -> Any:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    client_instance = mock_genai_client.return_value
    mock_history = [{"role": "model", "parts": ["history fallback"]}]
    response = MagicMock()
    response.text = "Mocked Response"
    response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=20, total_token_count=30
    )
    response.id = "mock-interaction-id"
    candidate = MagicMock()
    candidate.content.parts = [MagicMock(text="Mocked Response", thought=False)]
    response.candidates = [candidate]
    response.automatic_function_calling_history = None
    response.response = MagicMock(automatic_function_calling_history=None)
    response.history = mock_history

    async def mock_generate(**kwargs: Any) -> Any:
        return response

    client_instance.aio.models.generate_content.side_effect = mock_generate
    result = await provider.generate("test", {"interaction_id": "interaction-123"})
    assert cast(Any, result.automatic_function_calling_history) == mock_history


@pytest.mark.asyncio
async def test_provider_tool_transformation(mock_genai_client: Any) -> None:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret

    def sync_tool() -> str:
        return "sync"

    async def async_tool() -> str:
        return "async"

    options = {
        "registry": {"sync_tool": sync_tool, "async_tool": async_tool},
        "use_search": True,
        "use_code_execution": True,
    }
    transformed = provider._transform_tools(cast(index.LLMOptions, options))
    assert transformed is not None
    assert len(transformed) >= 4


def test_transform_tools_returns_cached_result_on_cache_hit() -> Any:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret

    def sync_tool() -> str:
        return "sync"

    async def async_tool() -> str:
        return "async"

    options = {
        "registry": {"sync_tool": sync_tool, "async_tool": async_tool},
        "use_search": True,
        "use_code_execution": True,
    }
    first = provider._transform_tools(cast(index.LLMOptions, options))
    second = provider._transform_tools(cast(index.LLMOptions, options))
    assert first is second


def test_extract_declared_tool_names_supports_dict_function_declarations() -> None:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    names = provider._extract_declared_tool_names(
        [{"function_declarations": [{"name": "sync_tool"}, {"name": "async_tool"}]}]
    )
    assert names == {"sync_tool", "async_tool"}


def test_transform_tools_skips_registry_tools_with_predeclared_names() -> Any:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret

    def sync_tool() -> str:
        return "sync"

    async def async_tool() -> str:
        return "async"

    declared_tool = {"function_declarations": [{"name": "sync_tool"}]}
    options = {
        "tools": [declared_tool],
        "registry": {"sync_tool": sync_tool, "async_tool": async_tool},
    }
    transformed = provider._transform_tools(cast(index.LLMOptions, options))
    assert transformed is not None
    assert len(transformed) == 2
    assert transformed[0] is declared_tool
    wrapped_tools = [tool for tool in transformed if callable(tool)]
    assert len(wrapped_tools) == 1
    assert wrapped_tools[0].__name__ == "async_tool"


def test_transform_tools_does_not_duplicate_code_execution_tool() -> None:
    from google.genai import types

    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    code_execution_tool = types.Tool(
        code_execution=types.ToolCodeExecution(),
    )
    transformed = provider._transform_tools(
        cast(
            index.LLMOptions,
            {"tools": [code_execution_tool], "use_code_execution": True},
        )
    )

    assert transformed is not None
    assert (
        sum(
            1
            for tool in transformed
            if getattr(tool, "code_execution", None) is not None
        )
        == 1
    )


GEMINI_3_FLASH = "gemini-3.1-flash"
GEMMA_4_IT = "gemma-4-31b-it"


def test_transform_search_gemini_3() -> None:
    provider = GeminiProvider(api_key="test-key", model_name=GEMINI_3_FLASH)
    transformed = provider._transform_tools(
        cast(index.LLMOptions, {"registry": {}, "use_search": True})
    )
    assert transformed is not None
    assert "google_search" in {t.__name__ for t in transformed if callable(t)}


def test_transform_search_gemma() -> None:
    provider = GeminiProvider(api_key="test-key", model_name=GEMMA_4_IT)
    transformed = provider._transform_tools(
        cast(index.LLMOptions, {"registry": {}, "use_search": True})
    )
    assert transformed is not None
    assert any(hasattr(t, "google_search") for t in transformed)


@pytest.mark.asyncio
async def test_provider_prepare_contents() -> None:
    provider = GeminiProvider(api_key="test-key")  # pragma: allowlist secret
    history = [Content(role="user", parts=[Part(text="hello")])]
    contents = provider._prepare_contents("how are you?", history=history)
    assert len(contents) == 2
    assert contents[0].parts is not None
    assert contents[0].parts[0].text == "hello"
    assert contents[1].parts is not None
    assert contents[1].parts[0].text == "how are you?"


def test_provider_get_name_returns_model_name() -> None:
    provider = GeminiProvider(
        api_key="test-key",  # pragma: allowlist secret
        model_name="gemini-test-model",
    )
    assert provider.get_name() == "gemini-test-model"
