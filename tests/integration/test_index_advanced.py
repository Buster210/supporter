from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import supporter.index
from supporter.index import (
    DynamicPool,
    LazyFallbackProvider,
    LLMChunk,
    _is_model_in_cooldown,
    _mark_model_cooldown,
    clear_providers,
    get_provider,
    is_model_error,
)


@pytest.fixture(autouse=True)
async def pool_cleanup() -> Any:
    yield
    await DynamicPool.shutdown_all()
    clear_providers()


@pytest.mark.asyncio
async def test_is_model_error_google_classes() -> None:

    class InternalServerError(Exception):
        pass

    assert is_model_error(InternalServerError("test")) is True


@pytest.mark.asyncio
async def test_dynamic_pool_replace_missing_provider() -> None:
    lb = DynamicPool(["key1"], model_name="test")
    p_external = MagicMock()
    lb._replace_instance(p_external)
    assert len(lb.active_slots) == 0


@pytest.mark.asyncio
async def test_dynamic_pool_cooldown_generate() -> None:
    lb = DynamicPool(["key1"], model_name="cooldown-model")
    _mark_model_cooldown("cooldown-model", minutes=1)
    with pytest.raises(RuntimeError, match="in cooldown"):
        await lb.generate("test")


@pytest.mark.asyncio
async def test_dynamic_pool_non_fallbackable_error() -> None:
    lb = DynamicPool(["key1"], model_name="test")
    p1 = MagicMock()
    p1.generate = AsyncMock(side_effect=ValueError("Specific error"))
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(ValueError, match="Specific error"),
    ):
        await lb.generate("test")


@pytest.mark.asyncio
async def test_dynamic_pool_exhausted_retries() -> None:
    lb = DynamicPool(["key1", "key2"], model_name="test")
    p1 = MagicMock()
    p1.generate = AsyncMock(side_effect=Exception("quota"))
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(Exception, match="quota"),
    ):
        await lb.generate("test")


@pytest.mark.asyncio
async def test_dynamic_pool_streaming_error_handling() -> None:
    lb = DynamicPool(["key1"], model_name="stream-cooldown")
    _mark_model_cooldown("stream-cooldown", minutes=1)
    with pytest.raises(RuntimeError, match="in cooldown"):
        async for _ in lb.generate_stream("test"):
            pass
    lb2 = DynamicPool(["key1", "key2"], pool_size=1, model_name="test-stream")
    p1 = MagicMock()

    async def error_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        if False:
            yield
        raise Exception("quota")

    p1.generate_stream = MagicMock(side_effect=error_gen)
    p2 = MagicMock()

    async def success_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        yield LLMChunk(text="success", is_last=True)

    p2.generate_stream = MagicMock(side_effect=success_gen)
    with patch("supporter.index.GeminiProvider", side_effect=[p1, p2, p2, p2]):
        chunks = []
        async for chunk in lb2.generate_stream("test"):
            chunks.append(chunk)
        assert chunks[0].text == "success"


@pytest.mark.asyncio
async def test_lazy_fallback_streaming() -> None:
    primary = MagicMock()

    async def error_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        if False:
            yield
        raise Exception("quota")

    primary.generate_stream = MagicMock(side_effect=error_gen)
    primary.get_name.return_value = "primary"
    fallback = MagicMock()

    async def success_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        yield LLMChunk(text="fallback", is_last=True)

    fallback.generate_stream = MagicMock(side_effect=success_gen)
    fallback.get_name.return_value = "fallback"
    lfp = LazyFallbackProvider(lambda: primary, lambda: fallback)
    chunks = []
    async for chunk in lfp.generate_stream("test"):
        chunks.append(chunk)
    assert chunks[0].text == "fallback"
    assert lfp.get_name() == "primary -> [Fallback]"


@pytest.mark.asyncio
async def test_get_provider_live_and_registry() -> None:
    with patch.object(supporter.index.config, "gemini_api_keys", ["key1"]):
        clear_providers()
        p1 = get_provider(live=True, registry={"tool": lambda: "test"})
        new_registry = {"other": lambda: "val"}
        p2 = get_provider(live=True, registry=new_registry)
        assert p1 is p2
    with pytest.raises(ValueError, match="Unsupported"):
        get_provider(provider_type="unknown")


@pytest.mark.asyncio
async def test_get_provider_missing_keys() -> None:
    with patch.object(supporter.index.config, "gemini_api_keys", []):
        clear_providers()
        with pytest.raises(ValueError, match="GEMINI_API_KEYS is missing"):
            get_provider()


@pytest.mark.asyncio
async def test_dynamic_pool_5xx_error_generate() -> None:
    class StatusError(Exception):
        status: int

    lb = DynamicPool(["key1"], model_name="test-5xx")
    p1 = MagicMock()
    error = StatusError("Internal Server Error")
    error.status = 500
    p1.generate = AsyncMock(side_effect=error)
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(Exception, match="Internal Server Error"),
    ):
        await lb.generate("test")
    assert _is_model_in_cooldown("test-5xx") is True


@pytest.mark.asyncio
async def test_dynamic_pool_5xx_error_stream() -> None:
    class StatusError(Exception):
        status: int

    lb = DynamicPool(["key1"], model_name="test-5xx-stream")
    p1 = MagicMock()

    async def error_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        error = StatusError("Service Unavailable")
        error.status = 503
        if False:
            yield
        raise error

    p1.generate_stream = MagicMock(side_effect=error_gen)
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(Exception, match="Service Unavailable"),
    ):
        async for _ in lb.generate_stream("test"):
            pass
    assert _is_model_in_cooldown("test-5xx-stream") is True


@pytest.mark.asyncio
async def test_get_provider_factories() -> None:
    with (
        patch.object(supporter.index.config, "gemini_api_keys", ["key1"]),
        patch.object(
            supporter.index.config, "gemini_live_fallback_model", "fallback-live"
        ),
    ):
        clear_providers()
        p = get_provider(live=True)
        name = p.get_name()
        assert "Live" in name
        assert "Fallback" in name
        clear_providers()
        with patch.object(
            supporter.index.config, "gemini_fallback_model", "fallback-model"
        ):
            p_non_live = get_provider(live=False)
            name_non_live = p_non_live.get_name()
            assert "Dynamic Pool" in name_non_live
            assert "Fallback" in name_non_live


@pytest.mark.asyncio
async def test_dynamic_pool_bg_task_failure() -> None:
    lb = DynamicPool(["key1", "key2"], model_name="test-bg")
    p1 = MagicMock()
    lb.active_slots.append(p1)
    with patch("asyncio.create_task", side_effect=RuntimeError("no loop")):
        lb._replace_instance(p1)
        assert len(lb.active_slots) == 1
    for task in lb.background_tasks:
        task.cancel()


@pytest.mark.asyncio
async def test_dynamic_pool_all_keys_fail_non_fallback() -> None:
    lb = DynamicPool(["key1", "key2"], model_name="test-all-fail")
    p1 = MagicMock()
    p1.generate = AsyncMock(side_effect=ValueError("Invalid prompt"))
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(ValueError, match="Invalid prompt"),
    ):
        await lb.generate("test")


@pytest.mark.asyncio
async def test_lazy_fallback_stream_no_fallback() -> None:
    primary = MagicMock()

    async def error_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        if False:
            yield
        raise Exception("quota")

    primary.generate_stream = MagicMock(side_effect=error_gen)
    primary.get_name.return_value = "primary"
    lfp = LazyFallbackProvider(lambda: primary, fallback_factory=None)
    with pytest.raises(Exception, match="quota"):
        async for _ in lfp.generate_stream("test"):
            pass


@pytest.mark.asyncio
async def test_dynamic_pool_stream_fail_no_chunks_no_fallback() -> None:
    lb = DynamicPool(["key1"], model_name="test-stream-fail")
    p1 = MagicMock()

    async def error_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        if False:
            yield
        raise ValueError("Invalid input")

    p1.generate_stream = MagicMock(side_effect=error_gen)
    with (
        patch("supporter.index.GeminiProvider", return_value=p1),
        pytest.raises(ValueError, match="Invalid input"),
    ):
        async for _ in lb.generate_stream("test"):
            pass


@pytest.mark.asyncio
async def test_get_provider_live_without_fallback_model() -> None:
    with (
        patch.object(supporter.index.config, "gemini_api_keys", ["key1"]),
        patch.object(supporter.index.config, "gemini_live_fallback_model", None),
    ):
        clear_providers()
        p = get_provider(live=True)
        assert "Fallback" not in p.get_name()
