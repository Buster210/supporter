from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from google.genai.types import Content, GenerateContentConfig, Tool

from .config import (
    HTTP_INTERNAL_ERROR,
    HTTP_NOT_FOUND,
    HTTP_RATE_LIMIT,
    HTTP_SERVICE_UNAVAILABLE,
    config,
)
from .gemini_provider import GeminiProvider
from .logger import logger


class LLMOptions(TypedDict, total=False):
    history: list[Content]
    model: str
    tools: list[Tool]
    registry: dict[str, Callable]
    interaction_id: str
    use_search: bool
    use_code_execution: bool
    system_instruction: str
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    config: GenerateContentConfig


@dataclass
class LLMResult:
    text: str
    model: str | None = None
    duration: float | None = None
    interaction_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    automatic_function_calling_history: list[Content] | None = None
    candidates: list[Any] = field(default_factory=list)


@dataclass
class LLMChunk:
    text: str
    is_last: bool
    model: str | None = None
    raw: Any = None


class LLMProvider(Protocol):
    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult: ...

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]: ...

    def get_name(self) -> str: ...


def is_rate_limit(error: Any) -> bool:
    status = getattr(error, "status", None)
    if status == HTTP_RATE_LIMIT:
        return True
    message = str(error).lower()
    return any(sig in message for sig in ["quota", "too many requests", "429"])


def is_model_error(error: Any) -> bool:
    status = getattr(error, "status", None)
    if status in [HTTP_NOT_FOUND, HTTP_SERVICE_UNAVAILABLE, HTTP_INTERNAL_ERROR]:
        return True
    message = str(error).lower()
    transient_signals = ["unavailable", "overloaded", "internal error", "service level"]
    if any(sig in message for sig in transient_signals):
        return True
    return any(code in message for code in ["404", "503", "500"])


def should_trigger_fallback(error: Any) -> bool:
    return is_rate_limit(error) or is_model_error(error)


ProviderType = str


class RoundRobinPool(LLMProvider):
    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("RoundRobinPool requires at least one provider instance.")
        self.providers = providers
        self.current_index = 0

    def _get_next(self) -> tuple[LLMProvider, int]:
        provider = self.providers[self.current_index]
        key_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.providers)
        return provider, key_index

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        last_error = None
        for attempt in range(len(self.providers)):
            provider, key_index = self._get_next()
            logger.debug(f"  ├─ Key {key_index} [attempt {attempt + 1}/{len(self.providers)}]")
            try:
                result = await provider.generate(prompt, options)
                if not result.model:
                    result.model = provider.get_name()
                logger.debug(f"  ├─ Key {key_index} ✓ SUCCESS")
                return result
            except Exception as e:
                last_error = e
                if not is_rate_limit(e):
                    logger.error(f"  ├─ Key {key_index} ✗ FATAL ERROR: {type(e).__name__}: {str(e)[:60]}")
                    raise e
                logger.warning(
                    f"  ├─ Key {key_index} ✗ RATE LIMIT (429) - retrying next key..."
                )
        logger.error(f"  └─ All {len(self.providers)} keys exhausted!")
        raise last_error

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        provider, key_index = self._get_next()
        logger.debug(f"Streaming with key {key_index}")
        async for chunk in provider.generate_stream(prompt, options):
            yield chunk

    def get_name(self) -> str:
        base_name = self.providers[0].get_name()
        if len(self.providers) > 1:
            return f"{base_name} (Pool x{len(self.providers)})"
        return base_name


class FallbackProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        logger.info(f"[PRIMARY] Attempting with model: {self.primary.get_name()}")
        try:
            result = await self.primary.generate(prompt, options)
            if not result.model:
                result.model = self.primary.get_name()
            logger.info(f"[PRIMARY] Success! Response from {self.primary.get_name()}")
            return result
        except Exception as e:
            if not should_trigger_fallback(e):
                logger.error(f"[PRIMARY] Fatal error (not triggering fallback): {e}")
                raise e
            logger.warning(f"[PRIMARY] All attempts exhausted. Switching to fallback...")
            logger.info(f"[FALLBACK] Attempting with model: {self.fallback.get_name()}")
            try:
                result = await self.fallback.generate(prompt, options)
                if not result.model:
                    result.model = self.fallback.get_name()
                logger.info(f"[FALLBACK] Success! Response from {self.fallback.get_name()}")
                return result
            except Exception as fallback_error:
                logger.error(f"[FALLBACK] Failed: {fallback_error}")
                raise fallback_error

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        logger.info(f"[PRIMARY-STREAM] Attempting with model: {self.primary.get_name()}")
        try:
            async for chunk in self.primary.generate_stream(prompt, options):
                yield chunk
            logger.info(f"[PRIMARY-STREAM] Success!")
        except Exception as e:
            if not should_trigger_fallback(e):
                logger.error(f"[PRIMARY-STREAM] Fatal error (not triggering fallback): {e}")
                raise e
            logger.warning(f"[PRIMARY-STREAM] Failed. Switching to fallback...")
            logger.info(f"[FALLBACK-STREAM] Attempting with model: {self.fallback.get_name()}")
            try:
                async for chunk in self.fallback.generate_stream(prompt, options):
                    yield chunk
                logger.info(f"[FALLBACK-STREAM] Success!")
            except Exception as fallback_error:
                logger.error(f"[FALLBACK-STREAM] Failed: {fallback_error}")
                raise fallback_error

    def get_name(self) -> str:
        return f"{self.primary.get_name()} -> {self.fallback.get_name()}"


def get_provider(provider_type: ProviderType | None = None) -> LLMProvider:
    target_type = provider_type or config.provider
    if target_type != "gemini":
        raise ValueError(f"Unsupported provider type: {target_type}")
    keys = config.gemini_api_keys
    if not keys:
        raise ValueError("GEMINI_API_KEYS is missing/empty in environment")

    def _build_chain(model_name: str) -> LLMProvider:
        pool = []
        for key in keys:
            p = GeminiProvider(key)
            p.model_name = model_name
            pool.append(p)
        return RoundRobinPool(pool) if len(pool) > 1 else pool[0]

    primary = _build_chain(config.gemini_model)
    if config.gemini_fallback_model:
        logger.debug(
            f"Configuring fallback: {config.gemini_model} -> {config.gemini_fallback_model}"
        )
        return FallbackProvider(primary, _build_chain(config.gemini_fallback_model))
    return primary
