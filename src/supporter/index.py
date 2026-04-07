from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    Protocol,
    TypedDict,
)

from google.genai.types import Content, GenerateContentConfig, Tool

from .config import config
from .logger import logger
from .providers.gemini_provider import GeminiProvider


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
    message = str(error).lower()
    return (
        status == 429
        or "429" in message
        or "quota" in message
        or "too many requests" in message
    )


def is_model_error(error: Any) -> bool:
    status = getattr(error, "status", None)
    message = str(error).lower()
    return (
        status in [404, 503, 500]
        or "unavailable" in message
        or "overloaded" in message
        or "503" in message
        or "404" in message
        or "500" in message
        or "internal error" in message
        or "service level" in message
    )


def is_fallback_error(error: Any) -> bool:
    return is_rate_limit(error) or is_model_error(error)


ProviderType = str


class RoundRobinKeyProvider(LLMProvider):
    def __init__(self, instances: list[LLMProvider]):
        if not instances:
            raise ValueError(
                "RoundRobinKeyProvider requires at least one provider instance."
            )
        self.instances = instances
        self.current_index = 0

    def _get_next_instance(self) -> LLMProvider:
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return instance

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        last_error = None
        max_retries = len(self.instances)

        for _ in range(max_retries):
            instance = self._get_next_instance()
            try:
                result = await instance.generate(prompt, options)
                if not result.model:
                    result.model = instance.get_name()
                return result
            except Exception as e:
                last_error = e
                if is_rate_limit(e):
                    continue
                raise e
        raise last_error

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        instance = self._get_next_instance()
        async for chunk in instance.generate_stream(prompt, options):
            yield chunk

    def get_name(self) -> str:
        base_name = self.instances[0].get_name()
        if len(self.instances) > 1:
            return f"{base_name} (Round Robin x{len(self.instances)})"
        return base_name


class FallbackProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback

    def _should_fallback(self, error: Any) -> bool:
        return is_fallback_error(error)

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        try:
            result = await self.primary.generate(prompt, options)
            if not result.model:
                result.model = self.primary.get_name()
            return result
        except Exception as e:
            if self._should_fallback(e):
                logger.info(f"Fallback triggered due to error: {e}")
                result = await self.fallback.generate(prompt, options)
                if not result.model:
                    result.model = self.fallback.get_name()
                return result
            raise e

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        try:
            async for chunk in self.primary.generate_stream(prompt, options):
                yield chunk
        except Exception as e:
            if self._should_fallback(e):
                logger.info(f"Streaming fallback triggered due to error: {e}")
                async for chunk in self.fallback.generate_stream(prompt, options):
                    yield chunk
                return
            raise e

    def get_name(self) -> str:
        return f"{self.primary.get_name()} -> {self.fallback.get_name()}"


def get_provider(provider_type: ProviderType | None = None) -> LLMProvider:
    target_type = provider_type or config.provider

    logger.debug(f"Resolving provider for type: {target_type}")

    if target_type != "gemini":
        logger.error(f"Unsupported provider type: {target_type}")
        raise ValueError(f"Unsupported provider type: {target_type}")

    keys = config.gemini_api_keys

    if not keys:
        logger.error("GEMINI_API_KEYS is missing in environment variables")
        raise ValueError("GEMINI_API_KEYS is missing in environment variables")

    def create_provider_chain(model_name: str | None = None):
        instances = []
        for key in keys:
            p = GeminiProvider(key)
            if model_name:
                p.model_name = model_name
            instances.append(p)

        return RoundRobinKeyProvider(instances) if len(instances) > 1 else instances[0]

    primary_model = config.gemini_model
    fallback_model = config.gemini_fallback_model

    primary = create_provider_chain(primary_model)

    if fallback_model:
        logger.info(
            f"Using primary model: {primary_model} with fallback: {fallback_model}"
        )
        fallback = create_provider_chain(fallback_model)
        return FallbackProvider(primary, fallback)

    logger.info(f"Using primary model: {primary_model}")
    return primary


class LLMFactory:
    @staticmethod
    def get_provider(provider_type: ProviderType | None = None) -> LLMProvider:
        return get_provider(provider_type)
