from dataclasses import dataclass, field
import asyncio
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

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

logger.debug("--- Loading index module ---")


class LLMOptions(TypedDict, total=False):
    history: List[Content]
    model: str
    tools: List[Tool]
    registry: Dict[str, Callable]
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
    model: Optional[str] = None
    duration: Optional[float] = None
    interaction_id: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None
    automatic_function_calling_history: Optional[List[Content]] = None
    candidates: List[Any] = field(default_factory=list)


@dataclass
class LLMChunk:
    text: str
    is_last: bool
    model: Optional[str] = None
    raw: Any = None


class LLMProvider(Protocol):
    async def generate(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
    ) -> LLMResult: ...

    async def generate_stream(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
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


class DynamicPool(LLMProvider):
    def __init__(self, keys: List[str], model_name: str, pool_size: int = 2):
        self.keys = keys
        self.model_name = model_name
        self.pool_size = min(pool_size, len(keys))
        self.next_key_index = 0
        self.active_slots: List[GeminiProvider] = []

        logger.debug(
            f"Initializing DynamicPool (size: {self.pool_size}) for {self.model_name}"
        )
        for _ in range(self.pool_size):
            self._fill_slot()

    def _fill_slot(self) -> None:
        key = self.keys[self.next_key_index % len(self.keys)]
        logger.debug(
            f"  └─ Creating instance with key index {self.next_key_index % len(self.keys)}"
        )
        self.next_key_index += 1
        provider = GeminiProvider(key, model_name=self.model_name)
        self.active_slots.append(provider)

    def _rotate_and_get(self) -> GeminiProvider:
        if not self.active_slots:
            logger.debug(f"Both active slots empty. Filling synchronously.")
            self._fill_slot()

        provider = self.active_slots.pop(0)
        self.active_slots.append(provider)
        return provider

    def _replace_instance(self, provider: GeminiProvider) -> None:
        try:
            self.active_slots.remove(provider)
            logger.warning(
                f"Instance failed in pool {self.model_name}. Destroying and replacing in background..."
            )

            async def _bg_replace():
                await asyncio.sleep(
                    0
                )
                logger.debug(f"Background task: Refilling slot for {self.model_name}")
                self._fill_slot()

            try:
                asyncio.create_task(_bg_replace())
            except RuntimeError:
                self._fill_slot()

        except ValueError:
            pass

    async def generate(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
    ) -> LLMResult:
        logger.debug(f"Entering DynamicPool.generate ({self.model_name})")
        last_error = None
        for attempt in range(len(self.keys)):
            while not self.active_slots:
                await asyncio.sleep(0.1)

            provider = self._rotate_and_get()
            try:
                result = await provider.generate(prompt, options)
                if not result.model:
                    result.model = provider.get_name()
                return result
            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    self._replace_instance(provider)
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise e
        raise last_error

    async def generate_stream(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
    ) -> AsyncIterator[LLMChunk]:
        logger.debug(f"Entering DynamicPool.generate_stream ({self.model_name})")
        last_error = None
        for attempt in range(len(self.keys)):
            while not self.active_slots:
                await asyncio.sleep(0.1)

            provider = self._rotate_and_get()
            yielded_any = False
            try:
                async for chunk in provider.generate_stream(prompt, options):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    self._replace_instance(provider)
                    if not yielded_any:
                        logger.warning(
                            f"Stream failed before first chunk in pool {self.model_name} (Attempt {attempt + 1}/{len(self.keys)}). Retrying..."
                        )
                        await asyncio.sleep(0.2 * (attempt + 1))
                        continue
                raise e
        raise last_error

    def get_name(self) -> str:
        return f"{self.model_name} (Dynamic Pool x{len(self.active_slots)})"


class LazyFallbackProvider(LLMProvider):
    def __init__(
        self,
        primary_factory: Callable[[], LLMProvider],
        fallback_factory: Optional[Callable[[], Optional[LLMProvider]]] = None,
    ):
        self.primary_factory = primary_factory
        self.fallback_factory = fallback_factory
        self._primary: Optional[LLMProvider] = None
        self._fallback: Optional[LLMProvider] = None
        logger.debug("Initializing LazyFallbackProvider")

    @property
    def primary(self) -> LLMProvider:
        if self._primary is None:
            logger.debug("Lazy-initializing primary provider pool")
            self._primary = self.primary_factory()
        return self._primary

    @property
    def fallback(self) -> Optional[LLMProvider]:
        if self._fallback is None and self.fallback_factory:
            logger.debug("Lazy-initializing fallback provider pool")
            self._fallback = self.fallback_factory()
        return self._fallback

    async def generate(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
    ) -> LLMResult:
        logger.debug("Entering LazyFallbackProvider.generate")
        try:
            result = await self.primary.generate(prompt, options)
            return result
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.warning("Primary pool exhausted/failed. Triggering lazy fallback...")
            return await self.fallback.generate(prompt, options)

    async def generate_stream(
        self, prompt: Union[str, List[Content]], options: Optional[LLMOptions] = None
    ) -> AsyncIterator[LLMChunk]:
        logger.debug("Entering LazyFallbackProvider.generate_stream")
        try:
            async for chunk in self.primary.generate_stream(prompt, options):
                yield chunk
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.warning("Primary stream failed. Triggering lazy fallback...")
            async for chunk in self.fallback.generate_stream(prompt, options):
                yield chunk

    def get_name(self) -> str:
        if self.fallback:
            return f"{self.primary.get_name()} -> [Lazy Fallback]"
        return self.primary.get_name()


def get_provider(provider_type: Optional[ProviderType] = None) -> LLMProvider:
    logger.debug(f"get_provider called with type: {provider_type}")
    target_type = provider_type or config.provider
    if target_type != "gemini":
        raise ValueError(f"Unsupported provider type: {target_type}")

    keys = config.gemini_api_keys
    if not keys:
        raise ValueError("GEMINI_API_KEYS is missing/empty in environment")

    def primary_factory() -> LLMProvider:
        return DynamicPool(keys, config.gemini_model, pool_size=2)

    def fallback_factory() -> Optional[LLMProvider]:
        if not config.gemini_fallback_model:
            return None
        return DynamicPool(keys, config.gemini_fallback_model, pool_size=2)

    return LazyFallbackProvider(primary_factory, fallback_factory)
