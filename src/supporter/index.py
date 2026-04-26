import asyncio
import threading
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from typing import Any, ClassVar, cast

from google.genai.types import Content

from .config import (
    HTTP_RATE_LIMIT,
    config,
)
from .logger import logger
from .providers.gemini_live_provider import GeminiLiveProvider
from .providers.gemini_provider import GeminiProvider
from .types import LLMChunk, LLMOptions, LLMProvider, LLMResult

__all__ = [
    "DynamicPool",
    "GeminiLiveProvider",
    "GeminiProvider",
    "LLMChunk",
    "LLMOptions",
    "LLMProvider",
    "LLMResult",
    "LazyFallbackProvider",
    "clear_providers",
    "config",
    "get_provider",
    "is_model_error",
    "is_rate_limit",
    "should_trigger_fallback",
]


_model_cooldowns: dict[str, datetime] = {}
_provider_registry: dict[str, LLMProvider] = {}
_provider_lock = threading.Lock()


def is_rate_limit(error: Any) -> bool:
    if getattr(error, "status", None) == HTTP_RATE_LIMIT:
        return True
    message = str(error).lower()
    return any(sig in message for sig in config.rate_limit_signals)


def is_model_error(error: Any, model_name: str | None = None) -> bool:
    error_class_name = error.__class__.__name__
    if error_class_name in config.google_5xx_errors:
        return True

    status = getattr(error, "status", None)
    if status in config.http_errors_5xx:
        return True

    message = str(error).lower()
    if any(sig in message for sig in config.transient_signals):
        return True

    return any(code in message for code in ("503", "500", "502", "504"))


def should_trigger_fallback(error: Any) -> bool:
    return is_rate_limit(error) or is_model_error(error)


def _is_model_in_cooldown(model_name: str) -> bool:
    if model_name not in _model_cooldowns:
        return False
    if datetime.now() > _model_cooldowns[model_name]:
        del _model_cooldowns[model_name]
        logger.info(f"Model '{model_name}' cooldown expired, re-enabling")
        return False
    return True


def _mark_model_cooldown(model_name: str, minutes: int = 30) -> None:
    expiry = datetime.now() + timedelta(minutes=minutes)
    _model_cooldowns[model_name] = expiry
    logger.warning(
        f"Model '{model_name}' in cooldown until "
        f"{expiry.strftime('%Y-%m-%d %H:%M:%S')} due to 5XX error"
    )


class DynamicPool(LLMProvider):
    _instances: ClassVar[weakref.WeakSet["DynamicPool"]] = weakref.WeakSet()

    @classmethod
    async def shutdown_all(cls) -> None:

        for pool in list(cls._instances):
            await pool.shutdown()

    def __init__(
        self,
        keys: list[str],
        model_name: str,
        pool_size: int = 2,
    ):
        self.keys = keys
        self.model_name = model_name
        self.pool_size = min(pool_size, len(keys))
        self.next_key_index = 0
        self.active_slots: deque[GeminiProvider] = deque()
        self.slot_available = asyncio.Event()
        self._current_index = 0
        self.background_tasks: set[asyncio.Task[None]] = set()
        self._instances.add(self)

    def _fill_slot(self) -> None:
        key = self.keys[self.next_key_index % len(self.keys)]
        self.next_key_index += 1
        provider = GeminiProvider(key, model_name=self.model_name)
        self.active_slots.append(provider)
        self.slot_available.set()

    def _rotate_and_get(self) -> GeminiProvider:
        if len(self.active_slots) < self.pool_size:
            self._fill_slot()

        idx = self._current_index % len(self.active_slots)
        provider = self.active_slots[idx]
        self._current_index += 1
        return provider

    def _replace_instance(self, provider: GeminiProvider) -> None:
        if provider not in self.active_slots:
            return

        self.active_slots.remove(provider)
        if not self.active_slots:
            self.slot_available.clear()

        logger.warning(
            f"Instance failed in pool {self.model_name}. "
            "Destroying and replacing in background..."
        )

        async def _bg_replace() -> None:
            await asyncio.sleep(0)
            self._fill_slot()

        coro = _bg_replace()
        try:
            task = asyncio.create_task(coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        except RuntimeError:
            coro.close()
            self._fill_slot()

    async def shutdown(self) -> None:
        if self.background_tasks:
            logger.info(
                f"Shutting down pool {self.model_name}, awaiting background tasks"
            )
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()
        return

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        if _is_model_in_cooldown(self.model_name):
            raise RuntimeError(
                f"Model '{self.model_name}' is in cooldown due to repeated "
                "5XX errors. Please try again later."
            )

        last_error: BaseException | None = None
        for attempt in range(len(self.keys)):
            provider = self._rotate_and_get()
            try:
                provider_options = cast(LLMOptions, dict(options)) if options else None
                result: LLMResult = await provider.generate(prompt, provider_options)

                if not result.model:
                    result.model = provider.get_name()
                return result

            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    if is_model_error(e, self.model_name):
                        logger.warning(
                            f"[{self.model_name}] 5XX error detected; marking cooldown."
                        )
                        _mark_model_cooldown(self.model_name)

                    self._replace_instance(provider)
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise e

        if last_error:
            raise last_error
        raise RuntimeError("Unexpected: no provider available")

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        if _is_model_in_cooldown(self.model_name):
            raise RuntimeError(
                f"Model '{self.model_name}' is in cooldown due to repeated "
                "5XX errors. Please try again later."
            )

        last_error: BaseException | None = None
        for attempt in range(len(self.keys)):
            provider = self._rotate_and_get()
            yielded_any = False
            try:
                provider_options = cast(LLMOptions, dict(options)) if options else None
                async for chunk in provider.generate_stream(prompt, provider_options):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    if is_model_error(e, self.model_name):
                        logger.warning(
                            f"[{self.model_name}] 5XX error detected during stream; "
                            "marking cooldown."
                        )
                        _mark_model_cooldown(self.model_name)

                    self._replace_instance(provider)
                    if not yielded_any:
                        logger.warning(
                            f"[{self.model_name}] Stream failed before first chunk "
                            f"(Attempt {attempt + 1}/{len(self.keys)}). Retrying..."
                        )
                        await asyncio.sleep(0.2 * (attempt + 1))
                        continue
                raise e

        if last_error:
            raise last_error

    def get_name(self) -> str:
        return f"{self.model_name} (Dynamic Pool x{len(self.active_slots)})"


class LazyFallbackProvider(LLMProvider):
    def __init__(
        self,
        primary_factory: Callable[[], LLMProvider],
        fallback_factory: Callable[[], LLMProvider | None] | None = None,
    ):
        self.primary_factory = primary_factory
        self.fallback_factory = fallback_factory
        self._primary: LLMProvider | None = None
        self._fallback: LLMProvider | None = None

    @property
    def primary(self) -> LLMProvider:
        if self._primary is None:
            self._primary = self.primary_factory()
        return self._primary

    @property
    def fallback(self) -> LLMProvider | None:
        if self._fallback is None and self.fallback_factory:
            self._fallback = self.fallback_factory()
        return self._fallback

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        try:
            return await self.primary.generate(prompt, options)
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.warning("Primary pool exhausted. Triggering fallback...")
            return await self.fallback.generate(prompt, options)

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        try:
            async for chunk in self.primary.generate_stream(prompt, options):
                yield chunk
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.warning("Primary stream failed. Triggering fallback...")
            async for chunk in self.fallback.generate_stream(prompt, options):
                yield chunk

    def get_name(self) -> str:
        if self.fallback:
            return f"{self.primary.get_name()} -> [Fallback]"
        return self.primary.get_name()


def clear_providers() -> None:
    logger.info("Clearing provider registry")
    _provider_registry.clear()


def get_provider(
    provider_type: str | None = None,
    live: bool = False,
    shared: bool = True,
    model_name: str | None = None,
    registry: dict[str, Callable[..., Any]] | None = None,
) -> LLMProvider:
    target_type = provider_type or config.provider
    cache_key = f"{target_type}_{live}_{model_name or 'default'}"

    if shared and cache_key in _provider_registry:
        logger.debug(f"Returning cached provider for {cache_key}")
        provider = _provider_registry[cache_key]
        if live and registry and hasattr(provider, "registry"):
            provider.registry.update(registry)
        return provider

    with _provider_lock:
        if shared and cache_key in _provider_registry:
            provider = _provider_registry[cache_key]
            if live and registry and hasattr(provider, "registry"):
                provider.registry.update(registry)
            return provider

        if target_type != "gemini":
            raise ValueError(f"Unsupported provider type: {target_type}")

        keys = config.gemini_api_keys
        if not keys:
            raise ValueError("GEMINI_API_KEYS is missing/empty in environment")

        if live:

            def live_primary_factory() -> LLMProvider:
                target_model = model_name or config.gemini_live_model
                logger.info(f"Creating Live Primary Provider: {target_model}")
                return GeminiLiveProvider(
                    keys, model_name=target_model, registry=registry
                )

            def live_fallback_factory() -> LLMProvider | None:
                if not config.gemini_live_fallback_model:
                    return None
                logger.info(
                    "Creating Live Fallback Provider: "
                    f"{config.gemini_live_fallback_model}"
                )
                return GeminiLiveProvider(
                    keys,
                    model_name=config.gemini_live_fallback_model,
                    registry=registry,
                )

            provider = LazyFallbackProvider(live_primary_factory, live_fallback_factory)
        else:

            def primary_factory() -> LLMProvider:
                target_model = model_name or config.gemini_model
                return DynamicPool(keys, target_model, pool_size=2)

            def fallback_factory() -> LLMProvider | None:
                if not config.gemini_fallback_model:
                    return None
                return DynamicPool(keys, config.gemini_fallback_model, pool_size=2)

            provider = LazyFallbackProvider(primary_factory, fallback_factory)

        if shared:
            _provider_registry[cache_key] = provider

        return provider
