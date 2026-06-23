from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from typing import Any, ClassVar

from .config import (
    HTTP_RATE_LIMIT,
    config,
)
from .llm.types import GenOptions, Message
from .logger import logger
from .providers.gemini_provider import GeminiProvider
from .providers.registry import PROVIDER_FACTORIES
from .types import LLMChunk, LLMProvider, LLMResult

__all__ = [
    "DynamicPool",
    "LLMChunk",
    "LLMProvider",
    "LLMResult",
    "LazyFallbackProvider",
    "clear_providers",
    "config",
    "get_provider",
    "is_model_error",
    "is_rate_limit",
    "reset_model_cooldowns",
    "should_trigger_fallback",
]


def __getattr__(name: str) -> Any:
    if name == "GeminiLiveProvider":
        from .providers.gemini_live_provider import GeminiLiveProvider

        return GeminiLiveProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_model_cooldowns: OrderedDict[str, datetime] = OrderedDict()
_provider_registry: dict[str, LLMProvider] = {}
_provider_lock = threading.Lock()


def _genai_error_classes() -> tuple[type[BaseException], ...]:
    from .providers.gemini_codec import gemini_error_classes

    return gemini_error_classes()


def _status_code(error: Any) -> int | None:
    code = getattr(error, "status", None) or getattr(error, "code", None)
    if isinstance(code, int):
        return code
    return None


def is_rate_limit(error: Any) -> bool:
    if _status_code(error) == HTTP_RATE_LIMIT:
        return True
    message = str(error).lower()
    return any(sig in message for sig in config.rate_limit_error_strings)


def is_model_error(error: Any, model_name: str | None = None) -> bool:
    genai_errors = _genai_error_classes()
    if genai_errors and isinstance(error, genai_errors):
        status = _status_code(error)
        if status is not None and status in config.http_5xx_status_codes:
            return True

    error_class_name = error.__class__.__name__
    if error_class_name in config.google_api_5xx_exceptions:
        return True

    if _status_code(error) in config.http_5xx_status_codes:
        return True

    message = str(error).lower()
    if any(sig in message for sig in config.transient_error_strings):
        return True

    return any(str(code) in message for code in config.http_5xx_status_codes)


def should_trigger_fallback(error: Any) -> bool:
    return is_rate_limit(error) or is_model_error(error)


def _notify_keypool_failure(provider: LLMProvider, error: BaseException) -> None:
    """Best-effort: tell the keypool that ``provider``'s key just failed.

    Failures are recorded to the keypool so DynamicPool's slot selection
    (``_select_next_key_index``) can consult that health data and avoid
    refilling a slot with a cooling-down key.
    Safe to call when the keypool is unconfigured (no-op) or when
    ``provider`` is not a GeminiProvider.
    """
    try:
        from .keypool import (
            get_key_pool,
            reset_key_pool,  # noqa: F401  (kept for symmetry)
        )

        api_key = getattr(provider, "api_key", None)
        if not isinstance(api_key, str) or not api_key:
            return
        pool = get_key_pool()
        if pool is None:
            return
        pool.report_failure(api_key, error)
    except Exception as exc:
        logger.debug(f"keypool notification skipped [{type(exc).__name__}]: {exc}")


def _prune_expired_cooldowns() -> None:
    now = datetime.now()
    with _provider_lock:
        expired = [name for name, exp in _model_cooldowns.items() if now > exp]
        for name in expired:
            del _model_cooldowns[name]
            logger.info(f"Model '{name}' cooldown expired — re-enabling")


def _is_model_in_cooldown(model_name: str) -> bool:
    _prune_expired_cooldowns()
    with _provider_lock:
        return model_name in _model_cooldowns


def _mark_model_cooldown(model_name: str, minutes: int = 30) -> None:
    expiry = datetime.now() + timedelta(minutes=minutes)
    with _provider_lock:
        _model_cooldowns.pop(model_name, None)
        _model_cooldowns[model_name] = expiry
    logger.info(
        f"Model '{model_name}' placed in cooldown until "
        f"{expiry.strftime('%Y-%m-%d %H:%M:%S')} — repeated 5XX errors"
    )


def reset_model_cooldowns() -> None:
    """Clear all model cooldowns in place (test isolation / reconfiguration)."""
    global _model_cooldowns
    with _provider_lock:
        _model_cooldowns.clear()


class DynamicPool(LLMProvider):
    _instances: ClassVar[weakref.WeakSet[DynamicPool]] = weakref.WeakSet()

    @classmethod
    async def shutdown_all(cls) -> None:
        for pool in list(cls._instances):
            await pool.shutdown()

    def __init__(
        self,
        keys: list[str],
        model_name: str,
        pool_size: int = 2,
        provider_factory: Callable[[str, str], LLMProvider] | None = None,
    ):
        self.keys = keys
        self.model_name = model_name
        self.pool_size = min(pool_size, len(keys))
        self.next_key_index = 0
        self.active_slots: deque[LLMProvider] = deque()
        self._current_index = 0
        self.background_tasks: set[asyncio.Task[None]] = set()
        self._replace_lock = asyncio.Semaphore(1)
        self._provider_factory = provider_factory or (
            lambda key, model: GeminiProvider(key, model_name=model)
        )
        self._instances.add(self)

    def _select_next_key_index(self) -> int:
        # Best-effort keypool consult; ANY failure -> plain round-robin (lossless).
        pool = None
        try:
            from .keypool import get_key_pool

            pool = get_key_pool()
        except Exception:
            pool = None
        n = len(self.keys)
        start = self.next_key_index % n
        if pool is None:
            self.next_key_index = (start + 1) % n
            return start
        try:
            now = time.time()
            best_sick_idx = start
            best_recovery = float("inf")
            for off in range(n):
                idx = (start + off) % n
                health = pool.health(self.keys[idx])
                if health.is_available(now):
                    self.next_key_index = (idx + 1) % n
                    return idx
                rec = health.seconds_to_recovery(now)
                if rec < best_recovery:
                    best_recovery = rec
                    best_sick_idx = idx
            # All keys cooling down -> least-sick (earliest recovery).
            self.next_key_index = (best_sick_idx + 1) % n
            return best_sick_idx
        except Exception:
            # keypool consult blew up mid-scan -> plain round-robin.
            self.next_key_index = (start + 1) % n
            return start

    def _fill_slot(self) -> None:
        idx = self._select_next_key_index()
        key = self.keys[idx]
        provider = self._provider_factory(key, self.model_name)
        self.active_slots.append(provider)

    def _rotate_and_get(self) -> LLMProvider:
        if len(self.active_slots) < self.pool_size:
            self._fill_slot()

        idx = self._current_index % len(self.active_slots)
        provider = self.active_slots[idx]
        self._current_index += 1
        return provider

    def _replace_instance(self, provider: LLMProvider) -> None:
        try:
            self.active_slots.remove(provider)
        except ValueError:
            return

        logger.info(
            f"Pool '{self.model_name}': provider instance retired after failure — "
            f"scheduling replacement (slots remaining: {len(self.active_slots)})"
        )

        async def _bg_replace() -> None:
            async with self._replace_lock:
                await asyncio.sleep(0)
                if len(self.active_slots) < self.pool_size:
                    self._fill_slot()

        coro = _bg_replace()
        try:
            task = asyncio.create_task(coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        except RuntimeError:
            coro.close()
            self._fill_slot()

    async def _backoff(self, attempt: int, base: float = 0.5) -> None:
        delay = min(10.0, base * (2**attempt))
        logger.info(
            f"Pool '{self.model_name}': backing off for {delay:.2f}s "
            f"(attempt {attempt + 1})"
        )
        await asyncio.sleep(delay)

    async def shutdown(self) -> None:
        if self.background_tasks:
            logger.info(
                f"Pool '{self.model_name}' shutdown: draining "
                f"{len(self.background_tasks)} background task(s)"
            )
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()

    async def close(self) -> None:
        await self.shutdown()

    async def generate(
        self, prompt: str | list[Message], options: GenOptions | None = None
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
                result: LLMResult = await provider.generate(prompt, options)

                if not result.model:
                    result.model = provider.get_name()
                return result

            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    if is_model_error(e, self.model_name):
                        logger.info(
                            f"[{self.model_name}] 5XX on attempt {attempt + 1}: {e}"
                        )
                        _mark_model_cooldown(self.model_name)
                    else:
                        logger.info(
                            f"[{self.model_name}] Retriable error attempt "
                            f"{attempt + 1}: {e}"
                        )
                    _notify_keypool_failure(provider, e)
                    self._replace_instance(provider)
                    if is_rate_limit(e):
                        continue
                    await self._backoff(attempt)
                    continue
                raise e

        if last_error:
            raise last_error
        raise RuntimeError("Unexpected: no provider available")

    async def generate_stream(
        self, prompt: str | list[Message], options: GenOptions | None = None
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
                async for chunk in provider.generate_stream(prompt, options):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                last_error = e
                if should_trigger_fallback(e):
                    if is_model_error(e, self.model_name):
                        logger.info(
                            f"[{self.model_name}] 5XX error during stream attempt "
                            f"{attempt + 1}: {e}"
                        )
                        _mark_model_cooldown(self.model_name)
                    else:
                        logger.info(
                            f"[{self.model_name}] Retriable stream error attempt "
                            f"{attempt + 1}: {e}"
                        )
                    _notify_keypool_failure(provider, e)
                    self._replace_instance(provider)
                    if not yielded_any:
                        logger.info(
                            f"[{self.model_name}] Stream failed before first chunk — "
                            f"retrying (attempt {attempt + 1}/{len(self.keys)})"
                        )
                        if is_rate_limit(e):
                            continue
                        await self._backoff(attempt)
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
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> LLMResult:
        try:
            return await self.primary.generate(prompt, options)
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.info(
                f"Primary pool exhausted ({type(e).__name__}: {e}) — "
                "triggering fallback"
            )
            return await self.fallback.generate(prompt, options)

    async def generate_stream(
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        try:
            async for chunk in self.primary.generate_stream(prompt, options):
                yield chunk
        except Exception as e:
            if not should_trigger_fallback(e) or not self.fallback:
                raise e
            logger.info(
                f"Primary stream failed ({type(e).__name__}: {e}) — triggering fallback"
            )
            async for chunk in self.fallback.generate_stream(prompt, options):
                yield chunk

    def get_name(self) -> str:
        if self.fallback_factory:
            return f"{self.primary.get_name()} -> [Fallback]"
        return self.primary.get_name()

    async def close(self) -> None:
        for provider in (self._primary, self._fallback):
            close_fn = getattr(provider, "close", None)
            if close_fn:
                await close_fn()


async def clear_providers() -> None:
    logger.info("Clearing provider registry")
    for provider in list(_provider_registry.values()):
        close_fn = getattr(provider, "close", None)
        if close_fn:
            try:
                await close_fn()
            except Exception:
                logger.debug("Error closing provider", exc_info=True)
    _provider_registry.clear()


def get_provider(
    provider_type: str | None = None,
    live: bool = False,
    shared: bool = True,
    model_name: str | None = None,
    registry: dict[str, Callable[..., Any]] | None = None,
    system_instruction: str | None = None,
    pool_size: int = 2,
    keys: list[str] | None = None,
) -> LLMProvider:
    target_type = provider_type or config.provider
    cache_key = f"{target_type}_{live}_{model_name or 'default'}"

    with _provider_lock:
        if shared and cache_key in _provider_registry:
            provider = _provider_registry[cache_key]
            if live and registry and hasattr(provider, "registry"):
                provider.registry.update(registry)
            return provider

        if target_type not in PROVIDER_FACTORIES:
            registered = ", ".join(sorted(PROVIDER_FACTORIES.keys()))
            raise ValueError(
                f"Unsupported provider type: {target_type!r}. Registered: {registered}"
            )

        provider = PROVIDER_FACTORIES[target_type](
            keys=keys if keys is not None else config.gemini_api_keys,
            model_name=model_name,
            pool_size=pool_size,
            registry=registry,
            system_instruction=system_instruction,
            live=live,
        )

        if shared:
            _provider_registry[cache_key] = provider

        return provider
