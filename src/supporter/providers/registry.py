"""Provider factory registry.

To add a provider: implement a class satisfying ``LLMProvider`` using only the
neutral types (``Message``/``Part`` in, ``LLMResult``/``LLMChunk`` out) — keep
any vendor SDK behind a codec in this package — then register a factory here. No
other file changes; ``get_provider`` dispatches by ``PROVIDER_FACTORIES`` key.
See ``tests/integration/test_fake_provider.py`` for a zero-SDK example.

Register new providers by implementing a factory function with the signature::

    def my_factory(
        *,
        keys: list[str],
        model_name: str | None = None,
        pool_size: int = 2,
        registry: dict[str, Callable[..., Any]] | None = None,
        system_instruction: str | None = None,
        live: bool = False,
    ) -> LLMProvider: ...

then add it to ``PROVIDER_FACTORIES``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..types import LLMProvider

PROVIDER_FACTORIES: dict[str, Callable[..., LLMProvider]] = {}


def _gemini_factory(
    *,
    keys: list[str],
    model_name: str | None = None,
    pool_size: int = 2,
    registry: dict[str, Callable[..., Any]] | None = None,
    system_instruction: str | None = None,
    live: bool = False,
) -> LLMProvider:
    """Build a Gemini provider — REST DynamicPool or Live with fallback."""
    from ..logger import logger
    from ..pool import DynamicPool, LazyFallbackProvider, config
    from .gemini_live_provider import GeminiLiveProvider

    if not keys:
        raise ValueError("GEMINI_API_KEYS is missing/empty in environment")

    if live:
        target_model = model_name or config.gemini_live_model

        def _live_primary() -> LLMProvider:
            logger.info(f"Constructing Live Primary provider: model={target_model}")
            return GeminiLiveProvider(
                keys,
                model_name=target_model,
                registry=registry,
                system_instruction=system_instruction,
            )

        fallback_model = config.gemini_live_fallback_model
        live_fallback: Callable[[], LLMProvider | None] | None = None
        if fallback_model:

            def _mk_live_fallback() -> LLMProvider | None:
                logger.info(
                    f"Constructing Live Fallback provider: model={fallback_model}"
                )
                return GeminiLiveProvider(
                    keys,
                    model_name=fallback_model,
                    registry=registry,
                    system_instruction=system_instruction,
                )

            live_fallback = _mk_live_fallback

        return LazyFallbackProvider(_live_primary, live_fallback)

    target_model = model_name or config.gemini_model

    def _rest_primary() -> LLMProvider:
        return DynamicPool(keys, target_model, pool_size=pool_size)

    fallback_model = config.gemini_fallback_model  # type: ignore[assignment]
    rest_fallback: Callable[[], LLMProvider | None] | None = None
    if fallback_model:

        def _mk_rest_fallback() -> LLMProvider | None:
            return DynamicPool(keys, fallback_model, pool_size=pool_size)

        rest_fallback = _mk_rest_fallback

    return LazyFallbackProvider(_rest_primary, rest_fallback)


PROVIDER_FACTORIES["gemini"] = _gemini_factory
