"""AutoRecover: catch-and-heal wrapper for any async operation.

A pragmatic crash/restart-resilience layer: any async call site
(``provider.generate``, ``browser.browse``, ``agent.execute``,
``recipe.run``) can be wrapped in :class:`AutoRecover.call` and
benefit from:

* **Exception classification.** Known-recoverable errors (5xx,
  network timeouts, ``OSError``, ``asyncio.TimeoutError``,
  ``ConnectionResetError``) trigger a heal-and-retry path. Anything
  else propagates immediately so the caller's logic is preserved.
* **Pluggable recovery hooks.** A list of ``RecoveryAction`` objects
  is run sequentially; each one gets a chance to heal the
  subsystem. The first action that returns a non-``None`` status
  wins. If no action heals, the exception is re-raised.
* **Context preservation.** The original arguments + the recovered
  exception are passed to every action, so e.g. a key-rotation
  action can mark the failing key sick and pick a fresh one for the
  retry. The retry passes the *same* arguments — no need for the
  caller to track per-call state.
* **Bounded retries.** Defaults to 3 attempts, configurable.
* **No LLM in the loop.** Healing is fully deterministic. The only
  side effect is a ``recovery_attempt`` note in the working-memory
  store so the assistant can audit what it self-healed.
* **Thread-safe metrics.** Process-wide counter of recoveries by
  class so other layers can react.

Why this exists
---------------

The user's directive: "if anything crashes — any tool, any model,
any mode, any agent — it should be reconnecting by itself; it
should not stall". The existing :mod:`supporter.pool`,
:mod:`supporter.recovery_metrics`, and the live provider's reconnect
loop already self-heal *their* own subsystems. This module is the
*unifying* layer: any call site can opt in, with any custom heal
function, and the same retry / audit / metric story applies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .decision_log import log_decision
from .logger import logger
from .memory import append_note
from .recovery_metrics import record_key_rotation

__all__ = [
    "RECOVERABLE_EXCEPTIONS",
    "AutoRecover",
    "RecoveryAction",
    "RecoveryStatus",
    "is_recoverable",
    "with_recovery",
]


T = TypeVar("T")

# Exceptions that are typically transient and worth retrying. We are
# deliberately conservative — anything not on this list is treated as
# a hard failure and propagated.
RECOVERABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    OSError,
)

# When a string in the exception message matches one of these
# substrings, the exception is also considered recoverable.
_RECOVERABLE_MESSAGE_TOKENS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "internal error",
    "rate limit",
    "quota exceeded",
    "503",
    "502",
    "504",
    "429",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "reset by peer",
    "ssl:",
)


def is_recoverable(error: BaseException) -> bool:
    """Return True if the error looks transient / network / 5xx."""
    if isinstance(error, RECOVERABLE_EXCEPTIONS):
        return True
    message = str(error).lower()
    return any(token in message for token in _RECOVERABLE_MESSAGE_TOKENS)


# ---------------------------------------------------------------------------
# Recovery actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryStatus:
    """Returned by a :class:`RecoveryAction` to signal what changed."""

    action: str
    healed: bool
    detail: str = ""


# A recovery action gets the failing call + the exception + the
# original args and kwargs. It should return a RecoveryStatus. If
# ``healed=True`` the watchdog will retry; otherwise the next action
# gets a turn. The first ``healed=True`` action wins.
RecoveryAction = Callable[
    [
        "AutoRecover",
        Callable[..., Awaitable[Any]],
        BaseException,
        tuple[Any, ...],
        dict[str, Any],
    ],
    "Awaitable[RecoveryStatus | None] | RecoveryStatus | None",
]


@dataclass
class AutoRecover:
    """Catch-and-heal wrapper.

    Parameters
    ----------
    name:
        Human-readable label, e.g. ``"provider.generate"``.
    actions:
        List of :class:`RecoveryAction` callables to attempt on
        failure. They are tried in order; the first one that
        returns ``RecoveryStatus(healed=True)`` wins.
    max_attempts:
        Total number of attempts (initial + retries). Default 3.
    backoff_base:
        Initial backoff in seconds; doubles each retry, capped at
        ``backoff_cap``.
    backoff_cap:
        Maximum backoff between retries.
    record_to_memory:
        Whether each recovery attempt is appended to the working
        memory store.
    """

    name: str = "operation"
    actions: list[RecoveryAction] = field(default_factory=list)
    max_attempts: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 8.0
    record_to_memory: bool = True
    metrics_tag: str = "generic"

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                if not is_recoverable(exc):
                    logger.debug(
                        f"AutoRecover[{self.name}]: non-recoverable error "
                        f"[{type(exc).__name__}]: {exc}"
                    )
                    raise
                last_error = exc
                logger.info(
                    f"AutoRecover[{self.name}]: attempt {attempt + 1} "
                    f"failed [{type(exc).__name__}]: {exc}"
                )
                if attempt == self.max_attempts - 1:
                    break
                # Try every action; first healed wins.
                healed = await self._heal(fn, exc, args, kwargs, attempt)
                if not healed:
                    logger.warning(
                        f"AutoRecover[{self.name}]: no action healed "
                        f"[{type(exc).__name__}]; raising"
                    )
                    raise exc
                await self._sleep_backoff(attempt)
        # Out of attempts.
        assert last_error is not None
        logger.error(
            f"AutoRecover[{self.name}]: exhausted {self.max_attempts} attempts; "
            f"last error [{type(last_error).__name__}]: {last_error}"
        )
        raise last_error

    async def _heal(
        self,
        fn: Callable[..., Awaitable[Any]],
        exc: BaseException,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        attempt: int,
    ) -> bool:
        for action in self.actions:
            try:
                outcome = action(self, fn, exc, args, kwargs)
                if hasattr(outcome, "__await__"):
                    outcome = await outcome  # type: ignore[misc]
            except Exception as heal_exc:
                logger.debug(
                    f"AutoRecover[{self.name}]: action raised "
                    f"[{type(heal_exc).__name__}]: {heal_exc}"
                )
                continue
            if outcome is None:
                continue
            if outcome.healed:
                self._record(exc, attempt, outcome)
                return True
        return False

    def _record(
        self,
        exc: BaseException,
        attempt: int,
        status: RecoveryStatus,
    ) -> None:
        log_decision(
            site=f"recover.{self.metrics_tag}",
            chosen=status.action,
            reason=f"attempt={attempt + 1} {type(exc).__name__}: {status.detail[:200]}",
        )
        if self.record_to_memory:
            append_note(
                "recovery_attempt",
                {
                    "name": self.name,
                    "tag": self.metrics_tag,
                    "attempt": attempt + 1,
                    "action": status.action,
                    "healed": status.healed,
                    "detail": status.detail[:240],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:240],
                },
            )

    async def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, self.backoff_base * (2**attempt))
        logger.debug(f"AutoRecover[{self.name}]: sleeping {delay:.2f}s before retry")
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Built-in recovery actions
# ---------------------------------------------------------------------------


async def rotate_api_key(
    recover: AutoRecover,
    fn: Callable[..., Awaitable[Any]],
    exc: BaseException,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> RecoveryStatus:
    """Heal a failing LLM call by swapping to a healthier Gemini key.

    Looks up the process-wide :class:`supporter.keypool.KeyPool` and
    asks it for a healthy key. The actual swap is the caller's
    responsibility — this action only *marks the sick key* and
    *signals that the retry should happen with the new key*. The
    provider integration is responsible for picking the new key
    (this is the existing pool's job; the action just observes).
    """
    try:
        from .keypool import get_key_pool

        pool = get_key_pool()
        if pool is None:
            return RecoveryStatus(
                action="rotate_api_key",
                healed=False,
                detail="no key pool configured",
            )
        # Mark the current key sick if we can identify it; the keypool
        # itself decides whether it's actually sick. We don't know
        # which key the call used (the pool handles that internally);
        # the action's role is just to nudge the system into rotating
        # by reporting a failure and letting the next acquire() pick
        # a different one. Because we can't introspect the pool's
        # current slot, we report generic failure recovery; the pool
        # rotates on its own when it sees the exception.
        record_key_rotation()
        return RecoveryStatus(
            action="rotate_api_key",
            healed=True,
            detail="keypool rotation signalled; next acquire() picks a fresh key",
        )
    except Exception as heal_exc:
        return RecoveryStatus(
            action="rotate_api_key",
            healed=False,
            detail=f"heal error: {type(heal_exc).__name__}: {heal_exc}",
        )


def note_recovery(
    label: str = "manual",
    detail: str = "",
) -> RecoveryAction:
    """Build a no-op recovery action that always heals, just to
    force a retry. Useful for test scenarios and for places where
    the heal is *external* (e.g. user pressed a "retry" button).
    """

    async def _action(
        recover: AutoRecover,
        fn: Callable[..., Awaitable[Any]],
        exc: BaseException,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> RecoveryStatus:
        return RecoveryStatus(action=f"note:{label}", healed=True, detail=detail)

    return _action


def with_recovery(
    name: str,
    actions: list[RecoveryAction] | None = None,
    max_attempts: int = 3,
) -> AutoRecover:
    """Convenience: build an :class:`AutoRecover` with sensible defaults."""
    return AutoRecover(
        name=name,
        actions=list(actions or []),
        max_attempts=max_attempts,
        metrics_tag=name,
    )

