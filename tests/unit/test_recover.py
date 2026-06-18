"""Unit tests for the AutoRecover watchdog."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from supporter.recover import (
    RECOVERABLE_EXCEPTIONS,
    AutoRecover,
    RecoveryStatus,
    is_recoverable,
    note_recovery,
    rotate_api_key,
    with_recovery,
)

# ---------------------------------------------------------------------------
# is_recoverable
# ---------------------------------------------------------------------------


def test_is_recoverable_timeout() -> None:
    assert is_recoverable(TimeoutError())


def test_is_recoverable_connection_reset() -> None:
    assert is_recoverable(ConnectionResetError("reset"))


def test_is_recoverable_oserror() -> None:
    assert is_recoverable(OSError("disk"))


def test_is_recoverable_message_token() -> None:
    assert is_recoverable(RuntimeError("rate limit exceeded"))
    assert is_recoverable(RuntimeError("503 service unavailable"))
    assert is_recoverable(RuntimeError("connection refused"))


def test_is_not_recoverable_value_error() -> None:
    assert not is_recoverable(ValueError("bad input"))


def test_is_not_recoverable_key_error() -> None:
    assert not is_recoverable(KeyError("missing"))


def test_recoverable_exceptions_includes_core_set() -> None:
    assert asyncio.TimeoutError in RECOVERABLE_EXCEPTIONS
    assert ConnectionError in RECOVERABLE_EXCEPTIONS
    assert OSError in RECOVERABLE_EXCEPTIONS


# ---------------------------------------------------------------------------
# AutoRecover.call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_passes_through_on_success() -> None:
    recover = AutoRecover(name="op")
    fn = AsyncMock(return_value="ok")
    result = await recover.call(fn)
    assert result == "ok"
    assert fn.await_count == 1


@pytest.mark.asyncio
async def test_call_non_recoverable_propagates() -> None:
    recover = AutoRecover(name="op")
    fn = AsyncMock(side_effect=ValueError("bad"))
    with pytest.raises(ValueError):
        await recover.call(fn)
    assert fn.await_count == 1


@pytest.mark.asyncio
async def test_call_retries_with_heal_action() -> None:
    recover = AutoRecover(
        name="op",
        actions=[note_recovery("always", "ok")],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "second-attempt"])
    result = await recover.call(fn)
    assert result == "second-attempt"
    assert fn.await_count == 2


@pytest.mark.asyncio
async def test_call_exhausts_attempts() -> None:
    recover = AutoRecover(
        name="op",
        actions=[note_recovery("never", "ok")],
        max_attempts=3,
        backoff_base=0,
        backoff_cap=0,
    )
    # Action always heals, so we should get max_attempts calls.
    fn = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(asyncio.TimeoutError):
        await recover.call(fn)
    assert fn.await_count == 3


@pytest.mark.asyncio
async def test_call_no_action_heals_propagates() -> None:
    recover = AutoRecover(
        name="op",
        actions=[],
        max_attempts=2,
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(asyncio.TimeoutError):
        await recover.call(fn)
    # No heal → propagate after the first attempt.
    assert fn.await_count == 1


@pytest.mark.asyncio
async def test_call_first_heal_wins() -> None:
    calls = {"first": 0, "second": 0}

    async def _a1(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["first"] += 1
        return RecoveryStatus(action="a1", healed=True, detail="first")

    async def _a2(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["second"] += 1
        return RecoveryStatus(action="a2", healed=True, detail="second")

    recover = AutoRecover(
        name="op",
        actions=[_a1, _a2],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    assert calls["first"] == 1
    assert calls["second"] == 0


@pytest.mark.asyncio
async def test_call_falls_through_to_second_action() -> None:
    calls = {"first": 0, "second": 0}

    async def _a1(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["first"] += 1
        return RecoveryStatus(action="a1", healed=False, detail="no")

    async def _a2(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["second"] += 1
        return RecoveryStatus(action="a2", healed=True, detail="yes")

    recover = AutoRecover(
        name="op",
        actions=[_a1, _a2],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    assert calls["first"] == 1
    assert calls["second"] == 1


@pytest.mark.asyncio
async def test_call_records_to_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    memory_mod._MEMORY_SINGLETON = None

    recover = AutoRecover(
        name="op",
        actions=[note_recovery("manual")],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    notes = memory_mod.list_notes(kind="recovery_attempt")
    assert any(n.value.get("action", "").startswith("note:manual") for n in notes)


@pytest.mark.asyncio
async def test_call_skips_memory_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    memory_mod._MEMORY_SINGLETON = None

    recover = AutoRecover(
        name="op",
        actions=[note_recovery("manual")],
        backoff_base=0,
        backoff_cap=0,
        record_to_memory=False,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    assert memory_mod.list_notes(kind="recovery_attempt") == []


@pytest.mark.asyncio
async def test_call_action_exception_continues_to_next() -> None:
    calls = {"a": 0, "b": 0}

    async def _bad(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["a"] += 1
        raise RuntimeError("action boom")

    async def _good(*args: Any, **kwargs: Any) -> RecoveryStatus:
        calls["b"] += 1
        return RecoveryStatus(action="b", healed=True, detail="ok")

    recover = AutoRecover(
        name="op",
        actions=[_bad, _good],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    assert calls["a"] == 1
    assert calls["b"] == 1


@pytest.mark.asyncio
async def test_call_action_returning_none_continues() -> None:
    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    async def _good(*args: Any, **kwargs: Any) -> RecoveryStatus:
        return RecoveryStatus(action="g", healed=True, detail="ok")

    recover = AutoRecover(
        name="op",
        actions=[_noop, _good],
        backoff_base=0,
        backoff_cap=0,
    )
    fn = AsyncMock(side_effect=[TimeoutError(), "ok"])
    await recover.call(fn)
    assert fn.await_count == 2


# ---------------------------------------------------------------------------
# Built-in actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotate_api_key_with_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import keypool

    monkeypatch.setattr(keypool, "_default_state_path", lambda: tmp_path / "kp.json")
    monkeypatch.setattr(keypool.config, "gemini_api_keys", ["k1", "k2"])
    keypool.reset_key_pool()
    pool = keypool.get_key_pool()
    assert pool is not None
    recover = AutoRecover(name="op", actions=[rotate_api_key])
    fn = AsyncMock()
    status = await rotate_api_key(recover, fn, TimeoutError(), (), {})
    assert status.healed is True
    assert "rotation" in status.detail
    keypool.reset_key_pool()


@pytest.mark.asyncio
async def test_rotate_api_key_without_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from supporter import keypool

    keypool.reset_key_pool()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", [])
    recover = AutoRecover(name="op", actions=[rotate_api_key])
    fn = AsyncMock()
    status = await rotate_api_key(recover, fn, TimeoutError(), (), {})
    assert status.healed is False
    keypool.reset_key_pool()


# ---------------------------------------------------------------------------
# with_recovery helper
# ---------------------------------------------------------------------------


def test_with_recovery_defaults() -> None:
    recover = with_recovery("op")
    assert recover.name == "op"
    assert recover.max_attempts == 3
    assert recover.actions == []


def test_with_recovery_custom() -> None:
    action = note_recovery("manual")
    recover = with_recovery("op", actions=[action], max_attempts=5)
    assert recover.max_attempts == 5
    assert recover.actions[0] is action
