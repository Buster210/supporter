from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from supporter.tools.browser import session
from tests.browser_fakes import make_session

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Hermetic isolation  (mirrors tests/unit/test_browser_session.py)
# ---------------------------------------------------------------------------

_SESSION_GLOBALS = (
    "_PWS",
    "_CONTEXT",
    "_PAGE",
    "_LAUNCHING",
    "_LAUNCH_LOOP",
    "_CLONE_LOCK",
    "_ACTION_COUNT",
    "_LAST_ACTION_TS",
    "_KEEP_OPEN",
    "_LIFECYCLE_TASK",
    "_FRAME_SELECTOR",
)


@pytest.fixture(autouse=True)
def _reset_session_globals() -> Iterator[None]:
    saved = {name: getattr(session, name) for name in _SESSION_GLOBALS}
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(session, name, value)


# ---------------------------------------------------------------------------
# close_session — idempotency
# ---------------------------------------------------------------------------


async def test_close_session_when_inactive_is_noop() -> None:
    assert session._PAGE is None
    assert session._CONTEXT is None

    # Should complete without error despite all globals being None
    await session.close_session()

    assert session.is_active() is False
    assert session._PAGE is None


async def test_close_session_twice_is_idempotent() -> None:
    await session.close_session()
    await session.close_session()
    assert session.is_active() is False


async def test_close_session_after_partial_teardown_is_safe() -> None:
    session._PAGE = cast("Any", object())
    session._CONTEXT = cast("Any", object())
    session._PWS = cast("Any", object())
    # Simulate a partial teardown — page gone but context linger
    session._PAGE = None

    await session.close_session()  # should not raise

    assert session._PAGE is None
    assert session._CONTEXT is None
    assert session._PWS is None


# ---------------------------------------------------------------------------
# close_session — global reset
# ---------------------------------------------------------------------------


class _FakeAsyncioTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


async def test_close_session_resets_all_globals() -> None:
    _log, context, page = make_session()
    task_stub = _FakeAsyncioTask()
    lock_stub = object()

    session._PAGE = cast("Any", page)
    session._CONTEXT = cast("Any", context)
    session._PWS = cast("Any", object())
    session._LAUNCH_LOOP = cast("Any", object())
    session._ACTION_COUNT = 42
    session._LAST_ACTION_TS = 12345.0
    session._KEEP_OPEN = True
    session._FRAME_SELECTOR = "iframe#main"
    session._CLONE_LOCK = cast("Any", lock_stub)
    session._LIFECYCLE_TASK = cast("Any", task_stub)

    assert session.is_active() is True

    await session.close_session()

    assert session._PAGE is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session._LAUNCH_LOOP is None
    assert session._ACTION_COUNT == 0
    assert session._LAST_ACTION_TS == 0.0
    assert session._KEEP_OPEN is None
    assert session._FRAME_SELECTOR is None
    assert session._CLONE_LOCK is None
    assert session._LIFECYCLE_TASK is None
    assert task_stub.cancelled is True
    assert session.is_active() is False


async def test_close_session_resets_even_when_context_close_fails() -> None:
    class _BrokenContext:
        async def close(self) -> None:
            msg = "something went wrong"
            raise RuntimeError(msg)

    session._PAGE = cast("Any", object())
    session._CONTEXT = cast("Any", _BrokenContext())
    session._PWS = cast("Any", object())

    await session.close_session()  # warning logged, but reset continues

    assert session._PAGE is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session.is_active() is False


# ---------------------------------------------------------------------------
# is_active state transitions
# ---------------------------------------------------------------------------


async def test_is_active_reflects_close_session_transition() -> None:
    _log, _context, page = make_session()
    session._PAGE = cast("Any", page)

    assert session.is_active() is True

    await session.close_session()

    assert session.is_active() is False


async def test_close_session_clears_active_state_even_without_real_context() -> None:
    session._PAGE = cast("Any", object())
    session._CONTEXT = cast("Any", object())

    assert session.is_active() is True

    await session.close_session()

    assert session.is_active() is False
