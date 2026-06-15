from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from supporter.tools.browser import session
from tests.browser_fakes import make_session

if TYPE_CHECKING:
    from collections.abc import Iterator


_SCALAR_GLOBALS = (
    "_PWS",
    "_CONTEXT",
    "_LAUNCHING",
    "_LAUNCH_LOOP",
    "_CLONE_LOCK",
    "_LAST_ACTIVITY_TS",
    "_IDLE_TASK",
)
_DICT_GLOBALS = (
    "_PAGES",
    "_FRAME_SELECTORS",
    "_OWNED_PAGES",
    "_ACTION_COUNT",
    "_ACTION_CAP_CEILING",
    "_LAST_ACTION_TS",
    "_SESSION_START_TS",
    "_TEMPO",
)


@pytest.fixture(autouse=True)
def _reset_session_globals() -> Iterator[None]:
    saved = {name: getattr(session, name) for name in _SCALAR_GLOBALS}
    saved_dicts = {name: dict(getattr(session, name)) for name in _DICT_GLOBALS}
    token = session._AGENT_ID.set("main")
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(session, name, value)
        for name, value in saved_dicts.items():
            current = getattr(session, name)
            current.clear()
            current.update(value)
        session._AGENT_ID.reset(token)


async def test_close_session_when_inactive_is_noop() -> None:
    assert session._PAGES.get("main") is None
    assert session._CONTEXT is None

    await session.close_session()

    assert session.is_active() is False
    assert session._PAGES.get("main") is None


async def test_close_session_twice_is_idempotent() -> None:
    await session.close_session()
    await session.close_session()
    assert session.is_active() is False


async def test_close_session_after_partial_teardown_is_safe() -> None:
    session._PAGES["main"] = cast("Any", object())
    session._CONTEXT = cast("Any", object())
    session._PWS = cast("Any", object())
    session._PAGES.pop("main", None)

    await session.close_session()

    assert session._PAGES.get("main") is None
    assert session._CONTEXT is None
    assert session._PWS is None


class _FakeAsyncioTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


async def test_close_session_resets_all_globals() -> None:
    _log, context, page = make_session()
    lock_stub = object()

    session._PAGES["main"] = cast("Any", page)
    session._CONTEXT = cast("Any", context)
    session._PWS = cast("Any", object())
    session._LAUNCH_LOOP = cast("Any", object())
    session._ACTION_COUNT["main"] = 42
    session._LAST_ACTION_TS["main"] = 12345.0
    session._FRAME_SELECTORS["main"] = "iframe#main"
    session._CLONE_LOCK = cast("Any", lock_stub)

    assert session.is_active() is True

    await session.close_session()

    assert session._PAGES.get("main") is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session._LAUNCH_LOOP is None
    assert session._ACTION_COUNT == {}
    assert session._LAST_ACTION_TS == {}
    assert session._FRAME_SELECTORS == {}
    assert session._CLONE_LOCK is None
    assert session._IDLE_TASK is None
    assert session._LAST_ACTIVITY_TS == 0.0
    assert session.is_active() is False


async def test_close_session_resets_even_when_context_close_fails() -> None:

    class _BrokenContext:
        async def close(self) -> None:
            msg = "something went wrong"
            raise RuntimeError(msg)

    session._PAGES["main"] = cast("Any", object())
    session._CONTEXT = cast("Any", _BrokenContext())
    session._PWS = cast("Any", object())

    await session.close_session()

    assert session._PAGES.get("main") is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session.is_active() is False


async def test_is_active_reflects_close_session_transition() -> None:
    _log, _context, page = make_session()
    session._PAGES["main"] = cast("Any", page)

    assert session.is_active() is True

    await session.close_session()

    assert session.is_active() is False


async def test_close_session_clears_active_state_even_without_real_context() -> None:
    session._PAGES["main"] = cast("Any", object())
    session._CONTEXT = cast("Any", object())

    assert session.is_active() is True

    await session.close_session()

    assert session.is_active() is False
