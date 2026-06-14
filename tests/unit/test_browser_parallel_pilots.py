"""Parallel page-pilot isolation: each pilot is strictly bound to its own tab.

Backward-compat for single-agent ("main") flow lives in test_browser_session.py;
this module proves the per-agent contextvar, ownership, and lifecycle behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import pytest

from supporter.config import config
from supporter.tools.browser import recorder, session
from supporter.tools.browser.playbook_store import build_step
from supporter.tools.delegate.agents import _cache_key

if TYPE_CHECKING:
    from collections.abc import Iterator

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


class _FakePage:
    def __init__(self, ctx: _FakeContext, url: str = "about:blank") -> None:
        self.ctx = ctx
        self.url = url
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True
        if self in self.ctx.pages:
            self.ctx.pages.remove(self)


class _FakeContext:
    def __init__(self) -> None:
        self.pages: list[_FakePage] = []
        self.closed = False

    async def new_page(self) -> _FakePage:
        page = _FakePage(self)
        self.pages.append(page)
        return page

    async def close(self) -> None:
        self.closed = True


class _FakePWS:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(config, "browser_parallel_pilots", True)
    saved_dicts = {name: dict(getattr(session, name)) for name in _DICT_GLOBALS}
    saved_active = dict(recorder._ACTIVE)
    saved_pws = session._PWS
    saved_context = session._CONTEXT
    saved_keep = session._KEEP_OPEN
    token = session._AGENT_ID.set("main")
    try:
        yield
    finally:
        for name, value in saved_dicts.items():
            current = getattr(session, name)
            current.clear()
            current.update(value)
        recorder._ACTIVE.clear()
        recorder._ACTIVE.update(saved_active)
        session._PWS = saved_pws
        session._CONTEXT = saved_context
        session._KEEP_OPEN = saved_keep
        session._AGENT_ID.reset(token)


async def _under_agent[T](aid: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run fn in its own task with the browser agent id bound to aid."""

    async def runner() -> T:
        agent_token = session.set_agent_id(aid)
        try:
            return await fn()
        finally:
            session.reset_contextvar(agent_token)

    return await asyncio.create_task(runner())


async def test_two_pilots_have_distinct_active_pages() -> None:
    ctx = _FakeContext()
    session._CONTEXT = ctx  # type: ignore[assignment]
    page_a = await ctx.new_page()
    page_b = await ctx.new_page()

    async def activate_a() -> Any:
        session.set_active(page_a)
        return session.active_page()

    async def activate_b() -> Any:
        session.set_active(page_b)
        return session.active_page()

    a_active = await _under_agent("a", activate_a)
    b_active = await _under_agent("b", activate_b)

    assert a_active is page_a
    assert b_active is page_b
    assert a_active is not b_active
    assert session._PAGES["a"] is page_a
    assert session._PAGES["b"] is page_b


async def test_recorder_step_buffers_do_not_collide() -> None:
    async def record_a() -> None:
        recorder.start("goal-a")
        recorder.record(build_step("click", result="ok-a"))

    async def record_b() -> None:
        recorder.start("goal-b")
        recorder.record(build_step("type", result="ok-b1"))
        recorder.record(build_step("press", result="ok-b2"))

    # Interleave the two agents' recording sessions.
    await _under_agent("a", record_a)
    await _under_agent("b", record_b)

    assert recorder._ACTIVE["a"].goal == "goal-a"
    assert recorder._ACTIVE["b"].goal == "goal-b"
    assert len(recorder._ACTIVE["a"].steps) == 1
    assert len(recorder._ACTIVE["b"].steps) == 2
    assert recorder._ACTIVE["a"].steps[0].result_head == "ok-a"


async def test_strict_own_tab_enforcement() -> None:
    ctx = _FakeContext()
    session._CONTEXT = ctx  # type: ignore[assignment]
    a1 = await ctx.new_page()
    a2 = await ctx.new_page()
    b1 = await ctx.new_page()

    async def setup_a() -> list[Any]:
        session.set_active(a1)
        session.set_active(a2)
        return session.list_pages()

    async def setup_b() -> list[Any]:
        session.set_active(b1)
        return session.list_pages()

    a_pages = await _under_agent("a", setup_a)
    b_pages = await _under_agent("b", setup_b)

    assert len(a_pages) == 2
    assert len(b_pages) == 1
    # A B-index can only ever resolve to a B-owned page.
    assert b_pages[0] is b1
    assert a1 not in b_pages
    assert a2 not in b_pages
    assert b1 not in a_pages


async def test_release_one_keeps_others_then_last_tears_down() -> None:
    ctx = _FakeContext()
    pws = _FakePWS()
    session._CONTEXT = ctx  # type: ignore[assignment]
    session._PWS = pws  # type: ignore[assignment]
    session._KEEP_OPEN = None
    a1 = await ctx.new_page()
    b1 = await ctx.new_page()

    async def activate_a() -> None:
        session.set_active(a1)

    async def activate_b() -> None:
        session.set_active(b1)

    await _under_agent("a", activate_a)
    await _under_agent("b", activate_b)

    await session.release_agent("a")

    assert a1.is_closed()
    assert "a" not in session._OWNED_PAGES
    assert session._CONTEXT is ctx  # type: ignore[comparison-overlap]
    assert session._PWS is pws  # type: ignore[comparison-overlap]
    assert not b1.is_closed()
    assert "b" in session._OWNED_PAGES

    await session.release_agent("b")

    assert b1.is_closed()
    assert session._CONTEXT is None
    assert session._PWS is None
    assert ctx.closed
    assert pws.stopped


async def test_blank_page_reuse_race_yields_distinct_pages() -> None:
    ctx = _FakeContext()
    session._CONTEXT = ctx  # type: ignore[assignment]
    await ctx.new_page()  # single pre-existing blank tab

    async def acquire(aid: str) -> Any:
        return await _under_agent(aid, lambda: session._acquire_agent_page(aid))

    page_a, page_b = await asyncio.gather(acquire("a"), acquire("b"))

    assert page_a is not page_b
    assert session._PAGES["a"] is page_a
    assert session._PAGES["b"] is page_b
    assert len(ctx.pages) == 2  # one adopted, one freshly created


async def test_cleanup_respects_ownership() -> None:
    ctx = _FakeContext()
    session._CONTEXT = ctx  # type: ignore[assignment]
    b_blank = await ctx.new_page()
    orphan_blank = await ctx.new_page()
    session._OWNED_PAGES["b"] = {b_blank}  # type: ignore[arg-type]

    await session.cleanup_blank_tabs()

    assert not b_blank.is_closed()
    assert orphan_blank.is_closed()


async def test_contextvar_propagates_into_handler_scope() -> None:
    assert session.current_agent_id() == "main"

    async def handler_like() -> str:
        # Stands in for a browser handler reading the scheduler-set id.
        return session.current_agent_id()

    seen = await _under_agent("pilot-7", handler_like)
    assert seen == "pilot-7"
    # Outer scope is unaffected once the task completes.
    assert session.current_agent_id() == "main"


def test_cache_key_is_none_for_page_pilot() -> None:
    pilot = {"agent": "page-pilot", "model": "m", "live": False}
    explorer = {"agent": "explorer", "model": "m", "live": False}

    assert _cache_key(pilot) is None
    assert _cache_key(explorer) == ("explorer", "m", False)


async def test_second_pilot_allocates_page_after_first_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second pilot spin-waiting on _LAUNCHING must get its own page after launch."""
    ctx = _FakeContext()
    pws = _FakePWS()
    session._CONTEXT = ctx  # type: ignore[assignment]
    session._PWS = pws  # type: ignore[assignment]
    monkeypatch.setattr(session, "_LAUNCHING", True)

    async def mock_sleep(delay: float) -> None:
        # Flip _LAUNCHING off on the first spin-wait call so the loop exits.
        if delay == 0.1:
            session._LAUNCHING = False

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    async def get_as_b() -> tuple[Any, Any, Any]:
        return await session.get_session()

    _, _, b_page = await _under_agent("b", get_as_b)

    assert b_page is not None
    assert session._PAGES.get("b") is b_page
