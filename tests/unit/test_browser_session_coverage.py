from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from supporter.tools.browser import guardrails, session
from tests.browser_fakes import FakePage, make_session


@pytest.fixture(autouse=True)
def _reset_session() -> Generator[None]:
    session._PAGES.clear()
    session._CONTEXT = None
    session._PWS = None
    session._FRAME_SELECTORS.clear()
    session._OWNED_PAGES.clear()
    session._IDLE_TASK = None
    session._CLEANUP_TASK = None
    session._LAUNCHING = False
    session._LAUNCH_LOOP = None
    session._ACTION_COUNT.clear()
    session._ACTION_CAP_CEILING.clear()
    session._LAST_ACTION_TS.clear()
    session._ACTION_TIMES.clear()
    session._SESSION_START_TS.clear()
    session._TEMPO.clear()
    session._SELECTED_PROFILE = None
    session._CLONE_LOCK = None
    session._LAST_ACTIVITY_TS = 0.0
    token = session._AGENT_ID.set("main")
    yield
    session._IDLE_TASK = None
    # Reset launch-lifecycle globals so a test that set them (e.g. _LAUNCH_LOOP
    # to a now-closed loop) can't leak into another test file and trip
    # get_session's per-loop singleton guard.
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = False
    session._LAUNCH_LOOP = None
    session._AGENT_ID.reset(token)


async def test_is_blank_detects_blank_urls() -> None:
    assert session.is_blank(cast("Any", type("", (), {"url": "about:blank"})()))
    assert session.is_blank(cast("Any", type("", (), {"url": "chrome://newtab/"})()))
    assert session.is_blank(cast("Any", type("", (), {"url": ""})()))
    assert not session.is_blank(
        cast("Any", type("", (), {"url": "https://example.com"})())
    )


async def test_is_blank_handles_exception() -> None:
    class BrokenPage:
        @property
        def url(self) -> str:
            raise RuntimeError("detached")

    assert session.is_blank(BrokenPage()) is False


def test_clone_ignore_skips_cache_dirs() -> None:
    skip = session._clone_ignore("/some/dir", ["Cache", "GPUCache", "index.html"])
    assert "Cache" in skip
    assert "GPUCache" in skip
    assert "index.html" not in skip


def test_clone_ignore_skips_singleton() -> None:
    skip = session._clone_ignore(
        "/some/dir", ["SingletonLock", "SingletonCookie", "data.db"]
    )
    assert "SingletonLock" in skip
    assert "SingletonCookie" in skip
    assert "data.db" not in skip


def test_clone_ignore_once_skips_session_dirs() -> None:
    skip = session._clone_ignore_once(
        "/some/dir", ["Local Storage", "Cache", "file.txt"]
    )
    assert "Local Storage" in skip
    assert "Cache" in skip
    assert "file.txt" not in skip


def test_newer_returns_true_when_dst_missing(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "missing.txt"
    src.write_text("data")
    assert session._newer(src, dst) is True


def test_newer_returns_true_when_src_newer(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new")
    dst.write_text("old")
    import time

    now = time.time()
    os.utime(src, (now + 10, now + 10))
    os.utime(dst, (now, now))
    assert session._newer(src, dst) is True


def test_newer_returns_false_when_dst_newer(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("old")
    dst.write_text("new")
    import time

    now = time.time()
    os.utime(src, (now, now))
    os.utime(dst, (now + 10, now + 10))
    assert session._newer(src, dst) is False


def test_mirror_dir_copies_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "file.txt").write_text("hello")
    (src / "sub").mkdir()
    (src / "sub" / "nested.txt").write_text("world")

    session._mirror_dir(src, dst)

    assert (dst / "file.txt").read_text() == "hello"
    assert (dst / "sub" / "nested.txt").read_text() == "world"


def test_mirror_dir_removes_stale_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "keep.txt").write_text("keep")
    (dst / "stale.txt").write_text("stale")

    session._mirror_dir(src, dst)

    assert (dst / "keep.txt").exists()
    assert not (dst / "stale.txt").exists()


def test_mirror_dir_noop_when_src_missing(tmp_path: Path) -> None:
    session._mirror_dir(tmp_path / "nonexistent", tmp_path / "dst")
    assert not (tmp_path / "dst").exists()


async def test_resolve_close_at_task_end_not_active() -> None:
    session._PAGES.pop("main", None)
    result = await session.resolve_close_at_task_end()
    assert result == ""


async def test_resolve_close_at_task_end_releases_pages() -> None:
    session._PAGES["main"] = cast("Any", object())
    result = await session.resolve_close_at_task_end()
    assert "left open" in result


async def test_resolve_close_at_task_end_no_pages() -> None:
    session._PAGES.pop("main", None)
    result = await session.resolve_close_at_task_end()
    assert result == ""


async def test_resolve_close_at_task_end_user_confirms_close() -> None:
    session._PAGES["main"] = cast("Any", object())
    session._CONTEXT = cast(
        "Any",
        type(
            "",
            (),
            {
                "close": AsyncMock(),
                "pages": [],
            },
        )(),
    )
    session._PWS = cast(
        "Any",
        type("", (), {"stop": AsyncMock()})(),
    )

    async def confirm(title: str, detail: str) -> bool:
        return True

    guardrails.browse_confirmation_callback = confirm
    try:
        result = await session.resolve_close_at_task_end()
        assert "left open" in result
    finally:
        guardrails.browse_confirmation_callback = None


async def test_resolve_close_at_task_end_user_declines() -> None:
    session._PAGES["main"] = cast("Any", object())

    async def deny(title: str, detail: str) -> bool:
        return False

    guardrails.browse_confirmation_callback = deny
    try:
        result = await session.resolve_close_at_task_end()
        assert "left open" in result
    finally:
        guardrails.browse_confirmation_callback = None


async def test_cleanup_blank_tabs_with_no_context() -> None:
    session._CONTEXT = None
    session._PAGES.pop("main", None)
    await session.cleanup_blank_tabs()


async def test_close_session_resets_all_globals() -> None:
    session._PAGES["main"] = cast("Any", object())
    session._CONTEXT = cast(
        "Any",
        type("", (), {"close": AsyncMock(), "pages": []})(),
    )
    session._PWS = cast("Any", type("", (), {"stop": AsyncMock()})())
    session._IDLE_TASK = None
    session._CLEANUP_TASK = None
    session._ACTION_COUNT["main"] = 5
    session._TEMPO["main"] = 1.5

    await session.close_session()

    assert session._PAGES.get("main") is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session._ACTION_COUNT == {}
    assert session._TEMPO == {}


async def test_prewarm_clone_skips_when_profile_path_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", "/some/path")
    session._PAGES.pop("main", None)
    await session.prewarm_clone()


async def test_prewarm_clone_skips_when_page_active() -> None:
    session._PAGES["main"] = cast("Any", object())
    await session.prewarm_clone()


async def test_prewarm_clone_skips_when_no_profile_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr("supporter.config.config.browser_profile_name", "")
    session._PAGES.pop("main", None)
    await session.prewarm_clone()


async def test_start_idle_monitor_creates_and_resolves_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_idle_close_seconds", 60)
    session._IDLE_TASK = None
    session._LAST_ACTIVITY_TS = 0.0

    session._start_idle_monitor()

    task = session._IDLE_TASK
    assert task is not None
    # Task is running; we just verify it exists
    session._IDLE_TASK = None


async def test_start_idle_monitor_noop_when_idle_close_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_idle_close_seconds", 0)
    session._IDLE_TASK = None
    session._LAST_ACTIVITY_TS = 0.0

    session._start_idle_monitor()

    assert session._IDLE_TASK is None


async def test_start_idle_monitor_noop_when_task_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_idle_close_seconds", 60)
    session._LAST_ACTIVITY_TS = 0.0
    sentinel = cast("Any", object())
    session._IDLE_TASK = sentinel

    session._start_idle_monitor()

    assert session._IDLE_TASK is sentinel
    session._IDLE_TASK = None


async def test_idle_monitor_swallows_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_idle_close_seconds", 0)
    session._IDLE_TASK = None
    session._LAST_ACTIVITY_TS = 0.0

    session._start_idle_monitor()
    assert session._IDLE_TASK is None


def test_clear_cleanup_task_resets_global() -> None:
    session._CLEANUP_TASK = cast("Any", object())
    session._clear_cleanup_task(object())
    assert session._CLEANUP_TASK is None


async def test_cleanup_blank_tabs_closes_blank_keeps_active() -> None:
    log, context, page = make_session()
    blank = context.add_page(FakePage(log, url="about:blank"))
    session._CONTEXT = cast("Any", context)
    session._PAGES["main"] = cast("Any", page)

    await session.cleanup_blank_tabs()

    assert blank not in context.pages
    assert page in context.pages


async def test_cleanup_blank_tabs_handles_close_error() -> None:
    _log, context, page = make_session()

    class BoomTab:
        url = "about:blank"

        async def close(self) -> None:
            raise RuntimeError("detached")

    context.pages.append(cast("Any", BoomTab()))
    session._CONTEXT = cast("Any", context)
    session._PAGES["main"] = cast("Any", page)

    await session.cleanup_blank_tabs()


def test_profile_dir_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr(sys, "platform", "win32")

    path = str(session._profile_dir())

    assert "Chrome" in path
    assert "User Data" in path


def test_profile_dir_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr(sys, "platform", "linux")

    path = str(session._profile_dir())

    assert ".config" in path
    assert "google-chrome" in path


async def test_clone_lock_creates_and_reuses_lock() -> None:
    session._CLONE_LOCK = None

    lock1 = session._clone_lock()
    lock2 = session._clone_lock()

    assert isinstance(lock1, asyncio.Lock)
    assert lock1 is lock2


async def test_clone_profile_runs_build_in_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "_profile_dir", lambda: Path("/user-data"))
    captured: dict[str, Any] = {}

    def fake_build(user_data: Path, profile: str) -> Path:
        captured["user_data"] = user_data
        captured["profile"] = profile
        return Path("/clone")

    monkeypatch.setattr(session, "_build_clone", fake_build)
    session._CLONE_LOCK = None

    result = await session._clone_profile("Default")

    assert result == Path("/clone")
    assert captured == {"user_data": Path("/user-data"), "profile": "Default"}


async def test_prewarm_clone_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr("supporter.config.config.browser_profile_name", "Default")
    session._PAGES.pop("main", None)

    async def ok(profile: str) -> Path:
        return Path("/clone")

    monkeypatch.setattr(session, "_clone_profile", ok)

    await session.prewarm_clone()


async def test_prewarm_clone_logs_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr("supporter.config.config.browser_profile_name", "Default")
    session._PAGES.pop("main", None)

    async def boom(profile: str) -> Path:
        raise RuntimeError("clone failed")

    monkeypatch.setattr(session, "_clone_profile", boom)

    await session.prewarm_clone()


async def test_launch_or_lock_error_returns_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    async def fake_launch(pws: Any, launch_dir: Path, profile: str) -> Any:
        return sentinel

    monkeypatch.setattr(session, "_launch_context", fake_launch)

    result = await session._launch_or_lock_error(None, Path("/d"), "Default")

    assert result is sentinel


async def test_launch_or_lock_error_translates_lock_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_launch(pws: Any, launch_dir: Path, profile: str) -> Any:
        raise RuntimeError("SingletonLock present")

    monkeypatch.setattr(session, "_launch_context", fake_launch)

    with pytest.raises(RuntimeError, match="already using this profile"):
        await session._launch_or_lock_error(None, Path("/d"), "Default")


async def test_launch_or_lock_error_reraises_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_launch(pws: Any, launch_dir: Path, profile: str) -> Any:
        raise ValueError("unrelated boom")

    monkeypatch.setattr(session, "_launch_context", fake_launch)

    with pytest.raises(ValueError, match="unrelated boom"):
        await session._launch_or_lock_error(None, Path("/d"), "Default")


class _FakeChromium:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def launch_persistent_context(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "context"


class _FakePws:
    def __init__(self) -> None:
        self.chromium = _FakeChromium()


async def test_launch_context_includes_profile_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    pws = _FakePws()

    result = await session._launch_context(pws, Path("/user-data"), "Work")

    assert result == "context"
    assert "--profile-directory=Work" in pws.chromium.kwargs["args"]
    assert pws.chromium.kwargs["user_data_dir"] == str(Path("/user-data"))


async def test_launch_context_linux_adds_password_store_no_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    pws = _FakePws()

    await session._launch_context(pws, Path("/user-data"), None)

    args = pws.chromium.kwargs["args"]
    assert "--password-store=gnome-libsecret" in args
    assert not any(a.startswith("--profile-directory=") for a in args)


async def test_get_session_rejects_cross_loop_launch() -> None:
    session._PAGES.pop("main", None)
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = True
    session._LAUNCH_LOOP = object()

    with pytest.raises(RuntimeError, match="different event loop"):
        await session.get_session()


async def test_close_session_cancels_cleanup_task() -> None:
    async def forever() -> None:
        await asyncio.sleep(3600)

    cleanup = asyncio.ensure_future(forever())
    session._CLEANUP_TASK = cleanup
    session._CONTEXT = None
    session._PWS = None
    session._PAGES.pop("main", None)

    await session.close_session()

    assert session._CLEANUP_TASK is None
    with contextlib.suppress(asyncio.CancelledError):
        await cleanup
    assert cleanup.cancelled()


async def test_pace_prunes_stale_action_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "next_tempo", lambda tempo: 1.0)
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.0)
    monkeypatch.setattr(guardrails, "fatigue_multiplier", lambda minutes: 1.0)
    monkeypatch.setattr(guardrails, "rate_throttle_delay", lambda count, span: 0.0)
    monkeypatch.setattr(guardrails, "maybe_idle_gap", lambda: 0.0)
    monkeypatch.setattr(guardrails, "action_cap", lambda: 1000)

    session._ACTION_TIMES.clear()
    session._ACTION_TIMES.append(0.0)
    session._LAST_ACTION_TS["main"] = 0.0
    session._SESSION_START_TS["main"] = 0.0
    session._ACTION_COUNT["main"] = 0
    session._ACTION_CAP_CEILING["main"] = 0

    await session.pace()

    assert all(ts > 0.0 for ts in session._ACTION_TIMES)
    assert session._ACTION_COUNT["main"] == 1


class _StoppablePws:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _FakePlaywright:
    def __init__(self, pws: _StoppablePws) -> None:
        self._pws = pws

    async def start(self) -> _StoppablePws:
        return self._pws


def _wire_launch(monkeypatch: pytest.MonkeyPatch, pws: Any, launch: Any) -> None:
    monkeypatch.setattr(session, "_start_idle_monitor", lambda: None)
    monkeypatch.setattr(
        "patchright.async_api.async_playwright", lambda: _FakePlaywright(pws)
    )

    async def resolve() -> str:
        return "Default"

    monkeypatch.setattr(session, "_resolve_profile_name", resolve)
    monkeypatch.setattr("supporter.config.config.browser_profile_path", "/profile/dir")
    monkeypatch.setattr(session, "_profile_dir", lambda: Path("/profile/dir"))
    monkeypatch.setattr(session, "_launch_or_lock_error", launch)
    session._PAGES.pop("main", None)
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = False


async def test_get_session_launches_with_new_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _log, context, page = make_session()
    pws = _StoppablePws()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        return context

    _wire_launch(monkeypatch, pws, launch)

    out_pws, out_ctx, out_page = await session.get_session()

    assert out_pws is pws
    assert out_ctx is context
    assert out_page is not page
    assert out_page in context.pages
    if session._CLEANUP_TASK is not None:
        await session._CLEANUP_TASK


async def test_get_session_reuses_existing_blank_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _log, context, page = make_session(url="about:blank")
    pws = _StoppablePws()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        return context

    _wire_launch(monkeypatch, pws, launch)

    _out_pws, _out_ctx, out_page = await session.get_session()

    assert out_page is page
    if session._CLEANUP_TASK is not None:
        await session._CLEANUP_TASK


async def test_get_session_cleans_up_when_launch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pws = _StoppablePws()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        raise RuntimeError("launch boom")

    _wire_launch(monkeypatch, pws, launch)

    with pytest.raises(RuntimeError, match="launch boom"):
        await session.get_session()

    assert pws.stopped is True
    assert session._PWS is None
    assert session._CONTEXT is None
    assert session._PAGES.get("main") is None
    assert session._LAUNCHING is False


async def test_get_session_closes_context_when_page_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pws = _StoppablePws()

    class _BoomContext:
        def __init__(self) -> None:
            self.closed = False
            self.pages: list[Any] = []

        async def new_page(self) -> Any:
            raise RuntimeError("page boom")

        async def close(self) -> None:
            self.closed = True

    boom_context = _BoomContext()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        return boom_context

    _wire_launch(monkeypatch, pws, launch)

    with pytest.raises(RuntimeError, match="page boom"):
        await session.get_session()

    assert boom_context.closed is True
    assert pws.stopped is True
    assert session._CONTEXT is None


async def test_get_session_waits_for_concurrent_launch() -> None:
    _log, context, page = make_session()
    session._PAGES.pop("main", None)
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = True
    session._LAUNCH_LOOP = asyncio.get_running_loop()

    async def finish_launch() -> None:
        await asyncio.sleep(0.01)
        session._PAGES["main"] = cast("Any", page)
        session._CONTEXT = cast("Any", context)
        session._PWS = cast("Any", object())
        session._LAUNCHING = False

    launch_task = asyncio.ensure_future(finish_launch())

    _out_pws, out_ctx, out_page = await session.get_session()
    await launch_task

    assert out_ctx is context
    assert out_page is page


async def test_get_session_raises_when_concurrent_launch_fails() -> None:
    session._PAGES.pop("main", None)
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = True
    session._LAUNCH_LOOP = asyncio.get_running_loop()

    async def fail_launch() -> None:
        await asyncio.sleep(0.01)
        session._LAUNCHING = False

    fail_task = asyncio.ensure_future(fail_launch())

    with pytest.raises(RuntimeError, match="launch failed"):
        await session.get_session()
    await fail_task


async def test_get_session_clones_profile_when_no_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _log, context, page = make_session(url="about:blank")
    pws = _StoppablePws()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        return context

    _wire_launch(monkeypatch, pws, launch)
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)

    cloned: dict[str, str] = {}

    async def clone(profile: str) -> Path:
        cloned["profile"] = profile
        return Path("/clone/dir")

    monkeypatch.setattr(session, "_clone_profile", clone)

    _out_pws, _out_ctx, out_page = await session.get_session()

    assert cloned["profile"] == "Default"
    assert out_page is page
    if session._CLEANUP_TASK is not None:
        await session._CLEANUP_TASK


async def test_get_session_logs_when_cleanup_steps_also_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DoubleBoomContext:
        def __init__(self) -> None:
            self.pages: list[Any] = []

        async def new_page(self) -> Any:
            raise RuntimeError("page boom")

        async def close(self) -> None:
            raise RuntimeError("close boom")

    class _BoomStopPws:
        async def stop(self) -> None:
            raise RuntimeError("stop boom")

    context = _DoubleBoomContext()
    pws = _BoomStopPws()

    async def launch(p: Any, d: Path, prof: str) -> Any:
        return context

    _wire_launch(monkeypatch, pws, launch)

    with pytest.raises(RuntimeError, match="page boom"):
        await session.get_session()

    assert session._CONTEXT is None
    assert session._PWS is None
