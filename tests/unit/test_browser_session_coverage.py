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
    session._KEEP_OPEN = None
    session._PAGE = None
    session._CONTEXT = None
    session._PWS = None
    session._FRAME_SELECTOR = None
    session._LIFECYCLE_TASK = None
    session._CLEANUP_TASK = None
    session._LAUNCHING = False
    session._LAUNCH_LOOP = None
    session._ACTION_COUNT = 0
    session._ACTION_CAP_CEILING = 0
    session._LAST_ACTION_TS = 0.0
    session._ACTION_TIMES.clear()
    session._SESSION_START_TS = 0.0
    session._TEMPO = 1.0
    session._SELECTED_PROFILE = None
    session._CLONE_LOCK = None
    yield
    session._KEEP_OPEN = None


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
    session._PAGE = None
    result = await session.resolve_close_at_task_end()
    assert result == ""


async def test_resolve_close_at_task_end_pinned_open() -> None:
    session._PAGE = cast("Any", object())
    session._KEEP_OPEN = True
    result = await session.resolve_close_at_task_end()
    assert "persistent session" in result


async def test_resolve_close_at_task_end_no_callback() -> None:
    session._PAGE = cast("Any", object())
    session._KEEP_OPEN = None
    guardrails.browse_confirmation_callback = None
    result = await session.resolve_close_at_task_end()
    assert "persistent session" in result


async def test_resolve_close_at_task_end_user_confirms_close() -> None:
    session._PAGE = cast("Any", object())
    session._KEEP_OPEN = False
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
        assert "Browser closed" in result
    finally:
        guardrails.browse_confirmation_callback = None


async def test_resolve_close_at_task_end_user_declines() -> None:
    session._PAGE = cast("Any", object())
    session._KEEP_OPEN = False

    async def deny(title: str, detail: str) -> bool:
        return False

    guardrails.browse_confirmation_callback = deny
    try:
        result = await session.resolve_close_at_task_end()
        assert "left open" in result
    finally:
        guardrails.browse_confirmation_callback = None


async def test_await_lifecycle_answer_when_already_set() -> None:
    session._KEEP_OPEN = True
    await session._await_lifecycle_answer()
    assert session._KEEP_OPEN is True


async def test_await_lifecycle_answer_waits_for_task() -> None:
    session._KEEP_OPEN = None
    done = asyncio.Event()

    async def fake_prompt() -> None:
        session._KEEP_OPEN = True
        done.set()

    session._LIFECYCLE_TASK = asyncio.ensure_future(fake_prompt())
    await session._await_lifecycle_answer()
    assert session._KEEP_OPEN is True
    session._LIFECYCLE_TASK = None


async def test_cleanup_blank_tabs_with_no_context() -> None:
    session._CONTEXT = None
    session._PAGE = None
    await session.cleanup_blank_tabs()


async def test_close_session_resets_all_globals() -> None:
    session._PAGE = cast("Any", object())
    session._CONTEXT = cast(
        "Any",
        type("", (), {"close": AsyncMock(), "pages": []})(),
    )
    session._PWS = cast("Any", type("", (), {"stop": AsyncMock()})())
    session._LIFECYCLE_TASK = None
    session._CLEANUP_TASK = None
    session._KEEP_OPEN = True
    session._ACTION_COUNT = 5
    session._TEMPO = 1.5

    await session.close_session()

    assert session._PAGE is None
    assert session._CONTEXT is None
    assert session._PWS is None
    assert session._KEEP_OPEN is None
    assert session._ACTION_COUNT == 0
    assert session._TEMPO == 1.0


async def test_prewarm_clone_skips_when_profile_path_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", "/some/path")
    session._PAGE = None
    await session.prewarm_clone()


async def test_prewarm_clone_skips_when_page_active() -> None:
    session._PAGE = cast("Any", object())
    await session.prewarm_clone()


async def test_prewarm_clone_skips_when_no_profile_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr("supporter.config.config.browser_profile_name", "")
    session._PAGE = None
    await session.prewarm_clone()


async def test_start_lifecycle_prompt_creates_and_resolves_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "browse_confirmation_callback", None)
    session._KEEP_OPEN = None
    session._LIFECYCLE_TASK = None

    session._start_lifecycle_prompt()

    task = session._LIFECYCLE_TASK
    assert task is not None
    await task
    assert session._KEEP_OPEN is True


async def test_start_lifecycle_prompt_noop_when_keep_open_set() -> None:
    session._KEEP_OPEN = True
    session._LIFECYCLE_TASK = None

    session._start_lifecycle_prompt()

    assert session._LIFECYCLE_TASK is None


async def test_start_lifecycle_prompt_noop_when_task_already_running() -> None:
    session._KEEP_OPEN = None
    sentinel = cast("Any", object())
    session._LIFECYCLE_TASK = sentinel

    session._start_lifecycle_prompt()

    assert session._LIFECYCLE_TASK is sentinel
    session._LIFECYCLE_TASK = None


async def test_start_lifecycle_prompt_swallows_prompt_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom() -> None:
        raise RuntimeError("prompt failed")

    monkeypatch.setattr(session, "_prompt_lifecycle", boom)
    session._KEEP_OPEN = None
    session._LIFECYCLE_TASK = None

    session._start_lifecycle_prompt()
    task = session._LIFECYCLE_TASK
    assert task is not None
    await task


def test_clear_cleanup_task_resets_global() -> None:
    session._CLEANUP_TASK = cast("Any", object())
    session._clear_cleanup_task(object())
    assert session._CLEANUP_TASK is None


async def test_cleanup_blank_tabs_closes_blank_keeps_active() -> None:
    log, context, page = make_session()
    blank = context.add_page(FakePage(log, url="about:blank"))
    session._CONTEXT = cast("Any", context)
    session._PAGE = cast("Any", page)

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
    session._PAGE = cast("Any", page)

    await session.cleanup_blank_tabs()


async def test_await_lifecycle_answer_handles_task_error() -> None:
    session._KEEP_OPEN = None

    async def boom() -> None:
        raise RuntimeError("prompt failed")

    session._LIFECYCLE_TASK = asyncio.ensure_future(boom())

    await session._await_lifecycle_answer()

    assert session._KEEP_OPEN is True
    session._LIFECYCLE_TASK = None


async def test_resolve_close_at_task_end_no_callback_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session._PAGE = cast("Any", object())
    session._KEEP_OPEN = False
    monkeypatch.setattr(guardrails, "browse_confirmation_callback", None)

    result = await session.resolve_close_at_task_end()

    assert result == ""


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
    session._PAGE = None

    async def ok(profile: str) -> Path:
        return Path("/clone")

    monkeypatch.setattr(session, "_clone_profile", ok)

    await session.prewarm_clone()


async def test_prewarm_clone_logs_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("supporter.config.config.browser_profile_path", None)
    monkeypatch.setattr("supporter.config.config.browser_profile_name", "Default")
    session._PAGE = None

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
    session._PAGE = None
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
    session._PAGE = None

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
    session._LAST_ACTION_TS = 0.0
    session._SESSION_START_TS = 0.0
    session._ACTION_COUNT = 0
    session._ACTION_CAP_CEILING = 0

    await session.pace()

    assert all(ts > 0.0 for ts in session._ACTION_TIMES)
    assert session._ACTION_COUNT == 1


async def test_start_lifecycle_prompt_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def hang() -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "_prompt_lifecycle", hang)
    session._KEEP_OPEN = None
    session._LIFECYCLE_TASK = None

    session._start_lifecycle_prompt()
    task = session._LIFECYCLE_TASK
    assert task is not None
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_await_lifecycle_answer_propagates_cancellation() -> None:
    session._KEEP_OPEN = None

    async def hang() -> None:
        await asyncio.Event().wait()

    task = asyncio.ensure_future(hang())
    await asyncio.sleep(0)
    task.cancel()
    session._LIFECYCLE_TASK = task

    with pytest.raises(asyncio.CancelledError):
        await session._await_lifecycle_answer()

    session._LIFECYCLE_TASK = None


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
    monkeypatch.setattr(session, "_start_lifecycle_prompt", lambda: None)
    monkeypatch.setattr(
        "patchright.async_api.async_playwright", lambda: _FakePlaywright(pws)
    )

    async def resolve() -> str:
        return "Default"

    monkeypatch.setattr(session, "_resolve_profile_name", resolve)
    monkeypatch.setattr("supporter.config.config.browser_profile_path", "/profile/dir")
    monkeypatch.setattr(session, "_profile_dir", lambda: Path("/profile/dir"))
    monkeypatch.setattr(session, "_launch_or_lock_error", launch)
    session._PAGE = None
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
    assert session._PAGE is None
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
    session._PAGE = None
    session._CONTEXT = None
    session._PWS = None
    session._LAUNCHING = True
    session._LAUNCH_LOOP = asyncio.get_running_loop()

    async def finish_launch() -> None:
        await asyncio.sleep(0.01)
        session._PAGE = cast("Any", page)
        session._CONTEXT = cast("Any", context)
        session._PWS = cast("Any", object())
        session._LAUNCHING = False

    launch_task = asyncio.ensure_future(finish_launch())

    _out_pws, out_ctx, out_page = await session.get_session()
    await launch_task

    assert out_ctx is context
    assert out_page is page


async def test_get_session_raises_when_concurrent_launch_fails() -> None:
    session._PAGE = None
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
