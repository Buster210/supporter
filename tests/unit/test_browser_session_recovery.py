"""Tests for browser session self-healing: stale-profile-lock recovery and the
launch-retry path that lets two tasks reuse one window instead of being forced
into a separate one."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from supporter.tools.browser import session
from supporter.types import AppConfig

_POSIX_ONLY = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="POSIX process control required"
)


def _make_singletons(d: Path, pid: int) -> None:
    """Lay down the three Chromium singleton files with a lock pointing at pid."""
    os.symlink(f"my-host-name-{pid}", d / "SingletonLock")
    (d / "SingletonSocket").symlink_to("socket-endpoint")
    (d / "SingletonCookie").symlink_to("12345")


# ---------------------------------------------------------------------------
# _pid_from_singleton_lock
# ---------------------------------------------------------------------------


def test_pid_from_lock_parses_trailing_pid(tmp_path: Path) -> None:
    lock = tmp_path / "SingletonLock"
    os.symlink("host-with-dashes-4242", lock)
    assert session._pid_from_singleton_lock(lock) == 4242


def test_pid_from_lock_returns_none_for_regular_file(tmp_path: Path) -> None:
    lock = tmp_path / "SingletonLock"
    lock.write_text("not a symlink")
    assert session._pid_from_singleton_lock(lock) is None


def test_pid_from_lock_returns_none_for_missing(tmp_path: Path) -> None:
    assert session._pid_from_singleton_lock(tmp_path / "nope") is None


def test_pid_from_lock_returns_none_for_non_numeric(tmp_path: Path) -> None:
    lock = tmp_path / "SingletonLock"
    os.symlink("host-notapid", lock)
    assert session._pid_from_singleton_lock(lock) is None


def test_pid_from_lock_rejects_pid_below_two(tmp_path: Path) -> None:
    lock = tmp_path / "SingletonLock"
    os.symlink("host-1", lock)
    assert session._pid_from_singleton_lock(lock) is None


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------


def test_pid_alive_true_for_current_process() -> None:
    assert session._pid_alive(os.getpid()) is True


@_POSIX_ONLY
def test_pid_alive_false_for_reaped_child() -> None:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)  # reap so the PID is fully gone
    assert session._pid_alive(pid) is False


# ---------------------------------------------------------------------------
# _process_cmdline / _process_holds_profile_dir
# ---------------------------------------------------------------------------


def test_process_cmdline_returns_str_for_self() -> None:
    line = session._process_cmdline(os.getpid())
    assert line is not None
    assert isinstance(line, str)
    assert line


def test_process_cmdline_none_for_dead_pid() -> None:
    if not hasattr(os, "fork"):
        pytest.skip("POSIX process control required")
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    assert session._process_cmdline(pid) is None


def test_holds_profile_dir_true_when_cmdline_has_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        session,
        "_process_cmdline",
        lambda pid: f"/x/chrome --user-data-dir={tmp_path} --headless",
    )
    assert session._process_holds_profile_dir(1234, tmp_path) is True


def test_holds_profile_dir_false_when_cmdline_lacks_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        session, "_process_cmdline", lambda pid: "/x/chrome --user-data-dir=/other"
    )
    assert session._process_holds_profile_dir(1234, tmp_path) is False


def test_holds_profile_dir_false_when_cmdline_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(session, "_process_cmdline", lambda pid: None)
    assert session._process_holds_profile_dir(1234, tmp_path) is False


def test_holds_profile_dir_false_on_sibling_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A sibling dir that merely shares our prefix (<dir>-2) must NOT match.
    monkeypatch.setattr(
        session,
        "_process_cmdline",
        lambda pid: f"/x/chrome --user-data-dir={tmp_path}-2",
    )
    assert session._process_holds_profile_dir(1234, tmp_path) is False


def test_holds_profile_dir_true_on_child_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The dir followed by a separator (a child path) is still our dir.
    monkeypatch.setattr(
        session, "_process_cmdline", lambda pid: f"chrome --x={tmp_path}/Default/Foo"
    )
    assert session._process_holds_profile_dir(1234, tmp_path) is True


# ---------------------------------------------------------------------------
# _force_clear_profile_lock
# ---------------------------------------------------------------------------


def test_clear_lock_noop_when_nothing_to_clear(tmp_path: Path) -> None:
    assert session._force_clear_profile_lock(tmp_path, can_kill=True) is True


def test_clear_lock_removes_stale_files(tmp_path: Path) -> None:
    # A dead PID: forked child reaped, so the lock is stale.
    if not hasattr(os, "fork"):
        pytest.skip("POSIX process control required")
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    _make_singletons(tmp_path, pid)

    assert session._force_clear_profile_lock(tmp_path, can_kill=False) is True
    for name in session._SINGLETON_FILES:
        assert not (tmp_path / name).exists()
        assert not (tmp_path / name).is_symlink()


@_POSIX_ONLY
def test_clear_lock_kills_live_holder_bound_to_dir(tmp_path: Path) -> None:
    # A live holder whose command line references the profile dir IS our
    # orphaned Chrome -> kill it (real ps identity check) and clear the lock.
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(60)", str(tmp_path)]
    )
    try:
        _make_singletons(tmp_path, proc.pid)
        assert session._force_clear_profile_lock(tmp_path, can_kill=True) is True
        proc.wait(timeout=5)  # raises TimeoutExpired if it was not killed
        assert session._pid_alive(proc.pid) is False
        for name in session._SINGLETON_FILES:
            assert not (tmp_path / name).is_symlink()
    finally:
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.wait(timeout=5)


@_POSIX_ONLY
def test_clear_lock_refuses_live_holder_not_bound_to_dir(tmp_path: Path) -> None:
    # can_kill=True, but the live holder's command line does NOT reference the
    # dir: it is a reused PID (e.g. the user's real Chrome). Refuse to kill.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _make_singletons(tmp_path, proc.pid)
        assert session._force_clear_profile_lock(tmp_path, can_kill=True) is False
        assert session._pid_alive(proc.pid) is True
        assert (tmp_path / "SingletonLock").is_symlink()
    finally:
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.wait(timeout=5)


@_POSIX_ONLY
def test_clear_lock_refuses_live_holder_when_not_allowed(tmp_path: Path) -> None:
    pid = os.fork()
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    try:
        _make_singletons(tmp_path, pid)
        # can_kill=False: a live holder may be the user's real Chrome -> refuse.
        assert session._force_clear_profile_lock(tmp_path, can_kill=False) is False
        assert session._pid_alive(pid) is True
        assert (tmp_path / "SingletonLock").is_symlink()
    finally:
        with contextlib.suppress(OSError, ChildProcessError):
            os.kill(pid, 9)
            os.waitpid(pid, 0)


# ---------------------------------------------------------------------------
# _launch_with_recovery
# ---------------------------------------------------------------------------


async def test_launch_recovers_from_stale_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(session, "_LOCK_RETRY_SETTLE_S", 0.0)
    sentinel = object()
    calls = {"launch": 0, "clear": 0}

    async def fake_launch(pws: object, d: Path, profile: str) -> object:
        calls["launch"] += 1
        if calls["launch"] == 1:
            raise RuntimeError("Failed to launch: SingletonLock file exists")
        return sentinel

    def fake_clear(d: Path, *, can_kill: bool) -> bool:
        calls["clear"] += 1
        assert can_kill is True
        return True

    monkeypatch.setattr(session, "_launch_context", fake_launch)
    monkeypatch.setattr(session, "_force_clear_profile_lock", fake_clear)

    result = await session._launch_with_recovery(
        object(), tmp_path, "Default", can_kill=True
    )
    assert result is sentinel
    assert calls == {"launch": 2, "clear": 1}


async def test_launch_does_not_recover_non_lock_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"clear": 0}

    async def fake_launch(pws: object, d: Path, profile: str) -> object:
        raise RuntimeError("chrome binary not found")

    def fake_clear(d: Path, *, can_kill: bool) -> bool:
        calls["clear"] += 1
        return True

    monkeypatch.setattr(session, "_launch_context", fake_launch)
    monkeypatch.setattr(session, "_force_clear_profile_lock", fake_clear)

    with pytest.raises(RuntimeError, match="chrome binary not found"):
        await session._launch_with_recovery(
            object(), tmp_path, "Default", can_kill=True
        )
    assert calls["clear"] == 0


async def test_launch_raises_help_when_lock_unclearable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_launch(pws: object, d: Path, profile: str) -> object:
        raise RuntimeError("user data directory is already in use")

    monkeypatch.setattr(session, "_launch_context", fake_launch)
    monkeypatch.setattr(
        session, "_force_clear_profile_lock", lambda d, *, can_kill: False
    )

    with pytest.raises(RuntimeError, match="separate profile directory"):
        await session._launch_with_recovery(
            object(), tmp_path, "Default", can_kill=False
        )


async def test_launch_raises_help_when_retry_still_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(session, "_LOCK_RETRY_SETTLE_S", 0.0)

    async def fake_launch(pws: object, d: Path, profile: str) -> object:
        raise RuntimeError("ProcessSingleton: profile appears to be in use")

    monkeypatch.setattr(session, "_launch_context", fake_launch)
    monkeypatch.setattr(
        session, "_force_clear_profile_lock", lambda d, *, can_kill: True
    )

    with pytest.raises(RuntimeError, match="separate profile directory"):
        await session._launch_with_recovery(
            object(), tmp_path, "Default", can_kill=True
        )


async def test_launch_retry_reraises_non_lock_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Lock cleared, but the retry fails for an unrelated reason (e.g. disk
    # full): surface THAT error, not the misleading lock-help message.
    monkeypatch.setattr(session, "_LOCK_RETRY_SETTLE_S", 0.0)
    calls = {"launch": 0}

    async def fake_launch(pws: object, d: Path, profile: str) -> object:
        calls["launch"] += 1
        if calls["launch"] == 1:
            raise RuntimeError("SingletonLock file exists")
        raise ValueError("disk full")

    monkeypatch.setattr(session, "_launch_context", fake_launch)
    monkeypatch.setattr(
        session, "_force_clear_profile_lock", lambda d, *, can_kill: True
    )

    with pytest.raises(ValueError, match="disk full"):
        await session._launch_with_recovery(
            object(), tmp_path, "Default", can_kill=True
        )
    assert calls["launch"] == 2


# ---------------------------------------------------------------------------
# Idle default
# ---------------------------------------------------------------------------


def test_idle_close_default_is_ten_minutes() -> None:
    fields = {f.name: f for f in dataclasses.fields(AppConfig)}
    assert fields["browser_idle_close_seconds"].default == 600
