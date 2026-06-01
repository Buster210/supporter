from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from supporter.tools.browser import guardrails, session

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    "_SELECTED_PROFILE",
)


@pytest.fixture(autouse=True)
def _reset_session_globals() -> Iterator[None]:
    saved = {name: getattr(session, name) for name in _SESSION_GLOBALS}
    saved_cb = guardrails.browse_confirmation_callback
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(session, name, value)
        guardrails.browse_confirmation_callback = saved_cb


def test_newer_true_when_dst_missing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.write_text("x")
    assert session._newer(src, tmp_path / "absent") is True


def test_newer_compares_mtime(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    dst.write_text("old")
    src.write_text("new")
    import os

    os.utime(dst, (1, 1))
    os.utime(src, (2, 2))
    assert session._newer(src, dst) is True
    os.utime(src, (0, 0))
    assert session._newer(src, dst) is False


def test_mirror_dir_copies_creates_and_prunes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "keep.txt").write_text("keep")
    (src / "sub" / "nested.txt").write_text("nested")
    (dst / "stale_dir").mkdir(parents=True)
    (dst / "stale_dir" / "old.txt").write_text("old")
    (dst / "stale.txt").write_text("stale")

    session._mirror_dir(src, dst)

    assert (dst / "keep.txt").read_text() == "keep"
    assert (dst / "sub" / "nested.txt").read_text() == "nested"
    assert not (dst / "stale.txt").exists()
    assert not (dst / "stale_dir").exists()


def test_mirror_dir_skips_cache_subtrees(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "Cache").mkdir(parents=True)
    (src / "Cache" / "blob").write_text("junk")
    (src / "real.txt").write_text("real")

    session._mirror_dir(src, dst)

    assert (dst / "real.txt").exists()
    assert not (dst / "Cache").exists()


def test_mirror_dir_noop_when_src_absent(tmp_path: Path) -> None:
    session._mirror_dir(tmp_path / "absent", tmp_path / "dst")
    assert not (tmp_path / "dst").exists()


def _make_db(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE t (v TEXT)")
        con.execute("INSERT INTO t VALUES (?)", (value,))
        con.commit()
    finally:
        con.close()


def _read_db(path: Path) -> str:
    con = sqlite3.connect(str(path))
    try:
        row = con.execute("SELECT v FROM t").fetchone()
        return str(row[0])
    finally:
        con.close()


def test_build_clone_refreshes_sqlite_dirs_and_root_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "User Data"
    profile = "Profile 2"
    src_profile = source / profile
    src_profile.mkdir(parents=True)

    _make_db(src_profile / "Network" / "Cookies", "cookie-v1")
    _make_db(src_profile / "Login Data", "login-v1")
    (src_profile / "Local Storage").mkdir()
    (src_profile / "Local Storage" / "store.ldb").write_text("ls-v1")
    (source / "Local State").write_text("key-v1")

    clone_root = tmp_path / "clone"
    monkeypatch.setattr(session, "_CLONE_ROOT", clone_root)

    out = session._build_clone(source, profile)
    assert out == clone_root

    dst_profile = clone_root / profile
    assert _read_db(dst_profile / "Network" / "Cookies") == "cookie-v1"
    assert _read_db(dst_profile / "Login Data") == "login-v1"
    assert (dst_profile / "Local Storage" / "store.ldb").read_text() == "ls-v1"
    assert (clone_root / "Local State").read_text() == "key-v1"


def test_build_clone_missing_profile_returns_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "User Data"
    source.mkdir()
    monkeypatch.setattr(session, "_CLONE_ROOT", tmp_path / "clone")
    assert session._build_clone(source, "Nope") == source


async def test_prewarm_noop_when_profile_path_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session.config, "browser_profile_path", "/some/dir")  # type: ignore[attr-defined]
    monkeypatch.setattr(session.config, "browser_profile_name", "Profile2")  # type: ignore[attr-defined]
    called = False

    async def fail() -> Path:
        nonlocal called
        called = True
        return Path("/")

    monkeypatch.setattr(session, "_clone_profile", fail)
    session._PAGE = None
    await session.prewarm_clone()
    assert called is False


async def test_prewarm_noop_when_session_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session.config, "browser_profile_path", None)  # type: ignore[attr-defined]
    called = False

    async def fail() -> Path:
        nonlocal called
        called = True
        return Path("/")

    monkeypatch.setattr(session, "_clone_profile", fail)
    session._PAGE = cast("Any", object())
    await session.prewarm_clone()
    assert called is False


async def test_prewarm_builds_clone_when_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session.config, "browser_profile_path", None)  # type: ignore[attr-defined]
    monkeypatch.setattr(session.config, "browser_profile_name", "Profile2")  # type: ignore[attr-defined]
    session._PAGE = None
    built = False

    async def build(profile: str) -> Path:
        nonlocal built
        built = True
        return Path("/clone")

    monkeypatch.setattr(session, "_clone_profile", build)
    await session.prewarm_clone()
    assert built is True


async def test_prewarm_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session.config, "browser_profile_path", None)  # type: ignore[attr-defined]
    session._PAGE = None

    async def boom(profile: str) -> Path:
        raise OSError("disk gone")

    monkeypatch.setattr(session, "_clone_profile", boom)
    await session.prewarm_clone()


async def test_prewarm_noop_when_no_profile_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session.config, "browser_profile_path", None)  # type: ignore[attr-defined]
    monkeypatch.setattr(session.config, "browser_profile_name", None)  # type: ignore[attr-defined]
    session._PAGE = None
    called = False

    async def fail(profile: str) -> Path:
        nonlocal called
        called = True
        return Path("/")

    monkeypatch.setattr(session, "_clone_profile", fail)
    await session.prewarm_clone()
    assert called is False
