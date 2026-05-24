from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from supporter.config import config
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_keep_open_defaults_to_true_unless_explicitly_false(
    value: bool | None, expected: bool
) -> None:
    session._KEEP_OPEN = value
    assert session.keep_open() is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (True, True), (False, False)],
)
def test_pinned_open_only_when_keep_open_explicitly_true(
    value: bool | None, expected: bool
) -> None:
    session._KEEP_OPEN = value
    assert session.pinned_open() is expected


def test_is_active_reflects_page_presence() -> None:
    session._PAGE = None
    assert session.is_active() is False
    session._PAGE = cast("Any", object())
    assert session.is_active() is True


async def test_prompt_lifecycle_is_asked_once() -> None:
    session._KEEP_OPEN = None
    calls: list[tuple[str, str]] = []

    async def cb(title: str, detail: str) -> bool:
        calls.append((title, detail))
        return False

    guardrails.browse_confirmation_callback = cb
    try:
        await session._prompt_lifecycle()
        await session._prompt_lifecycle()
    finally:
        guardrails.browse_confirmation_callback = None

    assert len(calls) == 1
    assert session._KEEP_OPEN is False


async def test_prompt_lifecycle_fails_open_when_unwired() -> None:
    session._KEEP_OPEN = None
    guardrails.browse_confirmation_callback = None
    await session._prompt_lifecycle()
    assert session._KEEP_OPEN is True


def test_active_page_returns_current_page() -> None:
    session._PAGE = cast("Any", object())
    assert session.active_page() is session._PAGE


def test_list_pages_empty_when_no_context() -> None:
    session._CONTEXT = None
    assert session.list_pages() == []


def test_list_pages_returns_context_pages() -> None:
    pages = [cast("Any", object()), cast("Any", object())]
    context = cast("Any", type("", (), {"pages": pages})())
    session._CONTEXT = context
    assert session.list_pages() == pages


def test_set_active_updates_page_and_clears_frame() -> None:
    page_a = cast("Any", object())
    page_b = cast("Any", object())
    session._PAGE = page_a
    session._FRAME_SELECTOR = "iframe#main"

    session.set_active(page_b)

    assert session._PAGE is page_b
    assert session._FRAME_SELECTOR is None


def test_active_frame_selector_returns_frame() -> None:
    session._FRAME_SELECTOR = "iframe#content"
    assert session.active_frame_selector() == "iframe#content"

    session._FRAME_SELECTOR = None
    assert session.active_frame_selector() is None


def test_set_frame_updates_selector() -> None:
    session.set_frame("iframe#nav")
    assert session._FRAME_SELECTOR == "iframe#nav"

    session.set_frame(None)
    assert session._FRAME_SELECTOR is None


async def test_get_session_returns_existing_when_active() -> None:
    pws = cast("Any", object())
    context = cast("Any", object())
    page = cast("Any", object())
    session._PWS = pws
    session._CONTEXT = context
    session._PAGE = page

    result = await session.get_session()

    assert result == (pws, context, page)


def test_clear_lifecycle_task_sets_global_to_none() -> None:
    session._LIFECYCLE_TASK = cast("Any", object())
    session._clear_lifecycle_task(object())
    assert session._LIFECYCLE_TASK is None


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("ProcessSingleton lock", True),
        ("SingletonLock conflict", True),
        ("profile appears to be in use", True),
        ("user data directory is already in use", True),
        ("Some other error", False),
        ("", False),
    ],
)
def test_is_profile_lock_error_detects_known_markers(msg: str, expected: bool) -> None:
    exc = RuntimeError(msg)
    assert session._is_profile_lock_error(exc) is expected


async def test_pace_waits_when_elapsed_less_than_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now
    session._ACTION_COUNT = 0
    monkeypatch.setattr(guardrails, "random_gap", lambda: 10.0)

    slept: list[float] = []

    async def fake_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await session.pace()

    assert len(slept) == 1
    assert slept[0] == pytest.approx(10.0, rel=0.5)


async def test_pace_skips_sleep_when_elapsed_exceeds_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now - 60.0
    session._ACTION_COUNT = 0
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)

    slept: list[float] = []

    async def fake_sleep(secs: float) -> None:
        slept.append(secs)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await session.pace()

    assert len(slept) == 0


async def test_pace_increments_action_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now - 60.0
    session._ACTION_COUNT = 0
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)

    async def no_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    await session.pace()

    assert session._ACTION_COUNT == 1


async def test_pace_action_cap_triggers_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now - 60.0
    session._ACTION_COUNT = guardrails.ACTION_CAP - 1
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)

    async def no_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    prompted: list[str] = []

    async def cb(_title: str, detail: str) -> bool:
        prompted.append(detail)
        return True

    guardrails.browse_confirmation_callback = cb

    try:
        await session.pace()
    finally:
        guardrails.browse_confirmation_callback = None

    assert len(prompted) == 1
    assert session._ACTION_COUNT == 0


async def test_pace_action_cap_raises_when_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now - 60.0
    session._ACTION_COUNT = guardrails.ACTION_CAP - 1
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)

    async def no_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    async def deny(_title: str, _detail: str) -> bool:
        return False

    guardrails.browse_confirmation_callback = deny

    try:
        with pytest.raises(RuntimeError, match="Action cap reached"):
            await session.pace()
    finally:
        guardrails.browse_confirmation_callback = None


async def test_pace_action_cap_without_callback_resets_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    session._LAST_ACTION_TS = now - 60.0
    session._ACTION_COUNT = guardrails.ACTION_CAP - 1
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)

    async def no_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    guardrails.browse_confirmation_callback = None

    await session.pace()

    assert session._ACTION_COUNT == 0


def test_profile_dir_reads_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_path", "/custom/path")
    result = session._profile_dir()
    assert result == Path("/custom/path")


def test_profile_dir_fallback_uses_platform_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_path", None)
    result = session._profile_dir()
    home = Path.home()
    assert str(result).startswith(str(home))
    assert "Google" in str(result) or "Chrome" in str(result)


def test_profile_name_reads_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", "TestProfile")
    assert session._profile_name() == "TestProfile"
