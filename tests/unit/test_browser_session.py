from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from supporter.config import config
from supporter.tools.browser import guardrails, profiles, session
from supporter.tools.browser.profiles import ChromeProfile

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


async def test_cleanup_blank_tabs_keeps_working_tab_and_closes_other_blanks() -> None:
    class _Tab:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            ctx.pages.remove(self)

    working = _Tab("about:blank")
    restored_blank = _Tab("chrome://newtab/")
    content = _Tab("https://example.test/")
    ctx = cast("Any", type("", (), {"pages": [restored_blank, working, content]})())
    session._CONTEXT = ctx
    session._PAGE = cast("Any", working)

    await session.cleanup_blank_tabs()

    assert working.closed is False
    assert restored_blank.closed is True
    assert content.closed is False
    assert ctx.pages == [working, content]


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
    clock, slept = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.001)
    session._LAST_ACTION_TS = clock.t - 60.0

    await session.pace()

    assert slept == []
    _reset_pace_globals()


async def test_pace_increments_action_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, _ = _install_pace_harness(monkeypatch)
    session._LAST_ACTION_TS = clock.t - 60.0

    await session.pace()

    assert session._ACTION_COUNT == 1
    _reset_pace_globals()


async def test_pace_action_cap_triggers_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, _ = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "action_cap", lambda: 5)
    session._LAST_ACTION_TS = clock.t - 60.0
    session._ACTION_CAP_CEILING = 5
    session._ACTION_COUNT = 4

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
    assert session._ACTION_CAP_CEILING == 0
    _reset_pace_globals()


async def test_pace_action_cap_raises_when_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, _ = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "action_cap", lambda: 5)
    session._LAST_ACTION_TS = clock.t - 60.0
    session._ACTION_CAP_CEILING = 5
    session._ACTION_COUNT = 4

    async def deny(_title: str, _detail: str) -> bool:
        return False

    guardrails.browse_confirmation_callback = deny

    try:
        with pytest.raises(RuntimeError, match="Action cap reached"):
            await session.pace()
    finally:
        guardrails.browse_confirmation_callback = None
    _reset_pace_globals()


async def test_pace_action_cap_without_callback_resets_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, _ = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "action_cap", lambda: 5)
    session._LAST_ACTION_TS = clock.t - 60.0
    session._ACTION_CAP_CEILING = 5
    session._ACTION_COUNT = 4

    guardrails.browse_confirmation_callback = None

    await session.pace()

    assert session._ACTION_COUNT == 0
    _reset_pace_globals()


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


async def test_resolve_profile_returns_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", "EnvProfile")
    session._SELECTED_PROFILE = None
    result = await session._resolve_profile_name()
    assert result == "EnvProfile"
    assert session._SELECTED_PROFILE is None


async def test_resolve_profile_caches_callback_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", None)
    session._SELECTED_PROFILE = None

    async def fake_callback(profiles: list[Any]) -> str | None:
        return "PickedProfile"

    guardrails.browse_profile_select_callback = fake_callback
    test_profiles = [
        ChromeProfile(dir_name="Profile1", display_name="P1", email="a@b.com"),
        ChromeProfile(dir_name="Profile2", display_name="P2", email=""),
    ]
    monkeypatch.setattr(
        profiles,
        "list_profiles",
        lambda _: test_profiles,
    )

    try:
        result1 = await session._resolve_profile_name()
        result2 = await session._resolve_profile_name()
        assert result1 == "PickedProfile"
        assert result2 == "PickedProfile"
        assert session._SELECTED_PROFILE == "PickedProfile"
    finally:
        guardrails.browse_profile_select_callback = None
        session._SELECTED_PROFILE = None


async def test_resolve_profile_auto_skips_when_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", None)
    session._SELECTED_PROFILE = None

    test_profiles = [
        ChromeProfile(dir_name="OnlyProfile", display_name="Only", email=""),
    ]
    monkeypatch.setattr(
        profiles,
        "list_profiles",
        lambda _: test_profiles,
    )

    result = await session._resolve_profile_name()
    assert result == "OnlyProfile"
    assert session._SELECTED_PROFILE == "OnlyProfile"


async def test_resolve_profile_uses_default_when_no_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", None)
    session._SELECTED_PROFILE = None
    monkeypatch.setattr(profiles, "list_profiles", lambda _: [])

    result = await session._resolve_profile_name()
    assert result == "Default"
    assert session._SELECTED_PROFILE == "Default"


async def test_resolve_profile_raises_when_no_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", None)
    session._SELECTED_PROFILE = None
    guardrails.browse_profile_select_callback = None
    test_profiles = [
        ChromeProfile(dir_name="P1", display_name="P1", email="a@b.com"),
        ChromeProfile(dir_name="P2", display_name="P2", email="c@d.com"),
    ]
    monkeypatch.setattr(profiles, "list_profiles", lambda _: test_profiles)

    with pytest.raises(RuntimeError, match="no interactive picker available"):
        await session._resolve_profile_name()


async def test_resolve_profile_raises_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "browser_profile_name", None)
    session._SELECTED_PROFILE = None

    async def cancel_callback(profiles: list[Any]) -> str | None:
        return None

    guardrails.browse_profile_select_callback = cancel_callback
    test_profiles = [
        ChromeProfile(dir_name="P1", display_name="P1", email="a@b.com"),
        ChromeProfile(dir_name="P2", display_name="P2", email="c@d.com"),
    ]
    monkeypatch.setattr(profiles, "list_profiles", lambda _: test_profiles)

    with pytest.raises(RuntimeError, match="cancelled"):
        await session._resolve_profile_name()


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _reset_pace_globals() -> None:
    session._ACTION_COUNT = 0
    session._ACTION_CAP_CEILING = 0
    session._LAST_ACTION_TS = 0.0
    session._ACTION_TIMES.clear()
    session._SESSION_START_TS = 0.0
    session._TEMPO = 1.0


def _install_pace_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Clock, list[float]]:
    clock = _Clock()
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.t += seconds

    monkeypatch.setattr(time, "monotonic", clock)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.0)
    monkeypatch.setattr(guardrails, "maybe_idle_gap", lambda: 0.0)
    monkeypatch.setattr(guardrails, "fatigue_multiplier", lambda _m: 1.0)
    monkeypatch.setattr(guardrails, "next_tempo", lambda _t: 1.0)
    monkeypatch.setattr(guardrails, "action_cap", lambda: 10_000)
    _reset_pace_globals()
    return clock, slept


async def test_pace_sets_session_start_once(monkeypatch: pytest.MonkeyPatch) -> None:
    clock, _ = _install_pace_harness(monkeypatch)
    await session.pace()
    first = session._SESSION_START_TS
    assert first == 1000.0
    clock.t += 5.0
    await session.pace()
    assert first == session._SESSION_START_TS
    _reset_pace_globals()


async def test_pace_throttles_rapid_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    _clock, slept = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "random_gap", lambda: 0.5)
    for _ in range(40):
        await session.pace()
    assert any(s > 0.5 for s in slept), "governor never throttled a fast burst"
    window = session._ACTION_TIMES
    span = window[-1] - window[0]
    rate = len(window) / span * 60.0 if span > 0 else 0.0
    assert rate <= guardrails.ACTIONS_PER_MINUTE_MAX + 1.0
    _reset_pace_globals()


async def test_pace_idle_gap_fires_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, slept = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "maybe_idle_gap", lambda: 42.0)
    await session.pace()
    assert 42.0 in slept
    _reset_pace_globals()


async def test_pace_applies_fatigue_and_tempo_to_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock, slept = _install_pace_harness(monkeypatch)
    monkeypatch.setattr(guardrails, "random_gap", lambda: 1.0)
    monkeypatch.setattr(guardrails, "fatigue_multiplier", lambda _m: 1.5)
    monkeypatch.setattr(guardrails, "next_tempo", lambda _t: 1.2)
    session._LAST_ACTION_TS = clock.t
    await session.pace()
    assert slept and abs(slept[0] - 1.8) < 1e-9
    _reset_pace_globals()


async def test_close_session_resets_pace_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pace_harness(monkeypatch)
    await session.pace()
    session._TEMPO = 1.25
    assert session._SESSION_START_TS != 0.0
    session._PAGE = None
    session._CONTEXT = None
    session._PWS = None
    session._LIFECYCLE_TASK = None
    await session.close_session()
    assert session._SESSION_START_TS == 0.0
    assert session._TEMPO == 1.0
    assert len(session._ACTION_TIMES) == 0
    assert session._ACTION_COUNT == 0
