from __future__ import annotations

import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools.browser import guardrails, session
from supporter.tools.browser.tool import browse

PAGE = (
    "data:text/html,"
    "<html><body>"
    "<h1>Smoke</h1>"
    "<button id='b1' aria-label='Click me'>Click me</button>"
    "<input id='t1' type='text' placeholder='search' />"
    "<p id='out'>idle</p>"
    "<script>"
    "document.getElementById('b1').onclick="
    "()=>document.getElementById('out').textContent='clicked';"
    "document.getElementById('t1').oninput="
    "(e)=>document.getElementById('out').textContent='typed:'+e.target.value;"
    "</script>"
    "</body></html>"
)


@pytest.fixture
async def throwaway_browser(tmp_path: Path) -> AsyncIterator[None]:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()  # empty -> _ensure_profile makes a fresh login-free profile
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True
    guardrails.register_browse_callback(confirmation=_always_allow)
    try:
        yield
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


def _first_ref(snapshot: str, needle: str) -> str:
    for line in snapshot.splitlines():
        if needle in line:
            match = re.search(r"\[ref=(e\d+)\]", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no [ref=eN] line containing {needle!r} in:\n{snapshot}")


@pytest.mark.asyncio
async def test_browse_navigate_snapshot_click_type(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    assert "[ref=e" in snap
    assert 'button "Click me"' in snap

    click_ref = _first_ref(snap, '"Click me"')
    after_click = await browse("click", ref=click_ref)
    assert "clicked" in after_click

    type_ref = _first_ref(after_click, "textbox")
    after_type = await browse("type", ref=type_ref, text="hello")
    assert "typed:hello" in after_type


@pytest.mark.asyncio
async def test_browse_stale_ref_is_recoverable(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("click", ref="e9999")
    assert result.startswith("Error: ref e9999 not found")


@pytest.mark.asyncio
async def test_browse_unknown_action_is_recoverable(throwaway_browser: None) -> None:
    result = await browse("teleport")
    assert result.startswith("Error: Unknown action")


@pytest.mark.asyncio
async def test_lifecycle_prompt_asked_once_and_keep_open_skips_hint(
    throwaway_browser: None,
) -> None:
    # _always_allow answers the lifecycle prompt True -> keep open, no hint.
    first = await browse("navigate", url=PAGE)
    assert "call browse(action='close')" not in first
    assert session.keep_open() is True

    # second navigate must not re-prompt the lifecycle question.
    await browse("navigate", url=PAGE)
    assert session.keep_open() is True


@pytest.mark.asyncio
async def test_close_when_done_flow(tmp_path: Path) -> None:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True

    # First confirm (lifecycle) -> No (close when done); later confirms (close) -> Yes.
    answers = iter([False])

    async def scripted(_title: str, _detail: str) -> bool:
        return next(answers, True)

    guardrails.register_browse_callback(confirmation=scripted)
    try:
        nav = await browse("navigate", url=PAGE)
        assert "call browse(action='close')" in nav
        assert session.keep_open() is False

        closed = await browse("close")
        assert closed == "Browser closed."
        assert session.is_active() is False

        again = await browse("close")
        assert again == "Browser already closed."
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless
