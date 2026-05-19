from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools.browser import guardrails, session
from supporter.tools.browser.tool import browse

PAGE = (
    "data:text/html,"
    "<html><body>"
    "<h1>Tabs</h1>"
    "<div id='box' aria-label='Box'>hello <b>world</b></div>"
    "</body></html>"
)
SECOND = "data:text/html,<html><body><h2>Second</h2></body></html>"


@pytest.fixture
async def throwaway_browser(tmp_path: Path) -> AsyncIterator[None]:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
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


def _active_index(listing: str) -> int:
    for line in listing.splitlines():
        if line.startswith("*"):
            return int(line[line.index("[") + 1 : line.index("]")])
    raise AssertionError(f"no active tab marker in:\n{listing}")


@pytest.mark.asyncio
async def test_newtab_adds_one_active_tab(throwaway_browser: None) -> None:
    # A fresh session already owns a startup tab, so assert the delta, not a
    # fixed count: newtab adds exactly one and makes it active.
    await browse("navigate", url=PAGE)
    before = len(session.list_pages())

    new = await browse("newtab", url=SECOND)
    assert "Second" in new
    assert len(session.list_pages()) == before + 1

    listing = await browse("tabs")
    last = len(session.list_pages()) - 1
    assert _active_index(listing) == last


@pytest.mark.asyncio
async def test_tab_switch_changes_active(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    page_index = session.list_pages().index(session.active_page())
    await browse("newtab", url=SECOND)

    switched = await browse("tab", index=page_index)
    assert "Tabs" in switched
    assert session.active_page() is session.list_pages()[page_index]


@pytest.mark.asyncio
async def test_tab_index_out_of_range(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("tab", index=9)
    assert result.startswith("Error: tab index 9 out of range")


@pytest.mark.asyncio
async def test_closetab_activates_remaining(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    before = len(session.list_pages())
    await browse("newtab", url=SECOND)
    assert len(session.list_pages()) == before + 1

    after = await browse("closetab")  # closes active (the new SECOND tab)
    assert len(session.list_pages()) == before
    # active tab is repointed to the last surviving tab (PAGE, heading "Tabs").
    assert "Tabs" in after
    assert "Second" not in after


@pytest.mark.asyncio
async def test_closetab_explicit_index_zero_closes_that_tab(
    throwaway_browser: None,
) -> None:
    # index=0 must close tab 0 (a non-active tab here), NOT be treated as
    # "no index → active". The active tab must survive untouched.
    await browse("navigate", url=PAGE)
    await browse("newtab", url=SECOND)  # SECOND is now active, at the end
    active_before = session.active_page()
    tab_zero = session.list_pages()[0]
    count_before = len(session.list_pages())

    await browse("closetab", index=0)
    assert tab_zero not in session.list_pages()
    assert len(session.list_pages()) == count_before - 1
    assert session.active_page() is active_before  # active untouched


@pytest.mark.asyncio
async def test_extract_visible_text(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    text = await browse("extract", selector="#box")
    assert text.strip() == "hello world"


@pytest.mark.asyncio
async def test_extract_inner_html(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    html = await browse("extract", selector="#box", html=True)
    assert "<b>world</b>" in html


@pytest.mark.asyncio
async def test_extract_requires_target(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("extract")
    assert result.startswith("Error: extract needs")


@pytest.mark.asyncio
async def test_eval_returns_result(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("eval", script="1 + 2")
    assert result == "eval result: 3"


@pytest.mark.asyncio
async def test_eval_requires_script(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("eval")
    assert result.startswith("Error: 'script' is required")


@pytest.mark.asyncio
async def test_eval_always_confirms_and_denial_blocks(tmp_path: Path) -> None:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True

    prompts: list[str] = []

    async def deny_eval(title: str, _detail: str) -> bool:
        prompts.append(title)
        return "eval" not in title.lower()  # approve lifecycle, deny the script

    guardrails.register_browse_callback(confirmation=deny_eval)
    try:
        await browse("navigate", url=PAGE)
        result = await browse("eval", script="window.__x = 5")
        assert result == "Error: action cancelled."
        assert any("eval" in p.lower() for p in prompts)
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless
