from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools import resolved_project_root
from supporter.tools.browser import guardrails, session
from supporter.tools.browser.tool import browse

# Outer page embedding an iframe (srcdoc) with its own heading + button. Frame
# reads must see the INNER content, not the outer page.
FRAME_PAGE = (
    "data:text/html,"
    "<html><body><h1>Outer</h1>"
    "<iframe id='fr' srcdoc=\""
    "<html><body><h2 id='inner'>InnerHeading</h2>"
    "<button aria-label='InBtn'>go</button>"
    "<input id='cb' type='checkbox' aria-label='Agree' />"
    "<input id='inp' type='text' aria-label='Name' />"
    "</body></html>"
    '"></iframe>'
    "</body></html>"
)
PLAIN_PAGE = "data:text/html,<html><body><h1>Plain</h1></body></html>"


@pytest.fixture
async def throwaway_browser(tmp_path: Path) -> AsyncIterator[Path]:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True
    guardrails.register_browse_callback(confirmation=_always_allow)

    work = resolved_project_root() / ".tier5_tmp"
    work.mkdir(exist_ok=True)
    try:
        yield work
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


@pytest.mark.asyncio
async def test_frame_drill_shows_inner_tree(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    result = await browse("frame", selector="#fr")
    assert "InnerHeading" in result  # inner content surfaced
    assert "InBtn" in result
    assert session.active_frame_selector() == "#fr"


@pytest.mark.asyncio
async def test_extract_reads_inside_drilled_frame(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    text = await browse("extract", selector="#inner")
    assert text.strip() == "InnerHeading"


@pytest.mark.asyncio
async def test_frame_clear_returns_to_top(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    cleared = await browse("frame")  # no selector clears
    assert "top page" in cleared.lower()
    assert session.active_frame_selector() is None


@pytest.mark.asyncio
async def test_frame_no_match_reports_error(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    result = await browse("frame", selector="#does-not-exist")
    assert result.startswith("Error: no iframe matched")
    assert session.active_frame_selector() is None  # state untouched on failure


@pytest.mark.asyncio
async def test_extract_in_frame_rejects_ref(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    result = await browse("extract", ref="e1")
    assert result.startswith("Error: inside a frame")


@pytest.mark.asyncio
async def test_tab_switch_clears_frame(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    assert session.active_frame_selector() == "#fr"
    await browse("newtab", url=PLAIN_PAGE)  # switching tabs clears the frame
    assert session.active_frame_selector() is None


@pytest.mark.asyncio
async def test_waitnetwork_returns_snapshot(throwaway_browser: Path) -> None:
    await browse("navigate", url=PLAIN_PAGE)
    result = await browse("waitnetwork")
    assert "Plain" in result  # snapshot of the settled page


@pytest.mark.asyncio
async def test_click_inside_frame(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    # The checkbox starts unchecked; an in-frame click must toggle it.
    inner = session.active_page().frame_locator("#fr").locator("#cb")
    assert await inner.is_checked() is False
    await browse("click", selector="#cb")
    assert await inner.is_checked() is True


@pytest.mark.asyncio
async def test_type_inside_frame(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    await browse("type", selector="#inp", text="hi there")
    inner = session.active_page().frame_locator("#fr").locator("#inp")
    assert await inner.input_value() == "hi there"


@pytest.mark.asyncio
async def test_hover_inside_frame(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    # Hover is passive (no gate); it must resolve the in-frame target and
    # return a snapshot rather than an error.
    result = await browse("hover", selector="button")
    assert not result.startswith("Error")


@pytest.mark.asyncio
async def test_click_in_frame_rejects_ref(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    await browse("frame", selector="#fr")
    result = await browse("click", ref="e1")
    assert result.startswith("Error: inside a frame")


@pytest.mark.asyncio
async def test_click_top_page_still_uses_ref(throwaway_browser: Path) -> None:
    # No frame drilled: a ref click on the outer page works exactly as before
    # (lossless guard for the top-page path).
    snap = await browse("navigate", url=FRAME_PAGE)
    ref = ""
    for line in snap.splitlines():
        if '"Outer"' in line and "[ref=" in line:
            start = line.index("[ref=") + len("[ref=")
            ref = line[start : line.index("]", start)]
            break
    assert ref, f"no ref for the outer heading in:\n{snap}"
    result = await browse("click", ref=ref)
    assert not result.startswith("Error")
