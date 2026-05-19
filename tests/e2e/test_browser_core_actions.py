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
    "<html><body style='margin:0'>"
    "<h1>Core</h1>"
    "<button id='hov' aria-label='Hover target'>hover me</button>"
    "<select id='sel' aria-label='Pick'>"
    "<option value='a'>Apple</option>"
    "<option value='b'>Banana</option>"
    "</select>"
    "<input id='inp' type='text' aria-label='typehere' />"
    "<div style='height:3000px'>tall</div>"
    "<button id='bottom' aria-label='At bottom'>bottom</button>"
    "<p id='out'>idle</p>"
    "<script>"
    "document.getElementById('hov').onmouseover="
    "()=>document.getElementById('out').textContent='hovered';"
    "document.getElementById('sel').onchange="
    "(e)=>document.getElementById('out').textContent='picked:'+e.target.value;"
    "document.getElementById('inp').onkeydown="
    "(e)=>{if(e.key==='Enter')"
    "document.getElementById('out').textContent='pressed:Enter';};"
    "setTimeout(()=>{var d=document.createElement('div');"
    "d.id='late';d.textContent='here';document.body.appendChild(d);},300);"
    "</script>"
    "</body></html>"
)


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


def _first_ref(snapshot: str, needle: str) -> str:
    for line in snapshot.splitlines():
        if needle in line:
            match = re.search(r"\[ref=(e\d+)\]", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no [ref=eN] line containing {needle!r} in:\n{snapshot}")


@pytest.mark.asyncio
async def test_hover_fires_mouseover(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, '"Hover target"')
    after = await browse("hover", ref=ref)
    assert "hovered" in after


@pytest.mark.asyncio
async def test_select_by_value(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, "combobox")
    after = await browse("select", ref=ref, value="b")
    assert "picked:b" in after


@pytest.mark.asyncio
async def test_select_by_label(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, "combobox")
    after = await browse("select", ref=ref, text="Apple")
    assert "picked:a" in after


@pytest.mark.asyncio
async def test_press_enter_in_focused_input(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, "textbox")
    after = await browse("press", ref=ref, key="Enter")
    assert "pressed:Enter" in after


@pytest.mark.asyncio
async def test_scroll_into_view_of_ref(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, '"At bottom"')
    after = await browse("scroll", ref=ref)
    assert "[ref=e" in after


@pytest.mark.asyncio
async def test_scroll_by_delta(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    after = await browse("scroll", dy=500)
    assert "[ref=e" in after


@pytest.mark.asyncio
async def test_wait_for_selector_that_appears_late(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    after = await browse("wait", selector="#late")
    assert "[ref=e" in after


@pytest.mark.asyncio
async def test_forward_returns_to_second_page(throwaway_browser: None) -> None:
    second = "data:text/html,<html><body><h2>Second</h2></body></html>"
    await browse("navigate", url=PAGE)
    await browse("navigate", url=second)
    back = await browse("back")
    assert "Core" in back
    fwd = await browse("forward")
    assert "Second" in fwd


@pytest.mark.asyncio
async def test_press_requires_key(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("press")
    assert result.startswith("Error: 'key' is required")


@pytest.mark.asyncio
async def test_scroll_requires_target(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("scroll")
    assert result.startswith("Error: scroll needs")


@pytest.mark.asyncio
async def test_select_requires_option(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, "combobox")
    result = await browse("select", ref=ref)
    assert result.startswith("Error: select needs")
