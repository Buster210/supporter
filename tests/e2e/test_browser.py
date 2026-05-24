from __future__ import annotations

import re

import pytest

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
