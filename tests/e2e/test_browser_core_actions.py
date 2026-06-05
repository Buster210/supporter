from __future__ import annotations

import re

import pytest

from supporter.tools.browser.tool import browse

PAGE = (
    "data:text/html,"
    "<html><body>"
    "<h1>Core</h1>"
    "<button id='b1' aria-label='Hover target'>Hover target</button>"
    "<p id='out'>idle</p>"
    "<script>"
    "document.getElementById('b1').onmouseover="
    "()=>document.getElementById('out').textContent='hovered';"
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
async def test_hover_fires_mouseover(throwaway_browser: None) -> None:
    snap = await browse("navigate", url=PAGE)
    ref = _first_ref(snap, '"Hover target"')
    after = await browse("hover", ref=ref)
    assert "hovered" in after
