from __future__ import annotations

import pytest

from supporter.tools.browser import session
from supporter.tools.browser.tool import browse

FRAME_PAGE = (
    "data:text/html,"
    "<html><body><h1>Outer</h1>"
    "<iframe id='fr' srcdoc=\""
    "<html><body><h2 id='inner'>InnerHeading</h2>"
    "<button aria-label='InBtn'>go</button>"
    "</body></html>"
    '"></iframe>'
    "</body></html>"
)


@pytest.mark.asyncio
async def test_frame_drill_shows_inner_tree(throwaway_browser: None) -> None:
    await browse("navigate", url=FRAME_PAGE)
    result = await browse("frame", selector="#fr")
    assert "InnerHeading" in result
    assert "InBtn" in result
    assert session.active_frame_selector() == "#fr"
