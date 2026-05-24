from __future__ import annotations

import pytest

from supporter.tools.browser import session
from supporter.tools.browser.tool import browse

PAGE = (
    "data:text/html,"
    "<html><body>"
    "<h1>Tabs</h1>"
    "<div id='box' aria-label='Box'>hello <b>world</b></div>"
    "</body></html>"
)


@pytest.mark.asyncio
async def test_newtab_adds_one_active_tab(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    before = len(session.list_pages())

    await browse("newtab", url=PAGE)

    assert len(session.list_pages()) == before + 1


@pytest.mark.asyncio
async def test_eval_returns_result(throwaway_browser: None) -> None:
    await browse("navigate", url=PAGE)
    result = await browse("eval", script="1 + 1")
    assert "2" in result
