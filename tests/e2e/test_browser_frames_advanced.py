from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools import resolved_project_root
from supporter.tools.browser import guardrails, session
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


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


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


@pytest.mark.asyncio
async def test_frame_drill_shows_inner_tree(throwaway_browser: Path) -> None:
    await browse("navigate", url=FRAME_PAGE)
    result = await browse("frame", selector="#fr")
    assert "InnerHeading" in result
    assert "InBtn" in result
    assert session.active_frame_selector() == "#fr"
