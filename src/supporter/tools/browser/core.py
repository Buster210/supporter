from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...logger import logger
from . import guardrails

__all__ = ["BrowseRequest", "_page_host"]


@dataclass(frozen=True)
class BrowseRequest:
    action: str
    url: str = ""
    ref: str = ""
    text: str = ""
    depth: int = 0
    compact: bool = False
    delay_ms: int = 100
    key: str = ""
    value: str = ""
    selector: str = ""
    dx: int = 0
    dy: int = 0
    script: str = ""
    index: int = -1
    html: bool = False
    path: str = ""
    stamp: str = ""
    variable: str = ""
    force: bool = False
    full_page: bool = False


async def _page_host(page: Any) -> str:
    # page.url is Playwright's tracked main-frame URL — a sync property — so this
    # avoids a JS round-trip on every humanized action (this runs per action via
    # _effective_fast and _confirm_or_block).
    try:
        return guardrails.host_from_url(page.url or "")
    except Exception:
        logger.debug("Could not read page URL", exc_info=True)
        return ""
