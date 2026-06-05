from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ...logger import logger
from . import humanize

if TYPE_CHECKING:
    from patchright.async_api import Frame, Page

# Cloudflare serves the Turnstile widget from a same-origin-proxied iframe whose
# URL always contains this host. Detecting the frame is reliable; the parent
# page's accessibility tree does not include the cross-origin iframe contents.
_FRAME_MARKER = "challenges.cloudflare.com"

# Widget container on the host page (present for both inline and managed modes).
_WIDGET_SELECTOR = ".cf-turnstile, [data-sitekey]"

# Clickable checkbox lives inside the Turnstile iframe, behind a (closed) shadow
# root that Patchright pierces. The hidden `#cf-turnstile-response` token input is
# deliberately excluded — it is not clickable.
_CHECKBOX_SELECTORS: tuple[str, ...] = (
    "input[type=checkbox]",
    "[role=checkbox]",
)

# Poll interval and overall budget for confirming the click landed (token
# populated or frame detached), instead of a single fixed settle + one read.
_SOLVE_POLL_SECONDS = 0.4
_SOLVE_TIMEOUT_SECONDS = 5.0


def _turnstile_frame(page: Page) -> Frame | None:
    for frame in page.frames:
        if _FRAME_MARKER in (frame.url or ""):
            return frame
    return None


async def detect_turnstile_in_page(page: Page) -> bool:
    if _turnstile_frame(page) is not None:
        return True
    try:
        return await page.locator(_WIDGET_SELECTOR).first.count() > 0
    except Exception as exc:
        logger.debug(f"Turnstile widget check failed: {exc}")
        return False


async def _read_token(page: Page) -> str:
    try:
        token = await page.evaluate(
            "() => { const el = document.querySelector("
            "'[name=cf-turnstile-response]'); return el ? el.value : ''; }"
        )
        return token or ""
    except Exception as exc:
        logger.debug(f"Turnstile token check failed: {exc}")
        return ""


async def _looks_solved(page: Page, frame: Frame) -> bool:
    elapsed = 0.0
    while True:
        if frame not in page.frames:
            return True
        if await _read_token(page):
            return True
        if elapsed >= _SOLVE_TIMEOUT_SECONDS:
            return False
        await asyncio.sleep(_SOLVE_POLL_SECONDS)
        elapsed += _SOLVE_POLL_SECONDS


async def solve_cloudflare(page: Page) -> str:
    frame = _turnstile_frame(page)
    if frame is None:
        return "No Cloudflare Turnstile detected on the page."

    for selector in _CHECKBOX_SELECTORS:
        try:
            checkbox = frame.locator(selector).first
            if await checkbox.count() == 0:
                continue
            await humanize.human_click(page, "", locator=checkbox)
        except Exception as exc:
            logger.debug(f"Turnstile click failed ({selector}): {exc}")
            continue

        if await _looks_solved(page, frame):
            return "Cloudflare Turnstile solved."
        return (
            "Clicked Turnstile checkbox but could not confirm — re-snapshot to check."
        )

    return (
        "Turnstile present but no clickable checkbox "
        "(managed/invisible challenge) — manual solve needed."
    )
