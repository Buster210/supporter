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

# Seconds to let Cloudflare process the click before checking for completion.
_SETTLE_SECONDS = 1.2


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


async def _looks_solved(page: Page, frame: Frame) -> bool:
    await asyncio.sleep(_SETTLE_SECONDS)
    if frame not in page.frames:
        return True
    try:
        token = await page.evaluate(
            "() => { const el = document.querySelector("
            "'[name=cf-turnstile-response]'); return el ? el.value : ''; }"
        )
        return bool(token)
    except Exception as exc:
        logger.debug(f"Turnstile token check failed: {exc}")
        return False


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
