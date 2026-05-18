from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...logger import logger
from ..base import ToolError
from . import guardrails, humanize, session, snapshot


@dataclass(frozen=True)
class BrowseRequest:
    action: str
    url: str = ""
    ref: str = ""
    text: str = ""
    depth: int = 0
    compact: bool = False
    delay_ms: int = 100
    fast: bool = False


async def browse(
    action: str,
    url: str = "",
    ref: str = "",
    text: str = "",
    depth: int = 0,
    compact: bool = False,
    delay_ms: int = 100,
    *,
    fast: bool = False,
) -> str:
    """Browser automation tool using your Chrome profile and cookies.

    Navigate websites, read page content (accessibility snapshot), click
    elements, type text, and take screenshots. Runs with your real logged-in
    session so you can operate ChatGPT, Claude, Gemini, Twitter/X, etc.

    Actions and the params each uses:
        navigate (url): load a URL.
        back: go back one history entry.
        snapshot (depth, compact): return the accessibility tree.
        screenshot: capture the viewport (returns a byte/size summary).
        click (ref): click the element [ref=eN] from a snapshot.
        type (ref, text): type text into the element [ref=eN].
        close: close the browser (asks for confirmation).

    Args:
        action: What to do (see list above).
        url: URL to navigate to (navigate).
        ref: Element reference [ref=eN] from a snapshot (click/type).
        text: Text to type (type).
        depth: Accessibility tree depth for snapshot. 0 = full tree.
        compact: If True, return only interactive elements [ref=eN].
        delay_ms: Extra delay after action in ms. Default 100.
        fast: If True, skip humanized motion + pacing for this action (faster,
            less stealthy — use only on sites that don't fingerprint input).

    Returns:
        Snapshot text, confirmation prompt, or error message.
    """
    logger.info(
        f"Tool: browse — action={action}, url={url!r}, ref={ref!r}, "
        f"depth={depth}, compact={compact}, fast={fast}"
    )

    if action not in _HANDLERS:
        names = ", ".join(sorted(_HANDLERS))
        return f"Error: Unknown action '{action}'. Valid actions: {names}"

    req = BrowseRequest(
        action=action,
        url=url,
        ref=ref,
        text=text,
        depth=depth,
        compact=compact,
        delay_ms=delay_ms,
        fast=fast,
    )
    return await _HANDLERS[action](req)


async def _page_or_error() -> Any:
    try:
        _pws, _context, page = await session.get_session()
    except Exception as e:
        raise ToolError(f"Browser session failed: {e}") from e
    return page


def _wrap_action_errors(action: str) -> Callable[..., Any]:
    def deco(
        fn: Callable[[BrowseRequest], Awaitable[str]],
    ) -> Callable[[BrowseRequest], Awaitable[str]]:
        async def wrapped(req: BrowseRequest) -> str:
            try:
                return await fn(req)
            except ToolError:
                raise
            except RuntimeError as e:
                msg = str(e)
                if "Action cap" in msg:
                    return f"Error: {msg}"
                raise ToolError(f"Browser action failed: {e}") from e
            except Exception as e:
                raise ToolError(f"Browser action '{action}' failed: {e}") from e

        return wrapped

    return deco


@_wrap_action_errors("close")
async def _handle_close(req: BrowseRequest) -> str:
    if not session.is_active():
        return "Browser already closed."
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired."
    if not await cb("Close browser now?", "Task done — close browser now?"):
        return "Browser left open."
    await session.close_session()
    return "Browser closed."


@_wrap_action_errors("navigate")
async def _handle_navigate(req: BrowseRequest) -> str:
    if not req.url:
        return "Error: 'url' is required for navigate action."
    page = await _page_or_error()
    await page.goto(req.url, wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(req.delay_ms / 1000.0)
    result = await _snapshot_text(page, req)
    if not session.keep_open():
        result += (
            "\n(When the task is finished, "
            "call browse(action='close') to close the browser.)"
        )
    return result


@_wrap_action_errors("back")
async def _handle_back(req: BrowseRequest) -> str:
    page = await _page_or_error()
    await page.go_back(timeout=30_000)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_text(page, req)


@_wrap_action_errors("snapshot")
async def _handle_snapshot(req: BrowseRequest) -> str:
    page = await _page_or_error()
    return await _snapshot_text(page, req)


@_wrap_action_errors("screenshot")
async def _handle_screenshot(req: BrowseRequest) -> str:
    page = await _page_or_error()
    await page.wait_for_timeout(500)
    img_bytes = await page.screenshot(type="png", full_page=False)
    b64 = _b64_encode(img_bytes)
    return f"Screenshot taken ({len(img_bytes)} bytes):\n{b64[:200]}..."


@_wrap_action_errors("click")
async def _handle_click(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref:
        return (
            "Error: 'ref' is required for click/type. "
            "Get a snapshot first to find [ref=eN]."
        )
    locator = await _require_ref(page, req.ref)
    if locator is None:
        return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not req.fast:
        await session.pace()
        await humanize.human_click(page, req.ref)
    else:
        await locator.click()
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("type")
async def _handle_type(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref:
        return (
            "Error: 'ref' is required for click/type. "
            "Get a snapshot first to find [ref=eN]."
        )
    locator = await _require_ref(page, req.ref)
    if locator is None:
        return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not req.text:
        return "Error: 'text' is required for type action."

    if not req.fast:
        await session.pace()
        await humanize.human_type(page, req.ref, req.text)
    else:
        await locator.fill(req.text)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


async def _require_ref(page: Any, ref: str) -> Any:
    try:
        locator = page.locator(f"aria-ref={ref}")
        await locator.wait_for(state="visible", timeout=5_000)
        return locator
    except Exception:
        return None


async def _confirm_or_block(page: Any, req: BrowseRequest, locator: Any) -> str | None:
    aria_role, aria_name = await _resolve_role_and_name(locator, req.ref)
    host = await _page_host(page)

    if not guardrails.needs_confirmation(req.action, aria_role, aria_name, host):
        return None

    detail = (
        f"Action: {req.action}\n"
        f"Host: {host}\n"
        f"Role: {aria_role}\n"
        f"Name: {aria_name}\n"
        f"Ref: {req.ref}"
    )
    if req.action == "type":
        detail += f"\nText length: {len(req.text)} chars"
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired. Action cancelled."
    if not await cb("Authorize Browser Action?", detail):
        return "Error: action cancelled."
    return None


async def _page_host(page: Any) -> str:
    try:
        from urllib.parse import urlparse

        location = await page.evaluate("document.location.href")
        return (urlparse(location).hostname or "").lower().removeprefix("www.")
    except Exception:
        logger.debug("Could not read page URL", exc_info=True)
        return ""


async def _snapshot_text(page: Any, req: BrowseRequest) -> str:
    snap = await page.aria_snapshot(mode="ai", depth=req.depth)
    output = snapshot.filter_compact(snap) if req.compact else snap
    if req.compact and req.depth == 0:
        lines = output.splitlines()
        output = f"{len(lines)} interactive elements:\n{output}"
    return output or "(empty page)"


async def _post_action_snapshot(page: Any, req: BrowseRequest) -> str:
    await page.wait_for_timeout(300)
    snap = await page.aria_snapshot(mode="ai", depth=req.depth)
    output = snapshot.filter_compact(snap) if req.compact else snap
    if req.compact and req.depth == 0:
        lines = output.splitlines()
        output = f"{len(lines)} interactive elements after {req.action}:\n{output}"
    return output or "(empty page)"


_ROLE_NAME_JS = """
el => {
    const implicitRole = (e) => {
        const tag = e.tagName.toLowerCase();
        if (tag === 'button') return 'button';
        if (tag === 'a' && e.hasAttribute('href')) return 'link';
        if (tag === 'select') return 'combobox';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'input') {
            const t = (e.getAttribute('type') || 'text').toLowerCase();
            if (t === 'password') return 'password';
            if (['checkbox', 'radio', 'button', 'submit', 'reset'].includes(t)) {
                return t === 'submit' || t === 'reset' || t === 'button' ? 'button' : t;
            }
            return 'textbox';
        }
        return '';
    };
    const role = el.getAttribute('role') || implicitRole(el) || '';
    const name = (
        el.getAttribute('aria-label')
        || (el.textContent || '').trim()
        || el.getAttribute('value')
        || el.getAttribute('placeholder')
        || el.getAttribute('name')
        || el.getAttribute('title')
        || (el.tagName.toLowerCase() === 'input'
            ? (el.getAttribute('type') || '') : '')
        || ''
    );
    return [role, name.slice(0, 200)];
}
"""


async def _resolve_role_and_name(locator: Any, ref: str) -> tuple[str, str]:
    try:
        role, name = await locator.evaluate(_ROLE_NAME_JS)
        return role or "", name or ""
    except Exception:
        logger.debug(f"Could not resolve role/name for {ref}", exc_info=True)
        return "", ""


def _b64_encode(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


_HANDLERS: dict[str, Callable[[BrowseRequest], Awaitable[str]]] = {
    "navigate": _handle_navigate,
    "back": _handle_back,
    "snapshot": _handle_snapshot,
    "screenshot": _handle_screenshot,
    "click": _handle_click,
    "type": _handle_type,
    "close": _handle_close,
}
