from __future__ import annotations

import asyncio
from typing import Any

from ...logger import logger
from ..base import ToolError
from . import guardrails, humanize, session, snapshot

_VALID_ACTIONS = frozenset(
    {
        "navigate",
        "snapshot",
        "click",
        "type",
        "screenshot",
        "back",
        "close",
    }
)


async def browse(
    action: str,
    url: str = "",
    ref: str = "",
    text: str = "",
    depth: int = 0,
    compact: bool = False,
    delay_ms: int = 100,
) -> str:
    """Browser automation tool using your Chrome profile and cookies.

    Navigate websites, read page content (accessibility snapshot), click
    elements, type text, and take screenshots. Run with your real logged-in
    session so you can operate ChatGPT, Claude, Gemini, Twitter/X, etc.

    Args:
        action: What to do. One of: navigate, snapshot, click, type,
            screenshot, back, close.
        url: URL to navigate to (for navigate action).
        ref: Element reference from a snapshot [ref=eN] to act on
            (for click/type).
        text: Text to type (for type action).
        depth: Accessibility tree depth for snapshot. 0 = full tree.
        compact: If True, return only interactive elements [ref=eN].
        delay_ms: Extra delay after action in ms. Default 100.

    Returns:
        Snapshot text, confirmation prompt, or error message.
    """
    logger.info(
        f"Tool: browse — action={action}, url={url!r}, ref={ref!r}, "
        f"depth={depth}, compact={compact}"
    )

    if action not in _VALID_ACTIONS:
        names = ", ".join(sorted(_VALID_ACTIONS))
        return f"Error: Unknown action '{action}'. Valid actions: {names}"

    if action == "close":
        if not session.is_active():
            return "Browser already closed."
        cb = guardrails.browse_confirmation_callback
        if cb is None:
            return "Error: browser confirmation not wired."
        if not await cb("Close browser now?", "Task done — close browser now?"):
            return "Browser left open."
        await session.close_session()
        return "Browser closed."

    try:
        _pws, _context, page = await session.get_session()
    except Exception as e:
        raise ToolError(f"Browser session failed: {e}") from e

    try:
        if action in ("navigate", "back", "screenshot"):
            result = await _handle_read_action(
                action, page, url, depth, compact, delay_ms
            )
            if action == "navigate" and not session.keep_open():
                result += (
                    "\n(When the task is finished, "
                    "call browse(action='close') to close the browser.)"
                )
            return result

        if not ref:
            return (
                "Error: 'ref' is required for click/type. "
                "Get a snapshot first to find [ref=eN]."
            )

        return await _handle_interactive_action(
            action, page, ref, text, depth, compact, delay_ms
        )
    except ToolError:
        raise
    except RuntimeError as e:
        msg = str(e)
        if "Action cap" in msg:
            return f"Error: {msg}"
        raise ToolError(f"Browser action failed: {e}") from e
    except Exception as e:
        raise ToolError(f"Browser action '{action}' failed: {e}") from e


async def _handle_read_action(
    action: str,
    page: Any,
    url: str,
    depth: int,
    compact: bool,
    delay_ms: int,
) -> str:
    if action == "navigate":
        if not url:
            return "Error: 'url' is required for navigate action."
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(delay_ms / 1000.0)

    if action == "back":
        await page.go_back(timeout=30_000)
        await asyncio.sleep(delay_ms / 1000.0)

    snap = await page.aria_snapshot(mode="ai", depth=depth)
    output = snapshot.filter_compact(snap) if compact else snap

    if compact and depth == 0:
        lines = output.splitlines()
        output = f"{len(lines)} interactive elements:\n{output}"

    if action == "screenshot":
        await page.wait_for_timeout(500)
        img_bytes = await page.screenshot(type="png", full_page=False)
        b64 = _b64_encode(img_bytes)
        prefix = b64[:200]
        output = f"Screenshot taken ({len(img_bytes)} bytes):\n{prefix}..."

    return output or "(empty page)"


async def _handle_interactive_action(
    action: str,
    page: Any,
    ref: str,
    text: str,
    depth: int,
    compact: bool,
    delay_ms: int,
) -> str:
    try:
        locator = page.locator(f"aria-ref={ref}")
        await locator.wait_for(state="visible", timeout=5_000)
    except Exception:
        return f"Error: ref {ref} not found, take a fresh snapshot"

    aria_role, aria_name = await _resolve_role_and_name(locator, ref)

    host = ""
    try:
        from urllib.parse import urlparse

        location = await page.evaluate("document.location.href")
        parsed = urlparse(location)
        host = (parsed.hostname or "").lower().removeprefix("www.")
    except Exception:
        logger.debug("Could not read page URL", exc_info=True)

    if guardrails.needs_confirmation(action, aria_role, aria_name, host):
        detail = (
            f"Action: {action}\n"
            f"Host: {host}\n"
            f"Role: {aria_role}\n"
            f"Name: {aria_name}\n"
            f"Ref: {ref}"
        )
        if action == "type":
            detail += f"\nText length: {len(text)} chars"
        cb = guardrails.browse_confirmation_callback
        if cb is None:
            return "Error: browser confirmation not wired. Action cancelled."
        if not await cb("Authorize Browser Action?", detail):
            return "Error: action cancelled."

    if action == "click":
        await session.pace()
        await humanize.human_click(page, ref)
        await asyncio.sleep(delay_ms / 1000.0)

    elif action == "type":
        if not text:
            return "Error: 'text' is required for type action."
        await session.pace()
        await humanize.human_type(page, ref, text)
        await asyncio.sleep(delay_ms / 1000.0)

    await page.wait_for_timeout(300)
    snap = await page.aria_snapshot(mode="ai", depth=depth)
    output = snapshot.filter_compact(snap) if compact else snap

    if compact and depth == 0:
        lines = output.splitlines()
        output = f"{len(lines)} interactive elements after {action}:\n{output}"

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
