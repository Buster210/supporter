from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from ...logger import logger
from ..base import ToolError
from ..file_ops import validate_path
from . import guardrails, humanize, session, snapshot
from .task import (
    _record_step,
    finish_task,
    query_playbook,
    replay_playbook,
    start_task,
)

__all__ = [
    "HANDLERS",
    "BrowseRequest",
    "browse",
    "finish_task",
    "query_playbook",
    "replay_playbook",
    "start_task",
]


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


async def _page_or_error() -> Any:
    try:
        _pws, _context, page = await session.get_session()
    except Exception as e:
        raise ToolError(f"Browser session failed: {e}") from e
    return page


async def _session_parts() -> tuple[Any, Any, Any]:
    try:
        return await session.get_session()
    except Exception as e:
        raise ToolError(f"Browser session failed: {e}") from e


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


async def _require_ref(page: Any, ref: str) -> Any:
    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    try:
        locator = page.locator(f"aria-ref={ref}")
        await locator.wait_for(state="visible", timeout=5_000)
        return locator
    except PlaywrightTimeoutError:
        return None


async def _resolve_target(page: Any, req: BrowseRequest) -> tuple[Any, str | None]:
    frame_sel = session.active_frame_selector()
    if frame_sel is not None:
        if not req.selector:
            return (
                None,
                "Error: inside a frame, click/type/hover needs a CSS "
                "'selector', not a ref.",
            )
        return page.frame_locator(frame_sel).locator(req.selector).first, None
    if not req.ref:
        return (
            None,
            "Error: 'ref' is required for click/type. "
            "Get a snapshot first to find [ref=eN].",
        )
    locator = await _require_ref(page, req.ref)
    if locator is None:
        return None, f"Error: ref {req.ref} not found, take a fresh snapshot"
    return locator, None


def _record_locator(page: Any, req: BrowseRequest) -> Any:
    frame_sel = session.active_frame_selector()
    if frame_sel is not None:
        if not req.selector:
            return None
        return page.frame_locator(frame_sel).locator(req.selector).first
    if not req.ref:
        return None
    return page.locator(f"aria-ref={req.ref}")


async def _confirm_or_block(page: Any, req: BrowseRequest, locator: Any) -> str | None:
    if locator is not None:
        aria_role, aria_name = await _resolve_role_and_name(locator, req.ref)
    else:
        aria_role, aria_name = "", ""
    host = await _page_host(page)

    if not guardrails.needs_confirmation(req.action, aria_role, aria_name, host):
        return None

    target = req.ref or f"frame selector {req.selector!r}"
    detail = (
        f"Action: {req.action}\n"
        f"Host: {host}\n"
        f"Role: {aria_role}\n"
        f"Name: {aria_name}\n"
        f"Ref: {target}"
    )
    if req.action == "type":
        detail += f"\nText length: {len(req.text)} chars"
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired. Action cancelled."
    if not await cb("Authorize Browser Action?", detail):
        return "Error: action cancelled."
    return None


_EVAL_DETAIL_MAX = 500


async def _confirm_script(page: Any, script: str) -> str | None:
    host = await _page_host(page)
    logger.info(f"browse eval requested — host={host!r}, script_len={len(script)}")
    if not guardrails.needs_confirmation("eval", "", "", host):
        return None
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired. Action cancelled."
    body = script[:_EVAL_DETAIL_MAX]
    if len(script) > _EVAL_DETAIL_MAX:
        body += f"\n…(+{len(script) - _EVAL_DETAIL_MAX} more chars)"
    detail = (
        f"Host: {host}\n"
        f"Script length: {len(script)} chars\n"
        f"Runs arbitrary JS.\n\n"
        f"Script:\n{body}"
    )
    if not await cb("Authorize JavaScript eval?", detail):
        return "Error: action cancelled."
    return None


async def _confirm_always(title: str, detail: str) -> str | None:
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired. Action cancelled."
    if not await cb(title, detail):
        return "Error: action cancelled."
    return None


async def _page_host(page: Any) -> str:
    try:
        location = await page.evaluate("document.location.href")
        return guardrails.host_from_url(location)
    except Exception:
        logger.debug("Could not read page URL", exc_info=True)
        return ""


async def _effective_fast(page: Any, req: BrowseRequest) -> bool:
    return guardrails.host_is_fast(await _page_host(page))


def _render_snapshot(snap: str, req: BrowseRequest, label: str, page_url: str) -> str:
    if req.compact:
        output = snapshot.filter_interactive(snap)
        if output:
            count = len(output.splitlines())
            return f"{count} interactive elements{label}:\n{output}"
    else:
        output = snapshot.clean_snapshot(snap, page_url)
    return output or "(empty page)"


def _page_key(page: Any) -> str:
    try:
        return page.url or ""
    except Exception:
        return ""


_PAGE_IDS: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
_PAGE_ID_SEQ = 0


def _page_baseline_key(page: Any) -> str:
    global _PAGE_ID_SEQ
    try:
        token = _PAGE_IDS.get(page)
        if token is None:
            _PAGE_ID_SEQ += 1
            token = f"pg{_PAGE_ID_SEQ}"
            _PAGE_IDS[page] = token
        return token
    except Exception:
        logger.debug("Could not resolve a page baseline key", exc_info=True)
        return ""


def _diff_header(key: str) -> str:
    tail = key.split("//", 1)[-1] if key else ""
    if len(tail) > 60:
        tail = tail[:57] + "..."
    return f"diff vs last snapshot ({tail}):" if tail else "diff vs last snapshot:"


async def _capture(
    page: Any, req: BrowseRequest, *, force_full: bool, label: str
) -> str:
    page_url = _page_key(page)
    bkey = _page_baseline_key(page)
    snap = await page.aria_snapshot(mode="ai", depth=req.depth)
    cleaned = snapshot.clean_snapshot(snap, page_url)
    if force_full or req.compact or not snapshot.has_baseline(bkey):
        result = _render_snapshot(snap, req, label, page_url)
        snapshot.remember_snapshot(bkey, cleaned)
        snapshot.log_snapshot(req.action, result)
        return result
    diff = snapshot.diff_snapshot(bkey, cleaned)
    result = diff if diff.startswith("(") else f"{_diff_header(page_url)}\n{diff}"
    snapshot.log_snapshot(f"diff {req.action}", result)
    return result


async def _snapshot_text(page: Any, req: BrowseRequest) -> str:
    return await _capture(page, req, force_full=False, label="")


async def _snapshot_full(page: Any, req: BrowseRequest, label: str = "") -> str:
    return await _capture(page, req, force_full=True, label=label)


async def _post_action_snapshot(page: Any, req: BrowseRequest) -> str:
    await page.wait_for_timeout(humanize.jitter_ms(0.3, 0.3, 0.15, 0.6))
    return await _capture(page, req, force_full=False, label=f" after {req.action}")


async def _diff_text(page: Any, req: BrowseRequest) -> str:
    snap = await page.aria_snapshot(mode="ai", depth=req.depth)
    cleaned = snapshot.clean_snapshot(snap, _page_key(page))
    result = snapshot.diff_snapshot(_page_baseline_key(page), cleaned)
    snapshot.log_snapshot(f"diff {req.action}", result)
    return result


async def _live_refs_snapshot(page: Any) -> str:
    snap = await page.aria_snapshot(mode="ai")
    return snapshot.clean_snapshot(snap, _page_key(page))


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


def _render_script_result(result: Any) -> str:
    try:
        text = json.dumps(result, default=str)
    except Exception:
        text = str(result)
    return text if len(text) <= 2000 else text[:2000] + "...(truncated)"


def _validate_path_or_error(path: str) -> tuple[Any, str | None]:
    try:
        return validate_path(path), None
    except (ToolError, PermissionError) as e:
        return None, f"Error: {e}"


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
    key: str = "",
    value: str = "",
    selector: str = "",
    dx: int = 0,
    dy: int = 0,
    script: str = "",
    index: int = -1,
    html: bool = False,
    path: str = "",
    stamp: str = "",
) -> str:
    """Browser automation tool using your Chrome profile and cookies.

    Navigate websites, read page content (accessibility snapshot), click
    elements, type text, and take screenshots. Runs with your real logged-in
    session so you can operate ChatGPT, Claude, Gemini, Twitter/X, etc.

    Actions and the params each uses:
        navigate (url): load a URL.
        back: go back one history entry.
        forward: go forward one history entry.
        snapshot (depth, compact): return the accessibility tree.
        diff (depth): return only the lines that changed since the last
            snapshot of this page (+ added, - removed); far fewer tokens than
            a full re-snapshot after a small page change.
        screenshot: capture the viewport, save it as a PNG artifact under the
            project, and (in a Live session) show the image to the model so it
            can see the page when the accessibility snapshot is insufficient.
        click (ref): click the element [ref=eN] from a snapshot.
        type (ref, text): type text into the element [ref=eN].
        hover (ref): move the cursor onto the element [ref=eN], no click.
        scroll (dx, dy | ref): scroll the page by (dx, dy) pixels, or scroll
            the element [ref=eN] into view.
        press (key [, ref]): press a key or chord (e.g. "Enter",
            "Control+a"); focuses [ref=eN] first if given.
        select (ref, value | text): choose a dropdown option by value, or by
            visible label via text.
        wait (selector | delay_ms): wait for a CSS selector to appear, or just
            wait delay_ms when no selector is given.
        tabs: list the open tabs with their index, title, and URL.
        tab (index): switch to the tab at this index (see tabs).
        newtab (url): open a new tab (optionally navigating to url) and make
            it active.
        closetab (index): close the tab at this index, or the active tab when
            index is omitted (-1); then activates the last remaining tab.
        extract (ref | selector, html): return the visible text of an element,
            or its innerHTML when html=True.
        eval (script): run JavaScript in the page and return its result. ALWAYS
            asks for confirmation — this is arbitrary code in your logged-in
            session.
        upload (ref, path): set the file at `path` on the file input [ref=eN].
            Path must be inside the project directory.
        download (ref, path): click [ref=eN] and save the triggered download to
            `path` (a file path or directory) inside the project directory.
        cookies (key): list the current cookies' names + domains; pass key=NAME
            to return that single cookie's value.
        storage (key, value): read localStorage[key]; pass value to set it.
            Omit key to list all localStorage keys (names only).
        frame (selector): drill into the iframe matched by CSS `selector` and
            return its content's accessibility tree. While drilled,
            click/type/hover/extract act INSIDE that frame and address elements
            by CSS `selector` (not ref — refs are page-scoped). Call with no
            selector to clear it and act on the top page again.
        waitnetwork: wait until network activity goes idle, then snapshot.
        close: proactively close the browser when the task is done. Leaves it
            open with no prompt if the user chose to keep it open; otherwise
            asks for confirmation first. Use this when YOU decide the task is
            finished, not because the user asked.
        closenow: close the browser immediately, no confirmation. Use ONLY when
            the user explicitly asks to close it.

    Args:
        action: What to do (see list above).
        url: URL to navigate to (navigate).
        ref: Element reference [ref=eN] from a snapshot (click/type/hover/
            scroll/press/select). When a frame is drilled, click/type/hover
            ignore ref and use selector instead.
        text: Text to type (type) or visible option label (select).
        depth: Accessibility tree depth for snapshot. 0 = full tree.
        compact: If True, return only interactive elements [ref=eN].
        delay_ms: Extra delay after action in ms. Default 100. For wait with no
            selector, how long to wait.
        fast: Deprecated as a per-call override and ignored off the allowlist.
            Raw, un-humanized input is used ONLY on hosts in
            guardrails.FAST_HOSTS; every other host is always humanized,
            regardless of this flag.
        key: Key or chord to press (press), e.g. "Enter", "Control+a"; or a
            cookie/localStorage key name (cookies/storage).
        value: Option value attribute to choose (select); or the value to set
            (storage).
        selector: CSS selector to wait for (wait), extract from (extract), the
            iframe to drill into (frame), or — while a frame is drilled — the
            in-frame element to click/type/hover.
        dx: Horizontal scroll delta in pixels (scroll).
        dy: Vertical scroll delta in pixels (scroll).
        script: JavaScript source to run (eval).
        index: Zero-based tab index (tab/closetab). -1 (default) means the
            active tab for closetab; tab requires an explicit index.
        html: If True, extract returns innerHTML instead of visible text.
        path: File path for upload (the file to attach) or download (where to
            save). Must resolve inside the project directory.
        stamp: Optional filename stem for a screenshot artifact. Empty uses an
            in-process counter.

    Returns:
        Snapshot text, confirmation prompt, or error message.
    """
    logger.info(
        f"Tool: browse — action={action}, url={url!r}, ref={ref!r}, "
        f"depth={depth}, compact={compact}, fast={fast}"
    )

    if action not in HANDLERS:
        names = ", ".join(sorted(HANDLERS))
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
        key=key,
        value=value,
        selector=selector,
        dx=dx,
        dy=dy,
        script=script,
        index=index,
        html=html,
        path=path,
        stamp=stamp,
    )
    result = await HANDLERS[action](req)
    await _record_step(req, result)
    return result


from .handlers import HANDLERS  # noqa: E402
