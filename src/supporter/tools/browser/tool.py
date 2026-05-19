from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...logger import logger
from ..base import ToolError
from ..file_ops import validate_path
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
    key: str = ""
    value: str = ""
    selector: str = ""
    dx: int = 0
    dy: int = 0
    script: str = ""
    index: int = -1
    html: bool = False
    path: str = ""


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
        screenshot: capture the viewport (returns a byte/size summary).
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
        close: close the browser (asks for confirmation).

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
        fast: If True, skip humanized motion + pacing for this action (faster,
            less stealthy — use only on sites that don't fingerprint input).
            Hosts in guardrails.FAST_HOSTS run fast automatically even when
            this is False.
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
        key=key,
        value=value,
        selector=selector,
        dx=dx,
        dy=dy,
        script=script,
        index=index,
        html=html,
        path=path,
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
    if not await _effective_fast(page, req):
        await humanize.reading_pause()
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
    await page.go_back(timeout=30_000, wait_until="commit")
    await asyncio.sleep(req.delay_ms / 1000.0)
    if not await _effective_fast(page, req):
        await humanize.reading_pause()
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
    locator, err = await _resolve_target(page, req)
    if err is not None:
        return err

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page, req):
        await session.pace()
        await humanize.human_click(page, req.ref, locator=locator)
    else:
        await locator.click()
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("type")
async def _handle_type(req: BrowseRequest) -> str:
    page = await _page_or_error()
    locator, err = await _resolve_target(page, req)
    if err is not None:
        return err

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not req.text:
        return "Error: 'text' is required for type action."

    if not await _effective_fast(page, req):
        await session.pace()
        await humanize.human_type(page, req.ref, req.text, locator=locator)
    else:
        await locator.fill(req.text)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("forward")
async def _handle_forward(req: BrowseRequest) -> str:
    page = await _page_or_error()
    await page.go_forward(timeout=30_000, wait_until="commit")
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_text(page, req)


@_wrap_action_errors("hover")
async def _handle_hover(req: BrowseRequest) -> str:
    page = await _page_or_error()
    locator, err = await _resolve_target(page, req)
    if err is not None:
        return err

    if not await _effective_fast(page, req):
        await session.pace()
        await humanize.human_hover(page, req.ref, locator=locator)
    else:
        await locator.hover()
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("scroll")
async def _handle_scroll(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref and not (req.dx or req.dy):
        return "Error: scroll needs a 'ref' or non-zero 'dx'/'dy'."

    locator = None
    if req.ref:
        locator = await _require_ref(page, req.ref)
        if locator is None:
            return f"Error: ref {req.ref} not found, take a fresh snapshot"

    fast = await _effective_fast(page, req)
    if not fast:
        await session.pace()

    if locator is not None:
        await locator.scroll_into_view_if_needed()
    elif not fast:
        await humanize.human_scroll(page, req.dx, req.dy)
    else:
        await page.mouse.wheel(req.dx, req.dy)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("press")
async def _handle_press(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.key:
        return "Error: 'key' is required for press (e.g. 'Enter', 'Control+a')."

    locator = None
    if req.ref:
        locator = await _require_ref(page, req.ref)
        if locator is None:
            return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if locator is not None:
        await locator.focus()
    if not await _effective_fast(page, req):
        await session.pace()
        await humanize.human_press(page, req.key)
    else:
        await page.keyboard.press(req.key)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("select")
async def _handle_select(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref:
        return "Error: 'ref' is required for select. Get a snapshot first."
    if not req.value and not req.text:
        return "Error: select needs 'value' or 'text' (the option to choose)."
    locator = await _require_ref(page, req.ref)
    if locator is None:
        return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page, req):
        await session.pace()
    if req.value:
        await locator.select_option(value=req.value)
    else:
        await locator.select_option(label=req.text)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("wait")
async def _handle_wait(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if req.selector:
        await page.wait_for_selector(req.selector, timeout=30_000)
        return await _snapshot_text(page, req)
    await page.wait_for_timeout(req.delay_ms)
    return await _snapshot_text(page, req)


async def _session_parts() -> tuple[Any, Any, Any]:
    try:
        return await session.get_session()
    except Exception as e:
        raise ToolError(f"Browser session failed: {e}") from e


@_wrap_action_errors("tabs")
async def _handle_tabs(req: BrowseRequest) -> str:
    await _page_or_error()
    pages = session.list_pages()
    active = session.active_page()
    lines = []
    for i, p in enumerate(pages):
        marker = "*" if p is active else " "
        try:
            title = await p.title()
        except Exception:
            title = ""
        lines.append(f"{marker} [{i}] {title} — {p.url}")
    return "\n".join(lines) if lines else "(no open tabs)"


@_wrap_action_errors("tab")
async def _handle_tab(req: BrowseRequest) -> str:
    await _page_or_error()
    pages = session.list_pages()
    if not 0 <= req.index < len(pages):
        return f"Error: tab index {req.index} out of range (0..{len(pages) - 1})."
    target = pages[req.index]
    session.set_active(target)
    await target.bring_to_front()
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_text(target, req)


@_wrap_action_errors("newtab")
async def _handle_newtab(req: BrowseRequest) -> str:
    _pws, context, _page = await _session_parts()
    new_page = await context.new_page()
    session.set_active(new_page)
    await new_page.bring_to_front()
    if req.url:
        await new_page.goto(req.url, wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_text(new_page, req)


@_wrap_action_errors("closetab")
async def _handle_closetab(req: BrowseRequest) -> str:
    await _page_or_error()
    pages = session.list_pages()
    active = session.active_page()
    if req.index >= 0:
        if req.index >= len(pages):
            return f"Error: tab index {req.index} out of range (0..{len(pages) - 1})."
        target = pages[req.index]
    else:
        target = active
    await target.close()

    remaining = session.list_pages()
    if not remaining:
        return "Closed the last tab; no tabs remain open."
    if target is active:
        session.set_active(remaining[-1])
        await remaining[-1].bring_to_front()
    return await _snapshot_text(session.active_page(), req)


@_wrap_action_errors("extract")
async def _handle_extract(req: BrowseRequest) -> str:
    page = await _page_or_error()
    frame_sel = session.active_frame_selector()
    if req.ref:
        if frame_sel is not None:
            return "Error: inside a frame, extract needs a CSS 'selector', not a ref."
        locator = await _require_ref(page, req.ref)
        if locator is None:
            return f"Error: ref {req.ref} not found, take a fresh snapshot"
    elif req.selector:
        if frame_sel is not None:
            locator = page.frame_locator(frame_sel).locator(req.selector).first
        else:
            locator = page.locator(req.selector).first
    else:
        return "Error: extract needs a 'ref' or a CSS 'selector'."

    if req.html:
        content = await locator.inner_html()
    else:
        content = await locator.inner_text()
    return content or "(empty)"


@_wrap_action_errors("eval")
async def _handle_eval(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.script:
        return "Error: 'script' is required for eval."

    blocked = await _confirm_script(page, req.script)
    if blocked is not None:
        return blocked

    if not req.fast:
        await session.pace()
    result = await page.evaluate(req.script)
    rendered = _render_script_result(result)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return f"eval result: {rendered}"


async def _confirm_script(page: Any, script: str) -> str | None:
    host = await _page_host(page)
    logger.info(f"browse eval requested — host={host!r}, script_len={len(script)}")
    if not guardrails.needs_confirmation("eval", "", "", host):
        return None
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return "Error: browser confirmation not wired. Action cancelled."
    detail = f"Host: {host}\nScript length: {len(script)} chars\nRuns arbitrary JS."
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


def _validate_path_or_error(path: str) -> tuple[Any, str | None]:
    try:
        return validate_path(path), None
    except (ToolError, PermissionError) as e:
        return None, f"Error: {e}"


@_wrap_action_errors("upload")
async def _handle_upload(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref:
        return "Error: 'ref' is required for upload. Get a snapshot first."
    if not req.path:
        return "Error: 'path' is required for upload (the file to attach)."

    target, err = await asyncio.to_thread(_validate_path_or_error, req.path)
    if err is not None:
        return err
    if not target.exists():
        return f"Error: file not found: {target}"

    locator = await _require_ref(page, req.ref)
    if locator is None:
        return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not req.fast:
        await session.pace()
    await locator.set_input_files(str(target))
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("download")
async def _handle_download(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.ref:
        return "Error: 'ref' is required for download. Get a snapshot first."
    if not req.path:
        return "Error: 'path' is required for download (where to save the file)."

    dest, err = await asyncio.to_thread(_validate_path_or_error, req.path)
    if err is not None:
        return err

    locator = await _require_ref(page, req.ref)
    if locator is None:
        return f"Error: ref {req.ref} not found, take a fresh snapshot"

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not req.fast:
        await session.pace()
    async with page.expect_download(timeout=30_000) as info:
        await locator.click()
    download = await info.value

    final = dest / download.suggested_filename if dest.is_dir() else dest
    final, err = await asyncio.to_thread(_validate_path_or_error, str(final))
    if err is not None:
        return err
    await download.save_as(str(final))
    await asyncio.sleep(req.delay_ms / 1000.0)
    return f"Downloaded to {final}"


@_wrap_action_errors("cookies")
async def _handle_cookies(req: BrowseRequest) -> str:
    _pws, context, _page = await _session_parts()
    cookies = await context.cookies()
    if req.key:
        match = next((c for c in cookies if c.get("name") == req.key), None)
        logger.info(f"browse cookies get — name={req.key!r}, found={match is not None}")
        if match is None:
            return f"Error: no cookie named {req.key!r}."
        blocked = await _confirm_always(
            "Reveal cookie value?",
            f"Cookie {req.key!r} @ {match.get('domain', '')} — returns its value.",
        )
        if blocked is not None:
            return blocked
        return f"{req.key}={match.get('value', '')}"
    logger.info(f"browse cookies list — count={len(cookies)}")
    if not cookies:
        return "(no cookies)"
    lines = [f"{c.get('name', '')} @ {c.get('domain', '')}" for c in cookies]
    return f"{len(cookies)} cookies:\n" + "\n".join(lines)


@_wrap_action_errors("storage")
async def _handle_storage(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.key:
        keys = await page.evaluate("Object.keys(window.localStorage)")
        logger.info(f"browse storage list — count={len(keys)}")
        if not keys:
            return "(empty localStorage)"
        return f"{len(keys)} localStorage keys:\n" + "\n".join(keys)
    if req.value:
        blocked = await _confirm_always(
            "Write localStorage?",
            f"Set localStorage[{req.key!r}] to a {len(req.value)}-char value.",
        )
        if blocked is not None:
            return blocked
        await page.evaluate(
            "([k, v]) => window.localStorage.setItem(k, v)", [req.key, req.value]
        )
        logger.info(f"browse storage set — key={req.key!r}, len={len(req.value)}")
        return f"Set localStorage[{req.key!r}] ({len(req.value)} chars)."
    blocked = await _confirm_always(
        "Reveal localStorage value?",
        f"Returns the value of localStorage[{req.key!r}].",
    )
    if blocked is not None:
        return blocked
    value = await page.evaluate("k => window.localStorage.getItem(k)", req.key)
    logger.info(f"browse storage get — key={req.key!r}, found={value is not None}")
    if value is None:
        return f"Error: no localStorage key {req.key!r}."
    return f"{req.key}={value}"


@_wrap_action_errors("frame")
async def _handle_frame(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if not req.selector:
        session.set_frame(None)
        return "Frame cleared (acting on the top page now)."

    frame_body = page.frame_locator(req.selector).locator("body")
    try:
        snap = await frame_body.aria_snapshot()
    except Exception as e:
        return f"Error: no iframe matched {req.selector!r} ({e})."
    session.set_frame(req.selector)
    await asyncio.sleep(req.delay_ms / 1000.0)
    body = snap or "(empty frame)"
    return f"Drilled into frame {req.selector!r}:\n{body}"


@_wrap_action_errors("waitnetwork")
async def _handle_waitnetwork(req: BrowseRequest) -> str:
    page = await _page_or_error()
    await page.wait_for_load_state("networkidle", timeout=30_000)
    return await _snapshot_text(page, req)


def _render_script_result(result: Any) -> str:
    import json

    try:
        text = json.dumps(result, default=str)
    except Exception:
        text = str(result)
    return text if len(text) <= 2000 else text[:2000] + "...(truncated)"


async def _effective_fast(page: Any, req: BrowseRequest) -> bool:
    if req.fast:
        return True
    return guardrails.host_is_fast(await _page_host(page))


async def _require_ref(page: Any, ref: str) -> Any:
    try:
        locator = page.locator(f"aria-ref={ref}")
        await locator.wait_for(state="visible", timeout=5_000)
        return locator
    except Exception:
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
    "forward": _handle_forward,
    "snapshot": _handle_snapshot,
    "screenshot": _handle_screenshot,
    "click": _handle_click,
    "type": _handle_type,
    "hover": _handle_hover,
    "scroll": _handle_scroll,
    "press": _handle_press,
    "select": _handle_select,
    "wait": _handle_wait,
    "tabs": _handle_tabs,
    "tab": _handle_tab,
    "newtab": _handle_newtab,
    "closetab": _handle_closetab,
    "extract": _handle_extract,
    "eval": _handle_eval,
    "upload": _handle_upload,
    "download": _handle_download,
    "cookies": _handle_cookies,
    "storage": _handle_storage,
    "frame": _handle_frame,
    "waitnetwork": _handle_waitnetwork,
    "close": _handle_close,
}
