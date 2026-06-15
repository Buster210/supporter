from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ...config import config
from ...logger import logger
from .. import resolved_project_root
from . import cloudflare, debug_overlay, guardrails, humanize, session, snapshot
from .core import BrowseRequest, _page_host
from .reader import _handle_links, _handle_read
from .support import (
    _confirm_always,
    _confirm_or_block,
    _confirm_script,
    _diff_text,
    _effective_fast,
    _navigate_with_retry,
    _page_baseline_key,
    _page_or_error,
    _post_action_snapshot,
    _render_script_result,
    _require_ref,
    _resolve_role_and_name,
    _resolve_target,
    _session_parts,
    _snapshot_full,
    _snapshot_text,
    _stale_ref_snapshot,
    _validate_path_or_error,
    _wrap_action_errors,
)

__all__ = [
    "config",
    "debug_overlay",
    "session",
]

_SCREENSHOT_SEQ = 0


async def _overlay_mark(page: Any, locator: Any, kind: str) -> None:
    """Draw the debug overlay at an element's center (fast-path; flag-gated)."""
    if not config.browser_debug_overlay:
        return
    try:
        box = await locator.bounding_box()
    except Exception:
        box = None
    if not box:
        return
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    if kind == "click":
        await debug_overlay.overlay_click(page, cx, cy)
    else:
        await debug_overlay.overlay_move(page, cx, cy)


@_wrap_action_errors("navigate")
async def _handle_navigate(req: BrowseRequest) -> str:
    if not req.url:
        return "Error: 'url' is required for navigate action."
    page = await _page_or_error()
    session.set_frame(None)
    await _navigate_with_retry(
        page,
        lambda: page.goto(req.url, wait_until="domcontentloaded", timeout=30_000),
    )
    await asyncio.sleep(req.delay_ms / 1000.0)
    if not await _effective_fast(page):
        await humanize.reading_pause(page)
    result, has_turnstile = await asyncio.gather(
        _snapshot_full(page, req),
        cloudflare.detect_turnstile_in_page(page),
    )
    if has_turnstile:
        result = "[Turnstile detected] — call solve_cloudflare to proceed.\n\n" + result
    return result


@_wrap_action_errors("back")
async def _handle_back(req: BrowseRequest) -> str:
    page = await _page_or_error()
    session.set_frame(None)
    await _navigate_with_retry(
        page, lambda: page.go_back(timeout=30_000, wait_until="commit")
    )
    await asyncio.sleep(req.delay_ms / 1000.0)
    if not await _effective_fast(page):
        await humanize.reading_pause(page)
    return await _snapshot_full(page, req)


async def _handle_forward(req: BrowseRequest) -> str:
    page = await _page_or_error()
    session.set_frame(None)
    await _navigate_with_retry(
        page, lambda: page.go_forward(timeout=30_000, wait_until="commit")
    )
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_full(page, req)


@_wrap_action_errors("wait")
async def _handle_wait(req: BrowseRequest) -> str:
    page = await _page_or_error()
    if req.selector:
        await page.wait_for_selector(req.selector, timeout=30_000)
        return await _snapshot_text(page, req)
    await page.wait_for_timeout(req.delay_ms)
    return await _snapshot_text(page, req)


@_wrap_action_errors("waitnetwork")
async def _handle_waitnetwork(req: BrowseRequest) -> str:
    page = await _page_or_error()
    await page.wait_for_load_state("networkidle", timeout=30_000)
    return await _snapshot_text(page, req)


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
    return await _snapshot_full(target, req)


@_wrap_action_errors("newtab")
async def _handle_newtab(req: BrowseRequest) -> str:
    _pws, context, _page = await _session_parts()
    async with session._get_alloc_lock():
        pages = session.list_pages()
        if len(pages) == 1 and session.is_blank(pages[0]):
            target = pages[0]
        else:
            target = await context.new_page()
        session.set_active(target)
    await target.bring_to_front()
    if config.browser_debug_overlay:
        await debug_overlay.inject_overlay(target)
    if req.url:
        await target.goto(req.url, wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _snapshot_full(target, req)


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
    closed_key = _page_baseline_key(target)
    await target.close()
    snapshot.forget_snapshot(closed_key)
    session.drop_page(target)

    remaining = session.list_pages()
    if not remaining:
        return "Closed the last tab; no tabs remain open."
    if target is active:
        session.set_active(remaining[-1])
        await remaining[-1].bring_to_front()
        return await _snapshot_full(session.active_page(), req)
    return await _snapshot_text(session.active_page(), req)


@_wrap_action_errors("snapshot")
async def _handle_snapshot(req: BrowseRequest) -> str:
    page = await _page_or_error()
    return await _snapshot_text(page, req)


@_wrap_action_errors("diff")
async def _handle_diff(req: BrowseRequest) -> str:
    page = await _page_or_error()
    return await _diff_text(page, req)


@_wrap_action_errors("screenshot")
async def _handle_screenshot(req: BrowseRequest) -> str:
    global _SCREENSHOT_SEQ
    page = await _page_or_error()
    await page.wait_for_timeout(humanize.jitter_ms(0.5, 0.3, 0.3, 0.9))
    img_bytes = await page.screenshot(type="png", full_page=False)

    stem = req.stamp.strip()
    if not stem:
        _SCREENSHOT_SEQ += 1
        stem = f"screenshot-{_SCREENSHOT_SEQ}"
    dest = resolved_project_root() / ".supporter" / "screenshots" / f"{stem}.png"
    target, err = await asyncio.to_thread(_validate_path_or_error, str(dest))
    if err is not None:
        return err

    def _write() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(img_bytes)

    await asyncio.to_thread(_write)

    vp = page.viewport_size or {}
    dims = f"{vp.get('width', '?')}x{vp.get('height', '?')}"

    sink = guardrails.browse_image_sink
    if sink is not None:
        try:
            await sink(img_bytes, "image/png")
        except Exception:
            logger.debug("Image sink failed", exc_info=True)

    return f"Screenshot saved to {target} ({dims}, {len(img_bytes)} bytes)."


@_wrap_action_errors("extract")
async def _handle_extract(req: BrowseRequest) -> str:
    page = await _page_or_error()
    frame_sel = session.active_frame_selector()
    if req.ref:
        if frame_sel is not None:
            return "Error: inside a frame, extract needs a CSS 'selector', not a ref."
        locator = await _require_ref(page, req.ref)
        if locator is None:
            return await _stale_ref_snapshot(page, req, req.ref)
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

    await session.pace()
    result = await page.evaluate(req.script)
    rendered = _render_script_result(result)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return f"eval result: {rendered}"


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


@_wrap_action_errors("click")
async def _handle_click(req: BrowseRequest) -> str:
    page = await _page_or_error()
    locator, err = await _resolve_target(page, req)
    if err is not None:
        return err

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page):
        await session.pace()
        await humanize.human_click(page, req.ref, locator=locator)
    else:
        await locator.click()
        await _overlay_mark(page, locator, "click")
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

    if not await _effective_fast(page):
        await session.pace()
        aria_role, aria_name = await _resolve_role_and_name(locator, req.ref)
        host = await _page_host(page)
        sensitive = guardrails.needs_confirmation("type", aria_role, aria_name, host)
        await humanize.human_type(
            page, req.ref, req.text, sensitive=sensitive, locator=locator
        )
    else:
        await locator.fill(req.text)
        await _overlay_mark(page, locator, "move")
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("hover")
async def _handle_hover(req: BrowseRequest) -> str:
    page = await _page_or_error()
    locator, err = await _resolve_target(page, req)
    if err is not None:
        return err

    if not await _effective_fast(page):
        await session.pace()
        await humanize.human_hover(page, req.ref, locator=locator)
    else:
        await locator.hover()
        await _overlay_mark(page, locator, "move")
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
            return await _stale_ref_snapshot(page, req, req.ref)

    fast = await _effective_fast(page)
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
            return await _stale_ref_snapshot(page, req, req.ref)

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if locator is not None:
        await locator.focus()
    if not await _effective_fast(page):
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
        return await _stale_ref_snapshot(page, req, req.ref)

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page):
        await session.pace()
    if req.value:
        await locator.select_option(value=req.value)
    else:
        await locator.select_option(label=req.text)
    await asyncio.sleep(req.delay_ms / 1000.0)
    return await _post_action_snapshot(page, req)


@_wrap_action_errors("status")
async def _handle_status(_req: BrowseRequest) -> str:
    import json as _json

    info = await session.session_status()
    return _json.dumps(info)


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


@_wrap_action_errors("closenow")
async def _handle_closenow(req: BrowseRequest) -> str:
    if not session.is_active():
        return "Browser already closed."
    await session.close_session(force=req.force)
    return "Browser closed."


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
        return await _stale_ref_snapshot(page, req, req.ref)

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page):
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
        return await _stale_ref_snapshot(page, req, req.ref)

    blocked = await _confirm_or_block(page, req, locator)
    if blocked is not None:
        return blocked

    if not await _effective_fast(page):
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


@_wrap_action_errors("solve_cloudflare")
async def _handle_solve_cloudflare(req: BrowseRequest) -> str:
    page = await _page_or_error()
    return await cloudflare.solve_cloudflare(page)


HANDLERS: dict[str, Callable[[BrowseRequest], Awaitable[str]]] = {
    "status": _handle_status,
    "navigate": _handle_navigate,
    "back": _handle_back,
    "forward": _handle_forward,
    "snapshot": _handle_snapshot,
    "read": _handle_read,
    "links": _handle_links,
    "diff": _handle_diff,
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
    "closenow": _handle_closenow,
    "solve_cloudflare": _handle_solve_cloudflare,
}
