from __future__ import annotations

import contextlib
import contextvars
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any
from weakref import WeakKeyDictionary

from ...config import config
from ...logger import logger
from ...recovery_metrics import record_re_snapshot_survived
from ..base import ToolError
from ..file_ops import validate_path
from . import guardrails, humanize, session, snapshot
from .core import BrowseRequest, _page_host

# WI-3: Track whether last _confirm_or_block needed confirmation
_last_confirmation_needed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_last_confirmation_needed", default=False
)

__all__ = [
    "_EVAL_DETAIL_MAX",
    "_NAV_MAX_ATTEMPTS",
    "_PAGE_IDS",
    "_PAGE_ID_SEQ",
    "_REF_VISIBLE_TIMEOUT_MS",
    "_ROLE_NAME_JS",
    "_SETTLE_TIMEOUT_MS",
    "_attach_viewport_image",
    "_capture",
    "_confirm_always",
    "_confirm_or_block",
    "_confirm_script",
    "_diff_header",
    "_diff_text",
    "_dom_click",
    "_dom_select",
    "_dom_set_value",
    "_effective_fast",
    "_is_transient_nav_error",
    "_live_refs_snapshot",
    "_navigate_with_retry",
    "_page_baseline_key",
    "_page_key",
    "_page_or_error",
    "_post_action_snapshot",
    "_record_locator",
    "_render_script_result",
    "_render_snapshot",
    "_require_ref",
    "_resolve_role_and_name",
    "_resolve_target",
    "_session_parts",
    "_snapshot_full",
    "_snapshot_text",
    "_stale_ref_snapshot",
    "_validate_path_or_error",
    "_wrap_action_errors",
    "config",
]

_PAGE_IDS: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
_PAGE_ID_SEQ = 0

_EVAL_DETAIL_MAX = 500
_REF_VISIBLE_TIMEOUT_MS = 8_000
# WHY: networkidle returns ~500ms after the network goes quiet, so this cap is
# only paid on pages that never idle (beacons/websockets/polling) — where a
# longer wait never improves the post-action snapshot. Capped low to avoid
# burning seconds per action; genuine slow settles use the `waitnetwork` action.
_SETTLE_TIMEOUT_MS = 1000
_NAV_MAX_ATTEMPTS = 3

# Chromium net errors safe to retry: timeouts and connection/DNS flaps. Errors
# that signal a permanent verdict (ERR_BLOCKED_BY_*, ERR_ABORTED, invalid URL)
# are deliberately absent so a blocked or malformed navigation fails fast.
_TRANSIENT_NAV_ERRORS = (
    "err_connection_reset",
    "err_connection_closed",
    "err_connection_refused",
    "err_connection_failed",
    "err_connection_timed_out",
    "err_name_not_resolved",
    "err_internet_disconnected",
    "err_network_changed",
    "err_address_unreachable",
    "err_socket_not_connected",
    "err_empty_response",
    "err_timed_out",
)


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


def _is_transient_nav_error(exc: BaseException) -> bool:
    """A navigation failure worth retrying: a Playwright timeout or a net flap."""
    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    if isinstance(exc, PlaywrightTimeoutError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_NAV_ERRORS)


async def _navigate_with_retry(page: Any, action: Callable[[], Awaitable[Any]]) -> None:
    """Run a navigation, retrying only transient failures with a stealthy backoff.

    Permanent failures (bad URL, blocked, HTTP errors) raise on the first attempt;
    transient ones retry up to ``_NAV_MAX_ATTEMPTS`` and then surface the last error.
    Backoff uses humanize jitter so retries keep the human-like timing profile.
    """
    for attempt in range(1, _NAV_MAX_ATTEMPTS + 1):
        try:
            await action()
            return
        except Exception as exc:
            if not _is_transient_nav_error(exc) or attempt == _NAV_MAX_ATTEMPTS:
                raise
            logger.debug(
                f"transient navigation error "
                f"(attempt {attempt}/{_NAV_MAX_ATTEMPTS}): {exc}"
            )
            await page.wait_for_timeout(humanize.jitter_ms(0.4, 0.4, 0.25, 1.0))


async def _require_ref(page: Any, ref: str) -> Any:
    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    try:
        locator = page.locator(f"aria-ref={ref}")
        await locator.wait_for(state="visible", timeout=_REF_VISIBLE_TIMEOUT_MS)
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
        return None, await _stale_ref_snapshot(page, req, req.ref)
    return locator, None


async def _stale_ref_snapshot(page: Any, req: BrowseRequest, ref: str) -> str:
    """Auto re-snapshot on a stale ref so fresh aria refs arrive in one turn.

    The model never has to ask for a new snapshot: when an element ref no longer
    resolves, capture a full snapshot and return it inline behind a short note.
    """
    snap = await _snapshot_full(page, req, label=" (re-snapshot)")
    record_re_snapshot_survived()
    return (
        f"Error: ref {ref} is stale; page auto re-snapshotted. "
        f"Use the fresh refs below:\n{snap}"
    )


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
    # WI-3: Reset confirmation flag each invocation
    _last_confirmation_needed.set(False)

    if locator is not None:
        aria_role, aria_name = await _resolve_role_and_name(locator, req.ref)
    else:
        aria_role, aria_name = "", ""
    host = await _page_host(page)

    if not guardrails.needs_confirmation(req.action, aria_role, aria_name, host):
        return None

    # WI-3: Mark that confirmation was needed (for trust promotion filtering)
    _last_confirmation_needed.set(True)

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


async def _effective_fast(page: Any) -> bool:
    # The debug overlay only paints a visible cursor trail along the humanized
    # movement path; fast mode would replace it with an instant click and show
    # nothing. When the overlay is on, force the visible path on every host.
    if config.browser_debug_overlay:
        return False
    return guardrails.host_is_fast(await _page_host(page))


def _render_snapshot(snap: str, req: BrowseRequest, label: str, *, cleaned: str) -> str:
    if req.compact:
        output = snapshot.filter_interactive(snap)
        if output:
            count = len(output.splitlines())
            return f"{count} interactive elements{label}:\n{output}"
    else:
        output = cleaned
    return output or "(empty page)"


def _page_key(page: Any) -> str:
    try:
        return page.url or ""
    except Exception:
        return ""


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
    _snap_t0 = time.perf_counter()
    snap = await page.aria_snapshot(mode="ai", depth=req.depth)
    cleaned = snapshot.clean_snapshot(snap, page_url)
    _snap_ms = (time.perf_counter() - _snap_t0) * 1000.0
    logger.debug(f"browser snapshot action={req.action} elapsed_ms={_snap_ms:.1f}")
    if force_full or req.compact or not snapshot.has_baseline(bkey):
        result = _render_snapshot(snap, req, label, cleaned=cleaned)
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
    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    await page.wait_for_timeout(humanize.jitter_ms(0.3, 0.3, 0.15, 0.6))
    with contextlib.suppress(PlaywrightTimeoutError):
        await page.wait_for_load_state("networkidle", timeout=_SETTLE_TIMEOUT_MS)
    return await _capture(page, req, force_full=False, label=f" after {req.action}")


async def _attach_viewport_image(page: Any) -> None:
    """Push a viewport screenshot to the model via the image sink, if wired.

    In a Live session ``guardrails.browse_image_sink`` is registered, so the
    model sees the page alongside the a11y snapshot. In non-live sessions the
    sink is ``None`` and this is a no-op (no AFC image path exists there).
    Best-effort: never raises, so it can't break a tool result.
    """
    sink = guardrails.browse_image_sink
    if sink is None:
        return
    try:
        img_bytes = await page.screenshot(type="png", full_page=False)
        await sink(img_bytes, "image/png")
    except Exception:
        logger.debug("auto image attach failed", exc_info=True)


async def _attach_fullpage_image(page: Any) -> None:
    sink = guardrails.browse_image_sink
    if sink is None:
        return
    try:
        max_height_px = config.browse_fullpage_shot_max_px
        clip = None
        if max_height_px > 0:
            page_height = await page.evaluate("() => document.body.scrollHeight")
            if isinstance(page_height, (int, float)) and page_height > max_height_px:
                page_width = await page.evaluate("() => document.body.scrollWidth")
                clip = {"x": 0, "y": 0, "width": page_width, "height": max_height_px}
        if clip is not None:
            img_bytes = await page.screenshot(type="png", clip=clip)
        else:
            img_bytes = await page.screenshot(type="png", full_page=True)
        await sink(img_bytes, "image/png")
    except Exception:
        logger.debug("auto full-page image attach failed", exc_info=True)


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


_DOM_CLICK_JS = "(el) => { el.click(); return 'clicked'; }"

_DOM_SET_VALUE_JS = """
(el, text) => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
        // Use the prototype's native value setter matching the element type, so
        // React/Vue controlled inputs observe the change (a plain el.value = ...
        // is swallowed by their value trackers). Wrong-prototype setters raise
        // "Illegal invocation", so the proto MUST match the tag.
        const proto = tag === 'textarea'
            ? window.HTMLTextAreaElement.prototype
            : window.HTMLInputElement.prototype;
        const desc = Object.getOwnPropertyDescriptor(proto, 'value');
        if (desc && desc.set) {
            desc.set.call(el, text);
        } else {
            el.value = text;
        }
    } else if (el.isContentEditable || el.contentEditable === 'true') {
        el.textContent = text;
    } else {
        throw new Error('not an input/textarea/contenteditable: ' + tag);
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return 'set';
}
"""

_DOM_SELECT_JS = """
(el, [value, label]) => {
    const options = Array.from(el.options);
    let target = null;
    if (label) {
        target = options.find(o => o.text.trim() === label.trim());
    }
    if (!target && value) {
        target = options.find(o => o.value === value);
    }
    if (!target) {
        const v = value || label;
        throw new Error('option not found: ' + v);
    }
    el.value = target.value;
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return target.value;
}
"""


async def _dom_click(locator: Any) -> None:
    """Click an element via direct DOM dispatch (fast trusted path)."""
    await locator.evaluate(_DOM_CLICK_JS)


async def _dom_set_value(locator: Any, text: str) -> None:
    """Set a value on an input/textarea/contenteditable via direct DOM (fast path).

    Covers input, textarea, and contenteditable elements. Dispatches bubbling
    ``input`` and ``change`` events so framework listeners see the update.
    Raises if the element is not a recognizable input target."""
    result = await locator.evaluate(_DOM_SET_VALUE_JS, text)
    if result != "set":
        raise RuntimeError(f"unexpected DOM set result: {result!r}")


async def _dom_select(locator: Any, *, value: str = "", label: str = "") -> str:
    """Select an option via direct DOM (fast path).

    Matches by visible label first (when ``label`` is given), then by value
    attribute. Returns the resolved value. Raises if no matching option exists.

    Playwright ``evaluate`` accepts a single ``arg``; pass ``[value, label]`` as
    one array and destructure it in the page function.
    """
    return str(await locator.evaluate(_DOM_SELECT_JS, [value, label]))


def _render_script_result(result: Any) -> str:
    # ponytail: cap from config (BROWSE_EVAL_CHARS_CAP)
    char_cap = config.browse_eval_chars_cap
    try:
        text = json.dumps(result, default=str)
    except Exception:
        text = str(result)
    if len(text) <= char_cap:
        return text
    omitted = len(text) - char_cap
    return text[:char_cap] + f"…(truncated: {omitted} more chars)"


def _validate_path_or_error(path: str) -> tuple[Any, str | None]:
    try:
        return validate_path(path), None
    except (ToolError, PermissionError) as e:
        return None, f"Error: {e}"
