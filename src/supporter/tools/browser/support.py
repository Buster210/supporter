from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any
from weakref import WeakKeyDictionary

from ...config import config
from ...logger import logger
from ..base import ToolError
from ..file_ops import validate_path
from . import guardrails, humanize, session, snapshot
from .core import BrowseRequest, _page_host

__all__ = [
    "_EVAL_DETAIL_MAX",
    "_NAV_MAX_ATTEMPTS",
    "_PAGE_IDS",
    "_PAGE_ID_SEQ",
    "_REF_VISIBLE_TIMEOUT_MS",
    "_ROLE_NAME_JS",
    "_SETTLE_TIMEOUT_MS",
    "_capture",
    "_confirm_always",
    "_confirm_or_block",
    "_confirm_script",
    "_diff_header",
    "_diff_text",
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
_SETTLE_TIMEOUT_MS = 2000
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
