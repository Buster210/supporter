from __future__ import annotations

import time

from ...config import config
from ...logger import logger
from . import guardrails, session
from .core import BrowseRequest, _page_host
from .handlers import HANDLERS
from .recorder import _record_step
from .support import _attach_fullpage_image, _attach_viewport_image

__all__ = ["browse", "session"]

_IMAGE_ACTIONS = frozenset(
    {
        "navigate",
        "back",
        "forward",
        "snapshot",
        "click",
        "type",
        "scroll",
        "press",
        "select",
        "hover",
        "tab",
        "newtab",
        "closetab",
        "wait",
        "waitnetwork",
        "frame",
        "upload",
        "download",
        "solve_cloudflare",
    }
)

_EXTRACT_IMAGE_ACTIONS = frozenset({"read", "links"})


async def browse(
    action: str,
    url: str = "",
    ref: str = "",
    text: str = "",
    depth: int = 0,
    compact: bool = False,
    delay_ms: int = 100,
    *,
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
    variable: str = "",
    full_page: bool = False,
    brief: bool = False,
) -> str:
    """Browser automation -- and your PRIMARY tool for web search and research.

    Drive a real Chrome session with your profile and cookies: navigate sites,
    read page content (accessibility snapshot), click, type, and screenshot.
    For any web search, lookup, or research, prefer this over google_search --
    it reaches real pages with full depth and reliability. Runs with your
    logged-in session so you can also operate ChatGPT, Claude, Gemini,
    Twitter/X, etc.

    Actions and the params each uses:
        navigate (url): load a URL.
        back: go back one history entry.
        forward: go forward one history entry.
        snapshot (depth, compact): return the accessibility tree. In a Live
            session this and the other state-changing actions (navigate, click,
            type, scroll, ...) also attach a viewport screenshot automatically,
            so you see the page by default — no separate screenshot call needed.
        read (url | selector, full_page): READ a page as clean article text.
            This is your PRIMARY reading action for research: it extracts the
            main content as markdown plus metadata (title/author/date/site) and
            the in-content links, dropping nav/ads/boilerplate. Pass url to
            navigate first (or SEVERAL urls separated by spaces/newlines to read
            them in one batch); omit url to read the current page. selector
            scopes to a container; full_page=True auto-scrolls to load lazy
            content first. Prefer this over snapshot/extract for reading.
        links (url | selector): list the in-content outbound links (text ->
            absolute URL) so you can decide what to open and follow next.
        diff (depth): return only the lines that changed since the last
            snapshot of this page (+ added, - removed); far fewer tokens than
            a full re-snapshot after a small page change.
        screenshot: capture the viewport and save it as a PNG artifact under the
            project (and show it to the model in a Live session). State-changing
            actions already attach a viewport image automatically; use this when
            you specifically want a saved PNG file.
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
        close: ORCHESTRATOR-ONLY — not available via browse for sub-agents.
            The orchestrator uses browser_supervise for browser teardown.
        closenow: ORCHESTRATOR-ONLY — not available via browse for sub-agents.
            The orchestrator uses browser_supervise for immediate teardown.
        solve_cloudflare: attempt to solve a Cloudflare Turnstile challenge on
            the current page. Call this when a snapshot shows
            "[Turnstile detected]". May fail on invisible/managed challenges
            (no checkbox) — returns "manual solve needed" in that case.

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
        variable: While recording a task (start_task), tag this step's input as
            a named template variable so replay_playbook can override its value
            per run, e.g. type(..., text="alice", variable="username").
        full_page: For read, auto-scroll the page to load lazy/infinite-scroll
            content before extracting. Default False.
        brief: If True, skip the lean-perception diff/snapshot pairing after
            state-changing actions (screenshot still attaches). Use in hot loops.

    Returns:
        Snapshot text, confirmation prompt, or error message.
    """
    logger.info(
        f"Tool: browse — action={action}, url={url!r}, ref={ref!r}, "
        f"depth={depth}, compact={compact}"
    )

    if action in {"close", "closenow"}:
        return (
            "Error: close/closenow actions are orchestrator-only. Use "
            "browser_supervise to tear down the browser session."
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
        variable=variable,
        full_page=full_page,
        brief=brief,
    )
    _t0 = time.perf_counter()
    result = await HANDLERS[action](req)
    await _record_step(req, result)

    from .support import _last_confirmation_needed

    success = not (isinstance(result, str) and result.startswith("Error"))
    if (
        success
        and action not in guardrails._ALWAYS_CONFIRM_ACTIONS
        and not _last_confirmation_needed.get()
    ):
        page = session.active_page()
        if page is not None:
            host = await _page_host(page)
            await guardrails.record_clean_interaction(host)

    if action in _EXTRACT_IMAGE_ACTIONS and not (
        isinstance(result, str) and result.startswith("Error")
    ):
        page = session.active_page()
        if page is not None:
            await _attach_fullpage_image(page)
    elif action in _IMAGE_ACTIONS and not (
        isinstance(result, str) and result.startswith("Error")
    ):
        page = session.active_page()
        if page is not None:
            await _attach_viewport_image(page)
            if not brief and action not in {
                "navigate",
                "newtab",
                "back",
                "forward",
                "upload",
                "download",
                "frame",
                "closetab",
                "tab",
                "wait",
                "waitnetwork",
                "solve_cloudflare",
                "snapshot",
            }:
                from .snapshot import (
                    clean_snapshot,
                    diff_snapshot,
                    filter_interactive,
                    has_baseline,
                    remember_snapshot,
                )
                from .support import _page_baseline_key, _page_key

                bkey = _page_baseline_key(page)
                page_url = _page_key(page)
                raw_snap = await page.aria_snapshot(mode="ai")
                cleaned = clean_snapshot(raw_snap, page_url)
                assert isinstance(result, str)

                prefix: str = ""
                if has_baseline(bkey):
                    diff_text = diff_snapshot(bkey, cleaned)
                    diff_lines = [
                        line
                        for line in diff_text.splitlines()
                        if line.startswith(("+", "-"))
                    ]
                    if len(diff_lines) <= config.browser_diff_threshold:
                        added = sum(1 for line in diff_lines if line.startswith("+"))
                        removed = sum(1 for line in diff_lines if line.startswith("-"))
                        prefix = f"[diff: +{added}/-{removed} lines]\n{diff_text}"
                    else:
                        compact_snap = filter_interactive(cleaned)
                        prefix = (
                            f"[full snapshot:"
                            f" {len(compact_snap.splitlines())}"
                            f" interactive elements]\n{compact_snap}"
                        )
                else:
                    compact_snap = filter_interactive(cleaned)
                    remember_snapshot(bkey, cleaned)
                    prefix = (
                        f"[full snapshot:"
                        f" {len(compact_snap.splitlines())}"
                        f" interactive elements]\n{compact_snap}"
                    )
                if prefix:
                    result = f"{prefix}\n\n{result}"

    if action in {"navigate", "newtab"} and not (
        isinstance(result, str) and result.startswith("Error")
    ):
        from .playbook_store import find_cookbook_hints

        current_url = req.url or (
            session.active_page().url if session.active_page() else ""
        )
        if current_url:
            hints = find_cookbook_hints(current_url)
            if hints:
                result = result + "\n" + "\n".join(hints)

    _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
    logger.debug(f"browse action={action} elapsed_ms={_elapsed_ms:.1f}")
    return result
