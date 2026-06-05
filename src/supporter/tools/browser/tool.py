from __future__ import annotations

from ...logger import logger
from .core import BrowseRequest
from .handlers import HANDLERS
from .recorder import _record_step

__all__ = ["browse"]


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

    Returns:
        Snapshot text, confirmation prompt, or error message.
    """
    logger.info(
        f"Tool: browse — action={action}, url={url!r}, ref={ref!r}, "
        f"depth={depth}, compact={compact}"
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
    )
    result = await HANDLERS[action](req)
    await _record_step(req, result)
    return result
