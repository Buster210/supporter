"""Orchestrator-only browser supervisor — observe and lifecycle only.

Dispatches to the same handlers as ``browse`` but enforces a strict action
whitelist so the orchestrator can never call a drive/mutate/exfil action.
"""

from __future__ import annotations

from ...logger import logger
from .core import BrowseRequest
from .handlers import HANDLERS

__all__ = ["browser_supervise"]

SUPERVISOR_ACTIONS = frozenset(
    {
        "status",
        "snapshot",
        "screenshot",
        "tabs",
        "closetab",
        "close",
        "closenow",
    }
)


async def browser_supervise(
    action: str,
    *,
    index: int = -1,
    force: bool = False,
    depth: int = 0,
    compact: bool = False,
    stamp: str = "",
) -> str:
    """Observe and manage the browser session (orchestrator-only).

    Whitelisted actions for recovery / lifecycle:
        status:   read-only session state (open, URL, tabs, idle, launching).
        snapshot: accessibility tree of the current page.
        screenshot: viewport capture saved as a PNG artifact.
        tabs:     list open tabs.
        closetab: close a tab by index (or the active tab when index=-1).
        close:    graceful teardown with confirmation.
        closenow: immediate teardown, no confirmation.
            When force=True, also ignores errors from a wedged session.

    Any action NOT in this list is hard-rejected. The full browser suite
    (navigate, click, type, eval, …) belongs exclusively to page-pilot
    via ``browse``.

    Args:
        action: Which supervisory action to run (see above).
        index:  Tab index for closetab; ignored by other actions.  -1 = active.
        force:  closenow only — swallow errors from a wedged/crashed session.
        depth:  Accessibility tree depth for snapshot. 0 = full tree.
        compact: If True, snapshot returns only interactive elements.
        stamp:  Filename stem for a screenshot artifact.

    Returns:
        Action result or error message.
    """
    logger.info(f"Tool: browser_supervise — action={action!r}, force={force}")

    if action not in SUPERVISOR_ACTIONS:
        names = ", ".join(sorted(SUPERVISOR_ACTIONS))
        return (
            f"Error: action {action!r} is not permitted for browser_supervise. "
            f"Allowed actions: {names}"
        )

    req = BrowseRequest(
        action=action,
        index=index,
        force=force,
        depth=depth,
        compact=compact,
        stamp=stamp,
    )
    return await HANDLERS[action](req)
