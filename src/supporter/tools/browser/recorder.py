from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ...logger import logger
from .core import BrowseRequest, _page_host
from .playbook_store import _RECORD_PARAMS, _REF_RESOLVABLE_ACTIONS, Step, build_step

__all__ = [
    "RECORDABLE_ACTIONS",
    "_ACTIVE",
    "_record_locator",
    "_record_step",
    "discard",
    "finish",
    "is_recording",
    "record",
    "start",
]


RECORDABLE_ACTIONS = frozenset(
    {
        "navigate",
        "back",
        "forward",
        "newtab",
        "frame",
        "click",
        "type",
        "select",
        "press",
        "scroll",
        "hover",
        "extract",
        "snapshot",
        "wait",
        "waitnetwork",
    }
)


@dataclass
class _ActiveTask:
    goal: str
    host: str = ""  # Phase 5: captured at start
    steps: list[Step] = field(default_factory=list)
    failed: bool = False
    variables: list[str] = field(default_factory=list)  # Phase 3: template variables


_ACTIVE: _ActiveTask | None = None


def _record_locator(page: Any, req: BrowseRequest) -> Any:
    from .session import active_frame_selector

    frame_sel = active_frame_selector()
    if frame_sel is not None:
        if not req.selector:
            return None
        return page.frame_locator(frame_sel).locator(req.selector).first
    if not req.ref:
        return None
    return page.locator(f"aria-ref={req.ref}")


def start(goal: str, host: str = "", variables: list[str] | None = None) -> str:
    global _ACTIVE
    goal = goal.strip()
    if not goal:
        return "Error: a task goal is required."
    _ACTIVE = _ActiveTask(goal=goal, host=host, variables=variables or [])
    return f"Recording task: {goal!r}. Browse normally, then call finish_task."


def discard() -> None:
    global _ACTIVE
    _ACTIVE = None


def is_recording() -> bool:
    return _ACTIVE is not None


def record(step: Step) -> None:
    if _ACTIVE is None:
        return
    if step.action not in RECORDABLE_ACTIONS:
        # Non-recordable actions are skipped, not fatal (Phase 0 hotfix)
        return
    if step.result_head.startswith("Error:"):
        _ACTIVE.failed = True
    _ACTIVE.steps.append(step)


async def finish(success: bool, host: str = "") -> str:
    global _ACTIVE
    if _ACTIVE is None:
        return "No task is being recorded."
    task = _ACTIVE
    _ACTIVE = None

    if not success:
        return f"Task {task.goal!r} ended (success=False); nothing saved."
    if task.failed:
        return f"Task {task.goal!r} had an error or unsafe step; nothing saved."
    if not task.steps:
        return f"Task {task.goal!r} recorded no reusable steps; nothing saved."
    # Phase 5: finish prefers stored host if not provided (host-at-start capture)
    final_host = task.host or host
    if not final_host:
        return f"Task {task.goal!r}: could not resolve a host; nothing saved."

    # Phase 3: template variables = those declared at start + any annotated
    # on a recorded step, so replay can validate overrides against the real set.
    variables = list(task.variables)
    for step in task.steps:
        for raw in step.variable.split(","):
            var = raw.strip()
            if var and var not in variables:
                variables.append(var)

    from .playbook_store import Playbook, save_playbook

    playbook = Playbook(
        host=final_host,
        goal=task.goal,
        created_ts=time.time(),
        steps=task.steps,
        variables=variables,
    )
    await save_playbook(playbook)
    stepped_count = len(task.steps)
    return f"Saved playbook for {task.goal!r} on {final_host} ({stepped_count} steps)."


async def _record_step(req: BrowseRequest, result: str) -> None:
    if not is_recording():
        return
    try:
        from .session import active_page
        from .support import _resolve_role_and_name

        page = active_page()
        if page is None:
            return
        host = await _page_host(page)
        role = name = ""
        # Phase 3: extract now participates in role/name capture
        if req.action in _REF_RESOLVABLE_ACTIONS and (req.ref or req.selector):
            locator = _record_locator(page, req)
            if locator is not None:
                role, name = await _resolve_role_and_name(locator, req.ref)
        params = {f: getattr(req, f) for f in _RECORD_PARAMS.get(req.action, ())}
        step = build_step(
            req.action,
            role=role,
            name=name,
            selector=req.selector,
            url_before=host,
            params=params,
            result=result,
            variable=req.variable,
        )
        record(step)
    except Exception:
        logger.debug("Failed to record task step", exc_info=True)
