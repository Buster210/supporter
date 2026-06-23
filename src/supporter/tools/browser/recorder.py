from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ...logger import logger
from .core import BrowseRequest, _page_host
from .playbook_store import (
    _RECORD_PARAMS,
    _REF_RESOLVABLE_ACTIONS,
    Playbook,
    Step,
    build_step,
)

__all__ = [
    "RECORDABLE_ACTIONS",
    "_ACTIVE",
    "_record_locator",
    "_record_step",
    "discard",
    "discard_all",
    "finish",
    "finish_repair",
    "get_repair_context",
    "is_recording",
    "is_repair_recording",
    "record",
    "start",
    "start_repair",
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


# Per-agent active task dict (lazy import from session to avoid cycles)
_ACTIVE: dict[str, _ActiveTask] = {}

# WI-2: Auto-repair recording buffer (per-agent)
_repair_steps: dict[str, list[Step]] = {}
_repair_context: dict[str, dict[str, Any]] = {}


def _get_agent_id() -> str:
    """Helper to get current agent ID (lazy import to avoid cycle)."""
    try:
        from . import session

        return session.current_agent_id()
    except Exception:
        return "main"


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
    aid = _get_agent_id()
    _repair_steps.pop(aid, None)
    _repair_context.pop(aid, None)
    _ACTIVE[aid] = _ActiveTask(goal=goal, host=host, variables=variables or [])
    return f"Recording task: {goal!r}. Browse normally, then call finish_task."


def discard(aid: str | None = None) -> None:
    """Discard an agent's active task (the current agent when aid is None)."""
    global _ACTIVE
    if aid is None:
        aid = _get_agent_id()
    _ACTIVE.pop(aid, None)


def discard_all() -> None:
    """Discard all active tasks (called during full teardown)."""
    global _ACTIVE
    _ACTIVE.clear()
    _repair_steps.clear()
    _repair_context.clear()


def is_recording() -> bool:
    aid = _get_agent_id()
    return aid in _ACTIVE


# ── WI-2: Repair recording ───────────────────────────────────────────────────


def start_repair(playbook: Playbook, good_prefix: list[Step]) -> None:
    """Start repair recording with context about the original playbook."""
    aid = _get_agent_id()
    _repair_steps[aid] = []
    _repair_context[aid] = {
        "original_playbook": playbook,
        "good_prefix": good_prefix,
    }


def is_repair_recording() -> bool:
    aid = _get_agent_id()
    return aid in _repair_steps


def get_repair_context() -> dict[str, Any] | None:
    aid = _get_agent_id()
    return _repair_context.get(aid)


def finish_repair() -> tuple[list[Step], dict[str, Any] | None]:
    aid = _get_agent_id()
    steps = _repair_steps.pop(aid, [])
    ctx = _repair_context.pop(aid, None)
    return steps, ctx


def record(step: Step) -> None:
    aid = _get_agent_id()
    task = _ACTIVE.get(aid)
    if task is None:
        return
    if step.action not in RECORDABLE_ACTIONS:
        # Non-recordable actions are skipped, not fatal (Phase 0 hotfix)
        return
    if step.result_head.startswith("Error:"):
        task.failed = True
    task.steps.append(step)


async def finish(success: bool, host: str = "") -> str:
    global _ACTIVE
    aid = _get_agent_id()
    task = _ACTIVE.get(aid)
    if task is None:
        return "No task is being recorded."
    del _ACTIVE[aid]

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

    from .playbook_store import Playbook, _normalize_url_path, save_playbook

    playbook = Playbook(
        host=final_host,
        goal=task.goal,
        created_ts=time.time(),
        steps=task.steps,
        variables=variables,
    )
    # Derive url_template from first navigate step's URL
    url_template = "/*"
    for step in task.steps:
        if step.action in {"navigate", "newtab"} and step.params.get("url"):
            url_template = _normalize_url_path(step.params["url"])
            break
    playbook.url_template = url_template
    await save_playbook(playbook)
    await _autorecord_recipe(task.goal, final_host)
    stepped_count = len(task.steps)
    return f"Saved playbook for {task.goal!r} on {final_host} ({stepped_count} steps)."


async def _autorecord_recipe(goal: str, host: str) -> None:
    """Save a zero-token replay recipe alongside the playbook: a human_break
    (randomized range, fresh jitter each replay) then a browser step that
    re-runs the playbook. Never raises — recipe storage is best-effort."""
    try:
        import random
        import re

        from ...recipes import save_recipe

        slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", f"{host}-{goal}").strip("-")[:57]
        lo = random.randint(200, 600)  # noqa: S311  # human-pacing jitter, not crypto
        hi = lo + random.randint(400, 1500)  # noqa: S311
        steps = [
            {"kind": "human_break", "value": f"{lo}||{hi}"},
            {"kind": "browser", "value": goal, "note": f"replay {goal!r} on {host}"},
        ]
        await asyncio.to_thread(
            save_recipe, f"browse:{slug}", f"Replay browse: {goal}", steps, ("browser",)
        )
    except Exception as exc:
        logger.debug(f"recorder: autorecord recipe failed: {exc}")


async def _record_step(req: BrowseRequest, result: str) -> None:
    if not is_recording() and not is_repair_recording():
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
        # WI-2: Also record to repair buffer if active
        if is_repair_recording() and req.action in RECORDABLE_ACTIONS:
            aid = _get_agent_id()
            if aid in _repair_steps:
                _repair_steps[aid].append(step)
    except Exception:
        logger.debug("Failed to record task step", exc_info=True)
