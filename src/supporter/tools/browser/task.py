from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...logger import logger
from .. import resolved_project_root
from . import session

if TYPE_CHECKING:
    from .tool import BrowseRequest

RECORDABLE_ACTIONS = frozenset(
    {
        "navigate",
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

_SLUG_MAX = 80
_RESULT_HEAD_MAX = 120

_TARGET_ACTIONS = frozenset({"click", "type", "hover", "select", "press", "scroll"})

_RECORD_PARAMS: dict[str, tuple[str, ...]] = {
    "navigate": ("url",),
    "type": ("text",),
    "select": ("value", "text"),
    "press": ("key",),
    "scroll": ("dx", "dy"),
    "wait": ("selector", "delay_ms"),
    "extract": ("html",),
}


@dataclass
class Step:
    action: str
    role: str = ""
    name: str = ""
    selector: str = ""
    url_before: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    result_head: str = ""


@dataclass
class Playbook:
    host: str
    goal: str
    created_ts: float
    steps: list[Step]


@dataclass
class _ActiveTask:
    goal: str
    steps: list[Step] = field(default_factory=list)
    failed: bool = False


_ACTIVE: _ActiveTask | None = None


def start(goal: str) -> str:
    global _ACTIVE
    goal = goal.strip()
    if not goal:
        return "Error: a task goal is required."
    _ACTIVE = _ActiveTask(goal=goal)
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
        _ACTIVE.failed = True
        return
    if step.result_head.startswith("Error:"):
        _ACTIVE.failed = True
    _ACTIVE.steps.append(step)


async def finish(success: bool, host: str) -> str:
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
    if not host:
        return f"Task {task.goal!r}: could not resolve a host; nothing saved."

    playbook = Playbook(
        host=host, goal=task.goal, created_ts=time.time(), steps=task.steps
    )
    await save_playbook(playbook)
    return f"Saved playbook for {task.goal!r} on {host} ({len(task.steps)} steps)."


def _slug(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", goal.lower()).strip("-._")
    return (slug or "task")[:_SLUG_MAX]


def _safe_host(host: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", host.lower()).strip("-._")


def _memory_root() -> Path:
    return resolved_project_root() / ".supporter" / "task_memory"


def _safe_path(host: str, goal: str) -> Path:
    root = _memory_root().resolve()
    path = (root / _safe_host(host) / f"{_slug(goal)}.json").resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Playbook path '{path}' escapes memory root '{root}'.")
    return path


def _save_playbook_sync(playbook: Playbook) -> None:
    path = _safe_path(playbook.host, playbook.goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(playbook), f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


async def save_playbook(playbook: Playbook) -> None:
    await asyncio.to_thread(_save_playbook_sync, playbook)


def load_playbook(host: str, goal: str) -> Playbook | None:
    try:
        path = _safe_path(host, goal)
        with path.open(encoding="utf-8") as f:
            data: Any = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("playbook is not a JSON object")
        steps = [Step(**s) for s in data["steps"]]
        return Playbook(
            host=str(data["host"]),
            goal=str(data["goal"]),
            created_ts=float(data["created_ts"]),
            steps=steps,
        )
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ):
        logger.debug(f"No usable playbook for {goal!r} on {host}", exc_info=True)
        return None


def format_playbook(playbook: Playbook) -> str:
    lines = [f"Playbook for {playbook.goal!r} on {playbook.host}:"]
    for i, step in enumerate(playbook.steps, 1):
        target = step.selector or (
            f"{step.role} {step.name!r}".strip() if step.role or step.name else ""
        )
        detail = f"{step.action}" + (f" → {target}" if target else "")
        params = {k: v for k, v in step.params.items() if v not in ("", 0, False)}
        if params:
            detail += f" {params}"
        lines.append(f"  {i}. {detail}")
    return "\n".join(lines)


def build_step(
    action: str,
    *,
    role: str = "",
    name: str = "",
    selector: str = "",
    url_before: str = "",
    params: dict[str, Any] | None = None,
    result: str = "",
) -> Step:
    return Step(
        action=action,
        role=role,
        name=name,
        selector=selector,
        url_before=url_before,
        params={k: v for k, v in (params or {}).items() if v not in ("", 0, False)},
        result_head=result[:_RESULT_HEAD_MAX],
    )


async def _record_step(req: BrowseRequest, result: str) -> None:
    if not is_recording():
        return
    try:
        from .tool import (
            _page_host,
            _record_locator,
            _resolve_role_and_name,
        )

        page = session.active_page()
        if page is None:
            return
        host = await _page_host(page)
        role = name = ""
        if req.action in _TARGET_ACTIONS and (req.ref or req.selector):
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
        )
        record(step)
    except Exception:
        logger.debug("Failed to record task step", exc_info=True)


async def start_task(goal: str) -> str:
    """Begin remembering a browser task so its steps can be reused later.

    Call this before you start a multi-step flow (e.g. "log in to X", "search
    GitHub for Y"). Browse normally afterward; each eligible step is recorded.
    Call finish_task when done to save it (on success) or discard it.

    Args:
        goal: A short, stable description of the task. The same wording lets
            query_playbook find this playbook on a future run.

    Returns:
        A confirmation message.
    """
    logger.info(f"Tool: start_task — goal={goal!r}")
    return start(goal)


async def finish_task(success: bool = True) -> str:
    """End the current task, save its steps, and resolve the browser lifecycle.

    Saves a reusable playbook only if success is True and no errored or unsafe
    step occurred; otherwise nothing is stored. Always clears the buffer. Then
    applies the user's keep-open choice: a persistent session is left untouched,
    while an atomic one prompts to close the browser now.

    Args:
        success: Whether the task completed successfully. False discards it.

    Returns:
        A message stating whether a playbook was saved and the browser's state.
    """
    logger.info(f"Tool: finish_task — success={success}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    saved = await finish(success, host)

    lifecycle = await session.resolve_close_at_task_end()
    return f"{saved}\n{lifecycle}" if lifecycle else saved


def _find_ref(snapshot_text: str, role: str, name: str) -> str:
    from .snapshot import _NAME_PATTERN, _REF_GROUP, _ROLE_PATTERN

    for raw in snapshot_text.splitlines():
        body = raw.lstrip().removeprefix("- ")
        role_match = _ROLE_PATTERN.match(body)
        if (role_match.group(1) if role_match else "") != role:
            continue
        name_match = _NAME_PATTERN.search(body)
        if (name_match.group(1) if name_match else "") != name:
            continue
        ref_match = _REF_GROUP.search(body)
        if ref_match:
            return ref_match.group(1)
    return ""


def _replay_params(step: Step) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(step.params)
    if step.selector:
        kwargs["selector"] = step.selector
    return kwargs


async def replay_playbook(goal: str) -> str:
    """Replay a saved playbook on the current site, step by step, until one fails.

    Executes each recorded step in order, re-deriving live element refs from a
    fresh snapshot (refs are ephemeral). On the first step that can't be resolved
    or whose result is an error, it STOPS and hands control back with what ran,
    the failing step, and the current page — so you continue from there yourself.
    Far cheaper than re-discovering a known flow; falls back to you on any drift.

    Args:
        goal: The task description used when the playbook was recorded.

    Returns:
        A success summary, or a handback describing where replay stopped.
    """
    logger.info(f"Tool: replay_playbook — goal={goal!r}")
    from .tool import _live_refs_snapshot, _page_host, browse

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then replay a playbook."
    playbook = load_playbook(host, goal)
    if playbook is None:
        return f"No playbook found for {goal!r} on {host}."

    done: list[str] = []
    for i, step in enumerate(playbook.steps, 1):
        kwargs = _replay_params(step)
        if step.action in _TARGET_ACTIONS and not step.selector:
            live_page = session.active_page()
            snapshot_text = (
                await _live_refs_snapshot(live_page) if live_page is not None else ""
            )
            ref = _find_ref(snapshot_text, step.role, step.name)
            if not ref:
                target = f"{step.role} {step.name!r}".strip()
                return (
                    f"Replayed {len(done)}/{len(playbook.steps)} step(s): "
                    f"{'; '.join(done) or 'none'}.\n"
                    f"Stopped at step {i} ({step.action} → {target}): element not "
                    f"found on the current page. Continue from here yourself.\n"
                    f"Current page:\n{snapshot_text}"
                )
            kwargs["ref"] = ref

        result = await browse(step.action, **kwargs)
        if result.startswith("Error:"):
            return (
                f"Replayed {len(done)}/{len(playbook.steps)} step(s): "
                f"{'; '.join(done) or 'none'}.\n"
                f"Stopped at step {i} ({step.action}): {result}\n"
                f"Continue from here yourself."
            )
        done.append(f"{i}.{step.action}")

    n = len(playbook.steps)
    return f"Replayed playbook {goal!r} on {host}: {n}/{n} steps succeeded."


async def query_playbook(goal: str) -> str:
    """Look up a saved playbook for this site + task to guide your next steps.

    Call before reasoning out a flow from scratch: if a playbook exists you can
    follow its steps instead of re-discovering them. The steps are advisory —
    read them and decide; nothing is executed automatically.

    Args:
        goal: The task description used when the playbook was recorded.

    Returns:
        A numbered step list, or a note that no playbook was found.
    """
    logger.info(f"Tool: query_playbook — goal={goal!r}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then query a playbook."
    playbook = load_playbook(host, goal)
    if playbook is None:
        return f"No playbook found for {goal!r} on {host}."
    return format_playbook(playbook)
