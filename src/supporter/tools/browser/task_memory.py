from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...logger import logger
from .. import resolved_project_root

# Actions safe to suggest for reuse: deterministic, single-tab, top-page reads
# and inputs. Anything not listed poisons the buffer (see record()).
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


# Module-global active recording buffer (mirrors session.py's global state).
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
