import json
import os
from collections import deque
from typing import Any

from ...config import config
from .agents import delegate_allowed_tool_names
from .backends import GEMINI_BACKEND, KNOWN_BACKENDS


def _resolve_agent_profile(task: dict[str, Any]) -> dict[str, Any]:
    agent_name = task.get("agent")
    env_model = (
        os.getenv(f"{agent_name.upper()}_MODEL", "").strip() or None
        if agent_name
        else None
    )

    if (
        agent_name
        and agent_name != "custom"
        and agent_name in config.delegate_agent_roster
    ):
        profile = config.delegate_agent_roster[agent_name].copy()
        profile.update(
            {k: task[k] for k in ["persona", "tools", "model"] if task.get(k)}
        )
        if not task.get("model") and env_model:
            profile["model"] = env_model
        if "live" in task:
            profile["live"] = bool(task["live"])
        return profile

    return {
        "persona": task.get("persona") or config.delegate_default_persona,
        "tools": task.get("tools"),
        "model": task.get("model") or env_model,
        "live": bool(task.get("live", False)),
    }


def _validate_single_task(
    t: dict[str, Any], index: int, seen_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(t, dict):
        raise ValueError(f"Task at index {index} must be an object")

    task_id = t.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Task at index {index} is missing a valid string 'id'")

    if task_id in seen_ids:
        raise ValueError(f"Duplicate task ID detected: {task_id}")
    seen_ids.add(task_id)

    task_desc = t.get("task")
    if not isinstance(task_desc, str) or not task_desc:
        raise ValueError(
            f"Task '{task_id}' is missing a valid string 'task' description"
        )

    profile = _resolve_agent_profile(t)
    raw_tools = profile.get("tools") or t.get("tools", "all")
    allowed_tools = delegate_allowed_tool_names(t.get("agent"))
    granted_tools = allowed_tools

    if isinstance(raw_tools, set):
        granted_tools = raw_tools.intersection(allowed_tools)
    elif raw_tools == "all":
        granted_tools = allowed_tools
    elif isinstance(raw_tools, str):
        requested = {tool.strip() for tool in raw_tools.split(",") if tool.strip()}
        granted_tools = requested.intersection(allowed_tools)

    raw_timeout = t.get("timeout")
    task_timeout = config.delegate_default_timeout
    if isinstance(raw_timeout, (int, float)):
        task_timeout = int(min(max(1, raw_timeout), config.delegate_max_timeout))

    raw_retries = t.get("retry", 0)
    max_retries = 0
    if isinstance(raw_retries, (int, float)):
        max_retries = min(max(0, int(raw_retries)), config.delegate_max_retries)

    depends_on = t.get("depends_on", [])
    if isinstance(depends_on, str):
        depends_on = [d.strip() for d in depends_on.split(",") if d.strip()]
    elif not isinstance(depends_on, list):
        depends_on = []

    pre_approved_commands = t.get("pre_approved_commands", [])
    if not isinstance(pre_approved_commands, list) or not all(
        isinstance(c, str) for c in pre_approved_commands
    ):
        pre_approved_commands = []

    raw_backend = t.get("backend")
    if raw_backend is None:
        backend = GEMINI_BACKEND
    elif isinstance(raw_backend, str):
        backend = raw_backend.lower()
    else:
        raise ValueError(
            f"Task '{task_id}' has a non-string 'backend': {raw_backend!r}"
        )
    if backend not in KNOWN_BACKENDS:
        known = ", ".join(sorted(KNOWN_BACKENDS))
        raise ValueError(
            f"Task '{task_id}' requests unknown backend '{backend}'; "
            f"known backends: {known}"
        )

    live = bool(profile.get("live", False))
    default_model = config.gemini_live_model if live else config.gemini_model
    return {
        "id": task_id,
        "task": task_desc,
        "agent": t.get("agent"),
        "backend": backend,
        "persona": profile["persona"],
        "tools": granted_tools,
        "model": profile.get("model") or default_model,
        "live": live,
        "context": t.get("context") or "",
        "timeout": task_timeout,
        "max_retries": max_retries,
        "depends_on": depends_on,
        "pre_approved_commands": pre_approved_commands,
        "tolerate_failures": bool(t.get("tolerate_failures", False)),
    }


RECON_ONLY_ROLES = frozenset(
    {"planner", "worker_planner", "explorer", "code_reviewer", "security_auditor"}
)


def _inject_plan_gate(tasks: list[Any]) -> list[Any]:
    """Prepend a synthetic planner task when a milestone needs a plan.

    ponytail: the recon-role allowlist is hardcoded because `agent` is
    free-form (no role enum) -- upgrade path is a real role taxonomy.
    """
    needs_plan = any(
        (t.get("agent") if isinstance(t, dict) else None) not in RECON_ONLY_ROLES
        or (t.get("backend") if isinstance(t, dict) else None) == "opencode"
        for t in tasks
        if isinstance(t, dict)
    )
    if not needs_plan:
        return tasks

    has_planner = any(
        isinstance(t, dict) and t.get("agent") == "planner" for t in tasks
    )
    if has_planner:
        return tasks

    existing_ids = {t.get("id") for t in tasks if isinstance(t, dict)}
    plan_id = "__plan__"
    n = 1
    while plan_id in existing_ids:
        plan_id = f"__plan___{n}"
        n += 1

    plan_task = {
        "id": plan_id,
        "agent": "planner",
        "task": (
            "Produce a Plan and Success Criteria for the milestone before execution."
        ),
        "depends_on": [],
    }
    # Wire the plan ahead of every root (empty depends_on). A valid acyclic
    # milestone always has >=1 root (a DAG source), so the plan can never end
    # up orphaned here: any input with zero roots is cyclic or references
    # missing ids, and the cycle/dep-existence checks that run downstream in
    # validate_tasks reject it before this list is ever returned.
    # ponytail: no orphan-planner guard -- downstream validation is the net;
    # add one only if injection ever moves after those checks.
    for t in tasks:
        if isinstance(t, dict) and not t.get("depends_on"):
            t["depends_on"] = [plan_id]
    return [plan_task, *tasks]


def validate_tasks(raw_json: str) -> list[dict[str, Any]]:
    try:
        tasks = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in 'tasks' parameter: {e}") from e

    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a JSON array")

    if not tasks:
        raise ValueError("'tasks' array cannot be empty")

    if config.router_enabled:
        tasks = _inject_plan_gate(tasks)

    if len(tasks) > config.delegate_max_tasks:
        raise ValueError(
            f"Too many tasks in one milestone (max {config.delegate_max_tasks})"
        )

    seen_ids: set[str] = set()
    validated = [_validate_single_task(t, i, seen_ids) for i, t in enumerate(tasks)]

    all_ids = {t["id"] for t in validated}
    for t in validated:
        for dep in t["depends_on"]:
            if dep not in all_ids:
                raise ValueError(
                    f"Task '{t['id']}' depends on '{dep}', which does not exist"
                )
        if t["id"] in t["depends_on"]:
            raise ValueError(f"Task '{t['id']}' cannot depend on itself")

    _detect_cycles(validated)
    return validated


def _detect_cycles(tasks: list[dict[str, Any]]) -> None:
    in_degree = {t["id"]: len(t["depends_on"]) for t in tasks}
    dependents: dict[str, list[str]] = {t["id"]: [] for t in tasks}

    for t in tasks:
        for dep in t["depends_on"]:
            dependents[dep].append(t["id"])

    queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        current = queue.popleft()
        visited += 1
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if visited != len(tasks):
        raise ValueError("Dependency cycle detected in task graph")
