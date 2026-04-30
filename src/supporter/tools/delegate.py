import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from ..agent import ChatAgent
from ..config import config
from ..logger import logger
from .bash import execute_bash
from .file_ops import read_file, write_file
from .search import google_search

_JOB_REGISTRY: dict[str, asyncio.Future[str]] = {}
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _resolve_agent_profile(task: dict[str, Any]) -> dict[str, Any]:
    agent_name = task.get("agent")

    if (
        agent_name
        and agent_name != "custom"
        and agent_name in config.delegate_agent_roster
    ):
        profile = config.delegate_agent_roster[agent_name].copy()
        profile.update(
            {k: task[k] for k in ["persona", "tools", "model"] if task.get(k)}
        )
        return profile

    return {
        "persona": task.get("persona") or config.delegate_default_persona,
        "tools": task.get("tools"),
        "model": task.get("model"),
    }


def _validate_tasks(raw_json: str) -> list[dict[str, Any]]:
    try:
        tasks = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in 'tasks' parameter: {e}") from e

    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a JSON array")

    if not tasks:
        raise ValueError("'tasks' array cannot be empty")

    if len(tasks) > config.delegate_max_tasks:
        raise ValueError(
            f"Too many tasks in one milestone (max {config.delegate_max_tasks})"
        )

    validated = []
    seen_ids = set()

    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            raise ValueError(f"Task at index {i} must be an object")

        task_id = t.get("id")
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Task at index {i} is missing a valid string 'id'")

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID detected: {task_id}")
        seen_ids.add(task_id)

        task_desc = t.get("task")
        if not task_desc or not isinstance(task_desc, str):
            raise ValueError(
                f"Task '{task_id}' is missing a valid string 'task' description"
            )

        profile = _resolve_agent_profile(t)
        raw_tools = profile.get("tools") or t.get("tools", "all")

        if isinstance(raw_tools, set):
            granted_tools = raw_tools.intersection(config.delegate_allowed_tools)
        elif raw_tools == "all":
            granted_tools = config.delegate_allowed_tools
        elif isinstance(raw_tools, str):
            requested = {tool.strip() for tool in raw_tools.split(",") if tool.strip()}
            granted_tools = requested.intersection(config.delegate_allowed_tools)
        else:
            granted_tools = config.delegate_allowed_tools

        raw_timeout = t.get("timeout")
        task_timeout = (
            min(max(1, raw_timeout), config.delegate_max_timeout)
            if isinstance(raw_timeout, (int, float))
            else config.delegate_default_timeout
        )

        depends_on = t.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [d.strip() for d in depends_on.split(",") if d.strip()]
        elif not isinstance(depends_on, list):
            depends_on = []

        validated.append(
            {
                "id": task_id,
                "task": task_desc,
                "agent": t.get("agent"),
                "persona": profile["persona"],
                "tools": granted_tools,
                "model": profile.get("model") or config.gemini_model,
                "context": t.get("context") or "",
                "timeout": task_timeout,
                "depends_on": depends_on,
            }
        )

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


def _build_tool_registry(tool_names: set[str]) -> dict[str, Callable[..., Any]]:
    all_tools: dict[str, Callable[..., Any]] = {
        "read_file": read_file,
        "write_file": write_file,
        "execute_bash": execute_bash,
        "google_search": google_search,
    }
    registry = {name: func for name, func in all_tools.items() if name in tool_names}
    registry.pop("delegate_tasks", None)
    return registry


def _create_sub_agent(task: dict[str, Any]) -> tuple[ChatAgent, str]:
    from ..index import get_provider

    registry = _build_tool_registry(task["tools"])
    provider = get_provider(shared=True, model_name=task["model"], registry=registry)
    agent = ChatAgent(
        provider=provider,
        registry=registry,
        use_search="google_search" in task["tools"],
        system_instruction=task["persona"],
    )

    prompt = f"TASK:\n{task['task']}"
    if task["context"]:
        prompt += f"\n\nCONTEXT:\n{task['context']}"
    return agent, prompt


async def _run_sub_agent(
    task: dict[str, Any], semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    async with semaphore:
        start_time = time.perf_counter()
        task_id = task["id"]
        agent_label = task.get("agent") or "custom"
        logger.info(f"Sub-agent '{task_id}' [{agent_label}] started execution")

        try:
            agent, prompt = _create_sub_agent(task)
            result = await asyncio.wait_for(
                agent.execute(prompt), timeout=task["timeout"]
            )
            duration = time.perf_counter() - start_time
            logger.info(f"Sub-agent '{task_id}' completed in {duration:.2f}s")

            output = result.text or "(No text output returned)"
            if len(output) > config.delegate_max_output_chars:
                output = (
                    output[: config.delegate_max_output_chars]
                    + "\n\n[Output truncated...]"
                )

            return {
                "id": task_id,
                "status": "completed",
                "output": output,
                "model": result.model,
                "duration": duration,
            }
        except TimeoutError:
            logger.warning(f"Sub-agent '{task_id}' timed out after {task['timeout']}s")
            return {
                "id": task_id,
                "status": "timeout",
                "output": f"Error: Task exceeded execution limit of {task['timeout']}s",
                "duration": time.perf_counter() - start_time,
            }
        except Exception as e:
            logger.error(f"Sub-agent '{task_id}' failed: {e}")
            return {
                "id": task_id,
                "status": "error",
                "output": f"Error [{type(e).__name__}]: {e}",
                "duration": time.perf_counter() - start_time,
            }


async def _run_and_push(
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
    result_queue: asyncio.Queue[dict[str, Any]],
) -> None:
    result = await _run_sub_agent(task, semaphore)
    await result_queue.put(result)


async def _execute_dag(
    tasks: list[dict[str, Any]], semaphore: asyncio.Semaphore
) -> list[dict[str, Any]]:
    task_map = {t["id"]: t for t in tasks}
    results: dict[str, dict[str, Any]] = {}
    launched: set[str] = set()
    remaining_deps = {t["id"]: set(t["depends_on"]) for t in tasks}
    result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def _get_ready_tasks() -> list[dict[str, Any]]:
        return [
            task_map[tid]
            for tid, deps in remaining_deps.items()
            if tid not in results and tid not in launched and not deps
        ]

    def _inject_dependency_context(task: dict[str, Any]) -> dict[str, Any]:
        if not task["depends_on"]:
            return task
        parts = [
            f"--- Output from '{d}' ---\n{results[d]['output']}"
            for d in task["depends_on"]
            if results.get(d, {}).get("status") == "completed"
        ]
        if not parts:
            return task
        enriched = task.copy()
        extra = "DEPENDENCY OUTPUTS:\n" + "\n\n".join(parts)
        enriched["context"] = (
            f"{enriched['context']}\n\n{extra}" if enriched["context"] else extra
        )
        return enriched

    def _should_skip(task: dict[str, Any]) -> str | None:
        for dep_id in task["depends_on"]:
            dep_result = results.get(dep_id)
            if dep_result and dep_result["status"] != "completed":
                return f"Dependency '{dep_id}' {dep_result['status']}"
        return None

    def _launch_ready() -> int:
        count = 0
        while True:
            newly_skipped = 0
            for task in _get_ready_tasks():
                skip_reason = _should_skip(task)
                if skip_reason:
                    results[task["id"]] = {
                        "id": task["id"],
                        "status": "skipped",
                        "output": f"Skipped: {skip_reason}",
                        "duration": 0.0,
                    }
                    launched.add(task["id"])
                    for _, deps in remaining_deps.items():
                        deps.discard(task["id"])
                    newly_skipped += 1
                else:
                    enriched_task = _inject_dependency_context(task)
                    t = asyncio.create_task(
                        _run_and_push(enriched_task, semaphore, result_queue)
                    )
                    _BACKGROUND_TASKS.add(t)
                    t.add_done_callback(_BACKGROUND_TASKS.discard)
                    launched.add(task["id"])
                    count += 1
            if newly_skipped == 0:
                break
        return count

    active_count = _launch_ready()
    while len(results) < len(tasks):
        if active_count == 0 and not any(tid not in results for tid in launched):
            break
        try:
            result = await asyncio.wait_for(
                result_queue.get(), timeout=config.delegate_heartbeat_interval
            )
        except TimeoutError:
            continue
        tid = result["id"]
        results[tid] = result
        active_count -= 1
        for _, deps in remaining_deps.items():
            deps.discard(tid)
        active_count += _launch_ready()

    return [results[t["id"]] for t in tasks if t["id"] in results]


def _format_results(
    milestone: str, results: list[dict[str, Any]], total_duration: float
) -> str:
    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    report = [
        f"MILESTONE REPORT: {milestone}",
        f"Summary: {completed}/{len(results)} completed"
        + (f", {skipped} skipped" if skipped else ""),
        f"Total Wall Duration: {total_duration:.2f}s",
        "\n" + "=" * 40 + "\n",
    ]
    for r in results:
        report.append(f"### Task: {r['id']}")
        report.append(f"Status: {r['status'].upper()}")
        if "model" in r:
            report.append(f"Model: {r['model']}")
        report.append(f"Duration: {r['duration']:.2f}s")
        report.append(f"\nOUTPUT:\n{r['output']}")
        report.append("\n" + "-" * 20 + "\n")
    return "\n".join(report)


async def delegate_tasks(milestone: str, tasks: str, max_parallel: int = 3) -> str:
    logger.info(f"Tool: delegate_tasks -- milestone='{milestone}'")
    try:
        validated_tasks = _validate_tasks(tasks)
        parallel_cap = max(1, min(max_parallel, config.delegate_max_hard_cap))
        semaphore = asyncio.Semaphore(parallel_cap)
        job_id = str(uuid.uuid4())[:8]
        future = asyncio.get_running_loop().create_future()
        _JOB_REGISTRY[job_id] = future

        async def _run_milestone() -> None:
            start_wall = time.perf_counter()
            results = await _execute_dag(validated_tasks, semaphore)
            total_wall = time.perf_counter() - start_wall
            report = _format_results(milestone, results, total_wall)
            future.set_result(report)

        task = asyncio.create_task(_run_milestone())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

        plan = [
            f"Delegation started for milestone: **{milestone}**",
            f"Job ID: `{job_id}`",
            "\n| # | Task ID | Agent | Dependencies |",
            "|---|---------|-------|--------------|",
        ]
        for i, t in enumerate(validated_tasks, 1):
            deps = ", ".join(t["depends_on"]) or "none"
            plan.append(
                f"| {i} | {t['id']} | {t['agent'] or 'custom'} | after: {deps} |"
            )
        plan.append(f"\nSub-agents are running with parallel limit: {parallel_cap}")
        plan.append(f"\nUse `collect_delegation(job_id='{job_id}')` to get results.")
        return "\n".join(plan)
    except Exception as e:
        logger.error(f"Tool Failure: delegate_tasks: {e}")
        return f"Error: {e}"


async def collect_delegation(job_id: str) -> str:
    if job_id not in _JOB_REGISTRY:
        return f"Error: Unknown Job ID: {job_id}"
    try:
        return await _JOB_REGISTRY[job_id]
    except Exception as e:
        return f"Error collecting results: {e}"
    finally:
        _JOB_REGISTRY.pop(job_id, None)
