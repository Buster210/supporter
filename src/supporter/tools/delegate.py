import asyncio
import functools
import inspect
import json
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from ..agent import ChatAgent
from ..config import (
    DELEGATE_ANOMALY_THRESHOLD,
    DELEGATE_HEARTBEAT_INTERVAL,
    DELEGATE_JOB_ID_LEN,
    DELEGATE_RETRY_BACKOFF,
    config,
)
from ..logger import logger
from ..types import (
    HeartbeatTick,
    LLMProvider,
    MilestoneCancelled,
    MilestoneCompleted,
    MilestoneStarted,
    TaskAnomaly,
    TaskCompleted,
    TaskFailed,
    TaskRetrying,
    TaskSkipped,
    TaskStarted,
    TaskStatus,
    TaskTimedOut,
)
from .base import ToolError
from .catalog import build_tool_catalog, select_delegate_tools
from .delegation_capsule import (
    create_capsule,
    extract_task_capsule_fields,
    mark_capsule_cancelled,
    mark_capsule_completed,
    mark_task_completed,
    mark_task_failed,
    mark_task_skipped,
    mark_task_started,
    mark_task_timed_out,
)
from .delegation_capsule import (
    serialize_capsule_result as _serialize_capsule_result,
)
from .event_bus import (
    DelegationBus as DelegationBus,
)
from .event_bus import (
    bus_exists as bus_exists,
)
from .event_bus import (
    get_bus as get_bus,
)
from .event_bus import (
    remove_bus as remove_bus,
)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_JOB_TASKS: dict[str, asyncio.Task[Any]] = {}
_on_delegation_start: Callable[[str], None] | None = None


def set_delegation_start_callback(cb: Callable[[str], None] | None) -> None:
    global _on_delegation_start
    _on_delegation_start = cb


def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    return _serialize_capsule_result(job_id)


@functools.cache
def _delegate_allowed_tool_names() -> set[str]:
    return set(select_delegate_tools(build_tool_catalog(), "all"))


def _create_sub_agent(
    task: dict[str, Any],
    provider: LLMProvider | None = None,
) -> tuple[ChatAgent, str]:
    from ..pool import get_provider

    registry = select_delegate_tools(build_tool_catalog(), task["tools"])
    if not provider:
        provider = get_provider(
            shared=False, model_name=task["model"], registry=registry
        )

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


def _truncate_delegate_output(output: str) -> str:
    if len(output) <= config.delegate_max_output_chars:
        return output
    return output[: config.delegate_max_output_chars] + "\n\n[Output truncated...]"


async def _run_sub_agent(
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    max_retries = task.get("max_retries", 0)
    retry_delays = DELEGATE_RETRY_BACKOFF
    last_result: dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        async with semaphore:
            start_time = time.perf_counter()
            task_id = task["id"]
            agent_label = task.get("agent") or "custom"

            if attempt > 0:
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                logger.info(
                    f"Sub-agent '{task_id}' retry {attempt}/{max_retries} "
                    f"after {delay}s backoff"
                )
                bus.publish(
                    TaskRetrying(
                        job_id=job_id,
                        task_id=task_id,
                        attempt=attempt + 1,
                        reason=last_result.get("output", "unknown error"),
                    )
                )
                await asyncio.sleep(delay)

            logger.info(
                f"Sub-agent '{task_id}' [{agent_label}] attempt {attempt + 1} started"
            )

            try:
                agent, prompt = _create_sub_agent(
                    task,
                    provider=provider,
                )
                result = await asyncio.wait_for(
                    agent.execute(prompt), timeout=task["timeout"]
                )
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Sub-agent '{task_id}' completed in {duration:.2f}s "
                    f"(attempt {attempt + 1})"
                )

                output = _truncate_delegate_output(
                    result.text or "(No text output returned)"
                )

                return {
                    "id": task_id,
                    "status": TaskStatus.COMPLETED,
                    "output": output,
                    "model": result.model,
                    "duration": duration,
                    "tokens": result.usage,
                }
            except TimeoutError:
                logger.warning(
                    f"Sub-agent '{task_id}' timed out after {task['timeout']}s "
                    f"(attempt {attempt + 1}) — no retry for timeouts"
                )
                return {
                    "id": task_id,
                    "status": TaskStatus.TIMEOUT,
                    "output": (
                        f"Error: Task exceeded execution limit of {task['timeout']}s"
                    ),
                    "duration": time.perf_counter() - start_time,
                    "tokens": {},
                }
            except Exception as e:
                logger.error(
                    f"Sub-agent '{task_id}' failed (attempt {attempt + 1}): {e}"
                )
                last_result = {
                    "id": task_id,
                    "status": TaskStatus.ERROR,
                    "output": f"Error [{type(e).__name__}]: {e}",
                    "duration": time.perf_counter() - start_time,
                    "tokens": {},
                }
                if attempt < max_retries:
                    continue
                return last_result

    return last_result


async def _safe_capsule_call(
    action: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    try:
        res = action(*args, **kwargs)
        if inspect.iscoroutine(res):
            return await res
        return res
    except Exception as e:
        logger.warning(f"Delegation capsule update failed [{type(e).__name__}]: {e}")
        return None


def _inject_dependency_context(
    task: dict[str, Any], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not task["depends_on"]:
        return task
    parts = []
    for dep_id in task["depends_on"]:
        dep = results.get(dep_id)
        if not dep:
            continue
        status = str(dep["status"]).upper()
        label = (
            f"'{dep_id}'"
            if dep["status"] == TaskStatus.COMPLETED
            else f"'{dep_id}' [{status}]"
        )
        parts.append(f"--- Output from {label} ---\n{dep.get('output', '')}")
    if not parts:
        return task
    enriched = task.copy()
    extra = "DEPENDENCY OUTPUTS:\n" + "\n\n".join(parts)
    enriched["context"] = (
        f"{enriched['context']}\n\n{extra}" if enriched["context"] else extra
    )
    return enriched


def _should_skip(
    task: dict[str, Any], results: dict[str, dict[str, Any]]
) -> str | None:
    if task.get("tolerate_failures"):
        return None
    for dep_id in task["depends_on"]:
        dep_result = results.get(dep_id)
        if dep_result and dep_result["status"] != TaskStatus.COMPLETED:
            return f"Dependency '{dep_id}' {dep_result['status']}"
    return None


def _compute_priority(task_id: str, tasks: list[dict[str, Any]]) -> int:
    dependents: dict[str, list[str]] = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t["depends_on"]:
            dependents[dep].append(t["id"])

    visited: set[str] = set()
    queue = deque([task_id])
    count = 0
    while queue:
        current = queue.popleft()
        for dep in dependents.get(current, []):
            if dep not in visited:
                visited.add(dep)
                count += 1
                queue.append(dep)
    return count


async def _execute_dag(
    tasks: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    parallel_limit: int,
) -> list[dict[str, Any]]:
    from ..pool import get_provider

    results: dict[str, dict[str, Any]] = {}
    task_done: dict[str, asyncio.Event] = {t["id"]: asyncio.Event() for t in tasks}
    priorities: dict[str, int] = {
        t["id"]: _compute_priority(t["id"], tasks) for t in tasks
    }
    milestone_pools: dict[str, LLMProvider] = {}

    async def _run_with_gate(task: dict[str, Any]) -> None:
        for dep_id in task["depends_on"]:
            await task_done[dep_id].wait()

        priority = priorities[task["id"]]
        if priority < max(priorities.values(), default=0):
            await asyncio.sleep(0)

        task_id = task["id"]
        agent_label = task.get("agent") or "custom"

        skip_reason = _should_skip(task, results)
        if skip_reason:
            state = {
                "id": task_id,
                "status": TaskStatus.SKIPPED,
                "output": f"Skipped: {skip_reason}",
                "duration": 0.0,
            }
            results[task_id] = state
            await _safe_capsule_call(mark_task_skipped, job_id, task_id, skip_reason)
            bus.update_task_state(
                task_id,
                {
                    "status": "SKIPPED",
                    "agent_label": agent_label,
                    "task_goal": task["task"],
                    "duration": 0.0,
                    "summary": f"Skipped: {skip_reason}",
                },
            )
            bus.publish(TaskSkipped(job_id=job_id, task_id=task_id, reason=skip_reason))
            task_done[task_id].set()
            return

        started_at = time.monotonic()
        bus.update_task_state(
            task_id,
            {
                "status": "RUNNING",
                "agent_label": agent_label,
                "task_goal": task["task"],
                "started_at": started_at,
                "timeout": task["timeout"],
                "anomaly_fired": False,
            },
        )
        enriched = _inject_dependency_context(task, results)
        await _safe_capsule_call(
            mark_task_started,
            job_id,
            task_id,
            dependency_context=enriched.get("context", ""),
        )
        bus.publish(
            TaskStarted(
                job_id=job_id,
                task_id=task_id,
                agent_label=agent_label,
                started_at=started_at,
                timeout=task["timeout"],
            )
        )

        model = task["model"]
        if model not in milestone_pools:
            milestone_pools[model] = get_provider(
                shared=False, model_name=model, pool_size=parallel_limit
            )

        result = await _run_sub_agent(
            enriched, semaphore, bus, job_id, provider=milestone_pools[model]
        )
        results[task_id] = result

        if result["status"] == TaskStatus.COMPLETED:
            parsed_fields = extract_task_capsule_fields(result.get("output", ""))
            evidence_raw = parsed_fields.get("evidence")
            evidence: dict[str, Any] = (
                evidence_raw if isinstance(evidence_raw, dict) else {}
            )
            evidence_counts = {
                key: len(value) if isinstance(value, list) else 0
                for key, value in evidence.items()
            }
            findings_raw = parsed_fields.get("findings")
            findings_count = len(findings_raw) if isinstance(findings_raw, list) else 0
            await _safe_capsule_call(
                mark_task_completed,
                job_id,
                task_id,
                result.get("output", ""),
                result.get("duration", 0.0),
                result.get("model") or task.get("model"),
                result.get("tokens", {}),
                parsed_fields,
            )
            bus.update_task_state(
                task_id,
                {
                    "status": "DONE",
                    "agent_label": agent_label,
                    "task_goal": task["task"],
                    "duration": result["duration"],
                    "summary": str(parsed_fields.get("summary", "")),
                },
            )
            bus.publish(
                TaskCompleted(
                    job_id=job_id,
                    task_id=task_id,
                    duration=result["duration"],
                    output=result.get("output", ""),
                    model=result.get("model", ""),
                    summary=str(parsed_fields.get("summary", "")),
                    confidence=str(parsed_fields.get("confidence", "unknown")),
                    findings_count=findings_count,
                    evidence_counts=evidence_counts,
                    handoff=str(parsed_fields.get("handoff", "")),
                )
            )
        elif result["status"] == TaskStatus.TIMEOUT:
            await _safe_capsule_call(
                mark_task_timed_out,
                job_id,
                task_id,
                result.get("output", "Task timed out"),
                result.get("duration", 0.0),
            )
            bus.update_task_state(
                task_id,
                {
                    "status": "TIMEOUT",
                    "agent_label": agent_label,
                    "task_goal": task["task"],
                    "duration": result["duration"],
                    "summary": "Execution timed out before completion.",
                },
            )
            bus.publish(
                TaskTimedOut(
                    job_id=job_id, task_id=task_id, duration=result["duration"]
                )
            )
        else:
            await _safe_capsule_call(
                mark_task_failed,
                job_id,
                task_id,
                result.get("output", "Unknown error"),
                result.get("duration", 0.0),
                result.get("output", ""),
            )
            bus.update_task_state(
                task_id,
                {
                    "status": "FAILED",
                    "agent_label": agent_label,
                    "task_goal": task["task"],
                    "duration": result["duration"],
                    "summary": result.get("output", "Unknown error"),
                },
            )
            bus.publish(
                TaskFailed(
                    job_id=job_id,
                    task_id=task_id,
                    duration=result["duration"],
                    error=result.get("output", "Unknown error"),
                )
            )

        task_done[task_id].set()

    await asyncio.gather(*[_run_with_gate(t) for t in tasks])
    return [results[t["id"]] for t in tasks if t["id"] in results]


def _format_results(
    milestone: str, results: list[dict[str, Any]], total_duration: float
) -> str:
    completed = sum(1 for r in results if r["status"] == TaskStatus.COMPLETED)
    skipped = sum(1 for r in results if r["status"] == TaskStatus.SKIPPED)
    total_tokens = sum(
        r.get("tokens", {}).get("total_tokens", 0)
        for r in results
        if r["status"] == TaskStatus.COMPLETED
    )
    token_line = f", Tokens: {total_tokens:,}" if total_tokens else ""
    report = [
        f"MILESTONE REPORT: {milestone}",
        f"Summary: {completed}/{len(results)} completed"
        + (f", {skipped} skipped" if skipped else "")
        + token_line,
        f"Total Wall Duration: {total_duration:.2f}s",
        "\n" + "=" * 40 + "\n",
    ]
    for r in results:
        report.append(f"### Task: {r['id']}")
        report.append(f"Status: {r['status'].upper()}")
        if "model" in r:
            report.append(f"Model: {r['model']}")
        report.append(f"Duration: {r['duration']:.2f}s")
        task_tokens = r.get("tokens", {}).get("total_tokens")
        if task_tokens:
            report.append(f"Tokens: {task_tokens:,}")
        report.append(f"\nOUTPUT:\n{r['output']}")
        report.append("\n" + "-" * 20 + "\n")
    return "\n".join(report)


def serialize_results(
    milestone: str,
    results: list[dict[str, Any]],
    total_duration: float,
    job_id: str,
    status: str = "completed",
) -> dict[str, Any]:
    completed = sum(1 for r in results if r["status"] == TaskStatus.COMPLETED)
    failed = sum(1 for r in results if r["status"] == TaskStatus.ERROR)
    skipped = sum(1 for r in results if r["status"] == TaskStatus.SKIPPED)
    timed_out = sum(1 for r in results if r["status"] == TaskStatus.TIMEOUT)
    total_tokens = sum(
        r.get("tokens", {}).get("total_tokens", 0)
        for r in results
        if r["status"] == TaskStatus.COMPLETED
    )
    summary_tasks = []
    for r in results:
        entry = {
            "id": r["id"],
            "status": str(r["status"]),
            "duration": round(r.get("duration", 0.0), 2),
        }
        if r.get("model"):
            entry["model"] = r["model"]
        token_total = r.get("tokens", {}).get("total_tokens")
        if token_total:
            entry["tokens"] = token_total
        output = r.get("output", "")
        if r["status"] == TaskStatus.ERROR:
            entry["error"] = output
        else:
            entry["output"] = output
        summary_tasks.append(entry)
    return {
        "job_id": job_id,
        "milestone": milestone,
        "status": status,
        "total_duration": round(total_duration, 2),
        "totals": {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "timed_out": timed_out,
            "tokens": total_tokens,
        },
        "tasks": summary_tasks,
    }


async def _run_heartbeat(
    bus: DelegationBus, job_id: str, interval: int = DELEGATE_HEARTBEAT_INTERVAL
) -> None:
    while bus_exists(job_id):
        await asyncio.sleep(interval)
        if not bus_exists(job_id):
            return

        now = time.monotonic()
        snapshot = bus.get_snapshot()
        bus.publish(
            HeartbeatTick(job_id=job_id, milestone=bus.milestone, snapshot=snapshot)
        )

        for task_id, state in snapshot.items():
            if state.get("status") != "RUNNING":
                continue
            started_at = state.get("started_at")
            task_timeout = state.get("timeout")
            if started_at is not None:
                elapsed = now - started_at
                if (
                    task_timeout is not None
                    and elapsed >= DELEGATE_ANOMALY_THRESHOLD * task_timeout
                    and not state.get("anomaly_fired")
                ):
                    bus.publish(
                        TaskAnomaly(
                            job_id=job_id,
                            task_id=task_id,
                            agent_label=state.get("agent_label", "?"),
                            elapsed_seconds=elapsed,
                            timeout=task_timeout,
                        )
                    )
                    bus.update_task_state(task_id, {**state, "anomaly_fired": True})


async def _run_milestone(
    milestone: str,
    tasks: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    parallel_limit: int,
    heartbeat_task: asyncio.Task[None] | None = None,
) -> None:
    start_wall = time.perf_counter()
    try:
        results = await _execute_dag(tasks, semaphore, bus, job_id, parallel_limit)
        total_wall = time.perf_counter() - start_wall
        await _safe_capsule_call(mark_capsule_completed, job_id)
        bus.publish(MilestoneCompleted(job_id, milestone, results, total_wall))
    except asyncio.CancelledError:
        total_wall = time.perf_counter() - start_wall
        logger.info(f"Milestone '{milestone}' (job={job_id}) cancelled")
        await _safe_capsule_call(mark_capsule_cancelled, job_id)
        bus.publish(MilestoneCancelled(job_id, milestone, total_wall))
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
        bus.close()
        remove_bus(job_id)
        _JOB_TASKS.pop(job_id, None)


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
    allowed_tools = _delegate_allowed_tool_names()
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

    return {
        "id": task_id,
        "task": task_desc,
        "agent": t.get("agent"),
        "persona": profile["persona"],
        "tools": granted_tools,
        "model": profile.get("model") or config.gemini_model,
        "context": t.get("context") or "",
        "timeout": task_timeout,
        "max_retries": max_retries,
        "depends_on": depends_on,
        "pre_approved_commands": pre_approved_commands,
        "tolerate_failures": bool(t.get("tolerate_failures", False)),
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


async def delegate_tasks(
    milestone: str,
    tasks: str,
    max_parallel: int = 3,
    notify_per_task: bool = True,
) -> str:
    """Orchestrates background sub-agents to complete a complex milestone."""
    logger.info(f"Tool: delegate_tasks -- milestone='{milestone}'")
    try:
        validated_tasks = _validate_tasks(tasks)
        parallel_cap = max(1, min(max_parallel, config.delegate_max_hard_cap))
        semaphore = asyncio.Semaphore(parallel_cap)
        job_id = str(uuid.uuid4())[:DELEGATE_JOB_ID_LEN]

        bus = get_bus(job_id, milestone)
        bus.notify_per_task = notify_per_task
        for validated_task in validated_tasks:
            bus.update_task_state(
                validated_task["id"],
                {
                    "status": "PENDING",
                    "agent_label": validated_task.get("agent") or "custom",
                    "task_goal": validated_task["task"],
                    "duration": 0.0,
                },
            )
        await _safe_capsule_call(
            create_capsule, job_id, milestone, validated_tasks, parallel_cap
        )
        if _on_delegation_start:
            _on_delegation_start(job_id)

        bus.publish(
            MilestoneStarted(
                job_id=job_id,
                milestone=milestone,
                task_ids=[t["id"] for t in validated_tasks],
                parallel_cap=parallel_cap,
            )
        )

        hb_task = asyncio.create_task(_run_heartbeat(bus, job_id))
        _BACKGROUND_TASKS.add(hb_task)
        hb_task.add_done_callback(_BACKGROUND_TASKS.discard)

        milestone_task = asyncio.create_task(
            _run_milestone(
                milestone,
                validated_tasks,
                semaphore,
                bus,
                job_id,
                parallel_cap,
                hb_task,
            )
        )
        _JOB_TASKS[job_id] = milestone_task
        _BACKGROUND_TASKS.add(milestone_task)
        milestone_task.add_done_callback(_BACKGROUND_TASKS.discard)

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
        plan.append(
            "\nResults will be automatically posted back here when the "
            "milestone is complete. DO NOT check for results constantly; "
            "wait for the system message."
        )
        plan.append(
            f"You can also use `check_delegation(job_id='{job_id}')` "
            "for a live non-blocking snapshot, but DO NOT do this "
            "unless asked by the user."
        )
        return "\n".join(plan)
    except Exception as e:
        raise ToolError(f"Delegation failed: {e}") from e


async def check_delegation(job_id: str) -> str:
    """Non-blocking snapshot of the current job state."""
    if not bus_exists(job_id):
        return f"Job `{job_id}` is unknown or already complete."

    now = time.monotonic()
    bus = get_bus(job_id)
    snapshot = bus.get_snapshot()

    if not snapshot:
        return f"Job `{job_id}` has no tasks tracked yet."

    rows = []
    for task_id, state in snapshot.items():
        status = state.get("status", "UNKNOWN")
        agent_label = state.get("agent_label", "?")
        if status == "RUNNING" and state.get("started_at") is not None:
            elapsed = f"{now - state['started_at']:.0f}s / {state.get('timeout', '?')}s"
        else:
            elapsed = f"{state.get('duration', 0):.1f}s"
        rows.append(f"| `{task_id}` | {status} | {agent_label} | {elapsed} |")

    header = "| Task | Status | Agent | Elapsed |"
    separator = "|---|---|---|---|"
    table = "\n".join([header, separator, *rows])
    return f"**Job `{job_id}` — {bus.milestone}**\n\n{table}"


async def cancel_delegation(job_id: str) -> str:
    """Cancels a running delegation job."""
    task = _JOB_TASKS.get(job_id)
    if task is None or task.done():
        return f"Job `{job_id}` is unknown or already complete."

    task.cancel()
    return f"Cancellation requested for job `{job_id}`."


__all__ = [
    "DelegationBus",
    "bus_exists",
    "cancel_delegation",
    "check_delegation",
    "delegate_tasks",
    "get_bus",
    "remove_bus",
    "serialize_capsule_result",
    "set_delegation_start_callback",
]
