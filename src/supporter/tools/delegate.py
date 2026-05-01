import asyncio
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
from .bash import execute_bash
from .event_bus import DelegationBus, bus_exists, get_bus, remove_bus
from .file_ops import read_file, write_file
from .search import google_search

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_on_delegation_start: Callable[[str], None] | None = None


def set_delegation_start_callback(cb: Callable[[str], None] | None) -> None:
    global _on_delegation_start
    _on_delegation_start = cb


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
    granted_tools = config.delegate_allowed_tools

    if isinstance(raw_tools, set):
        granted_tools = raw_tools.intersection(config.delegate_allowed_tools)
    elif raw_tools == "all":
        granted_tools = config.delegate_allowed_tools
    elif isinstance(raw_tools, str):
        requested = {tool.strip() for tool in raw_tools.split(",") if tool.strip()}
        granted_tools = requested.intersection(config.delegate_allowed_tools)

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


def _create_sub_agent(
    task: dict[str, Any], provider: LLMProvider | None = None
) -> tuple[ChatAgent, str]:
    from ..index import get_provider

    registry = _build_tool_registry(task["tools"])
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
                agent, prompt = _create_sub_agent(task, provider=provider)
                result = await asyncio.wait_for(
                    agent.execute(prompt), timeout=task["timeout"]
                )
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Sub-agent '{task_id}' completed in {duration:.2f}s "
                    f"(attempt {attempt + 1})"
                )

                output = result.text or "(No text output returned)"
                if len(output) > config.delegate_max_output_chars:
                    output = (
                        output[: config.delegate_max_output_chars]
                        + "\n\n[Output truncated...]"
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


def _inject_dependency_context(
    task: dict[str, Any], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not task["depends_on"]:
        return task
    parts = [
        f"--- Output from '{d}' ---\n{results[d]['output']}"
        for d in task["depends_on"]
        if results.get(d, {}).get("status") == TaskStatus.COMPLETED
    ]
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
    from ..index import get_provider

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
            bus.update_task_state(
                task_id,
                {"status": "SKIPPED", "agent_label": agent_label, "duration": 0.0},
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
                "started_at": started_at,
                "timeout": task["timeout"],
                "anomaly_fired": False,
            },
        )
        enriched = _inject_dependency_context(task, results)
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
            bus.update_task_state(
                task_id,
                {
                    "status": "DONE",
                    "agent_label": agent_label,
                    "duration": result["duration"],
                },
            )
            bus.publish(
                TaskCompleted(
                    job_id=job_id,
                    task_id=task_id,
                    duration=result["duration"],
                    output=result.get("output", ""),
                    model=result.get("model", ""),
                )
            )
        elif result["status"] == TaskStatus.TIMEOUT:
            bus.update_task_state(
                task_id,
                {
                    "status": "TIMEOUT",
                    "agent_label": agent_label,
                    "duration": result["duration"],
                },
            )
            bus.publish(
                TaskTimedOut(
                    job_id=job_id, task_id=task_id, duration=result["duration"]
                )
            )
        else:
            bus.update_task_state(
                task_id,
                {
                    "status": "FAILED",
                    "agent_label": agent_label,
                    "duration": result["duration"],
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
    results = await _execute_dag(tasks, semaphore, bus, job_id, parallel_limit)
    total_wall = time.perf_counter() - start_wall
    if heartbeat_task is not None:
        heartbeat_task.cancel()
    bus.publish(MilestoneCompleted(job_id, milestone, results, total_wall))
    bus.close()
    remove_bus(job_id)


async def delegate_tasks(milestone: str, tasks: str, max_parallel: int = 3) -> str:
    """Orchestrates background sub-agents to complete a complex milestone.

    Args:
        milestone: A brief label for the overall objective.
        tasks: A JSON string representing a list of task objects.
            EACH task object MUST include:
            - id: A unique string identifier (e.g., "t1", "analyze_file").
            - task: Detailed instructions for the sub-agent.
            - agent: (Optional) Role from the roster (e.g., "scout", "code_writer").
            - depends_on: (Optional) List of task IDs to wait for.
            Example: '[{"id": "t1", "agent": "scout", "task": "map src/app.py"}]'
        max_parallel: Max number of agents to run at once (Default: 3).

    Returns:
        A job confirmation message with a JOB_ID.
    """
    logger.info(f"Tool: delegate_tasks -- milestone='{milestone}'")
    try:
        validated_tasks = _validate_tasks(tasks)
        parallel_cap = max(1, min(max_parallel, config.delegate_max_hard_cap))
        semaphore = asyncio.Semaphore(parallel_cap)
        job_id = str(uuid.uuid4())[:DELEGATE_JOB_ID_LEN]

        bus = get_bus(job_id, milestone)
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

        task = asyncio.create_task(
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
        plan.append(
            "\nResults will be automatically posted back here when the "
            "milestone is complete."
        )
        plan.append(
            f"You can also use `check_delegation(job_id='{job_id}')` "
            "for a live non-blocking snapshot."
        )
        return "\n".join(plan)
    except Exception as e:
        logger.error(f"Tool Failure: delegate_tasks: {e}")
        return f"Error: {e}"


async def check_delegation(job_id: str) -> str:
    """Non-blocking snapshot of the current job state.

    IMPORTANT: Do not call this immediately after starting a delegation.
    It takes a few seconds for agents to initialize and report their first
    status. Wait for at least one heartbeat or progress update.

    Returns immediately with a Markdown table showing every task's current
    status, agent, and elapsed / total duration. Never blocks the caller.
    """
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
