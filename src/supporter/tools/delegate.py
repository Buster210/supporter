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
from .bash import execute_bash
from .delegation_capsule import (
    create_capsule,
    mark_capsule_cancelled,
    mark_capsule_completed,
    mark_task_completed,
    mark_task_failed,
    mark_task_skipped,
    mark_task_started,
    mark_task_timed_out,
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
from .file_ops import read_file, write_file
from .search import google_search

_BACKGROUND: set[asyncio.Task[Any]] = set()
_JOBS: dict[str, asyncio.Task[Any]] = {}
_ON_START: Callable[[str], None] | None = None


def set_delegation_start_callback(cb: Callable[[str], None] | None) -> None:
    global _ON_START
    _ON_START = cb


def _resolve_agent_profile(task: dict[str, Any]) -> dict[str, Any]:
    name = task.get("agent")
    if name and name != "custom" and name in config.delegate_agent_roster:
        profile = config.delegate_agent_roster[name].copy()
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
    t: dict[str, Any], idx: int, seen: set[str]
) -> dict[str, Any]:
    if not isinstance(t, dict):
        raise ValueError(f"Task at index {idx} must be an object")

    tid = t.get("id")
    if not isinstance(tid, str) or not tid:
        raise ValueError(f"Task at index {idx} is missing a valid string 'id'")
    if tid in seen:
        raise ValueError(f"Duplicate task ID detected: {tid}")
    seen.add(tid)

    desc = t.get("task")
    if not isinstance(desc, str) or not desc:
        raise ValueError(f"Task '{tid}' is missing a valid string 'task' description")

    profile = _resolve_agent_profile(t)
    raw_tools = profile.get("tools") or t.get("tools", "all")
    allowed = config.delegate_allowed_tools

    if isinstance(raw_tools, set):
        tools = raw_tools & allowed
    elif raw_tools == "all":
        tools = allowed
    elif isinstance(raw_tools, str):
        requested = {s.strip() for s in raw_tools.split(",") if s.strip()}
        tools = requested & allowed
    else:
        tools = allowed

    raw_to = t.get("timeout")
    timeout = config.delegate_default_timeout
    if isinstance(raw_to, (int, float)):
        timeout = int(min(max(1, raw_to), config.delegate_max_timeout))

    raw_retries = t.get("retry", 0)
    retries = 0
    if isinstance(raw_retries, (int, float)):
        retries = min(max(0, int(raw_retries)), config.delegate_max_retries)

    deps = t.get("depends_on", [])
    if isinstance(deps, str):
        deps = [d.strip() for d in deps.split(",") if d.strip()]
    elif not isinstance(deps, list):
        deps = []

    cmds = t.get("pre_approved_commands", [])
    if not isinstance(cmds, list) or not all(isinstance(c, str) for c in cmds):
        cmds = []

    return {
        "id": tid,
        "task": desc,
        "agent": t.get("agent"),
        "persona": profile["persona"],
        "tools": tools,
        "model": profile.get("model") or config.gemini_model,
        "context": t.get("context") or "",
        "timeout": timeout,
        "max_retries": retries,
        "depends_on": deps,
        "pre_approved_commands": cmds,
        "tolerate_failures": bool(t.get("tolerate_failures", False)),
    }


def _validate_tasks(raw: str) -> list[dict[str, Any]]:
    try:
        tasks = json.loads(raw)
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

    seen: set[str] = set()
    validated = [_validate_single_task(t, i, seen) for i, t in enumerate(tasks)]
    ids = {t["id"] for t in validated}

    for t in validated:
        for dep in t["depends_on"]:
            if dep not in ids:
                raise ValueError(
                    f"Task '{t['id']}' depends on '{dep}', which does not exist"
                )
        if t["id"] in t["depends_on"]:
            raise ValueError(f"Task '{t['id']}' cannot depend on itself")

    _detect_cycles(validated)
    return validated


def _detect_cycles(tasks: list[dict[str, Any]]) -> None:
    degrees = {t["id"]: len(t["depends_on"]) for t in tasks}
    adj: dict[str, list[str]] = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t["depends_on"]:
            adj[dep].append(t["id"])

    queue = deque(tid for tid, d in degrees.items() if d == 0)
    visited = 0
    while queue:
        curr = queue.popleft()
        visited += 1
        for neighbor in adj[curr]:
            degrees[neighbor] -= 1
            if degrees[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(tasks):
        raise ValueError("Dependency cycle detected in task graph")


def _build_tool_registry(tool_names: set[str]) -> dict[str, Callable[..., Any]]:
    all_tools: dict[str, Callable[..., Any]] = {
        "read_file": read_file,
        "write_file": write_file,
        "execute_bash": execute_bash,
        "google_search": google_search,
    }
    return {n: f for n, f in all_tools.items() if n in tool_names}


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
    sem: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    retries = task.get("max_retries", 0)
    delays = DELEGATE_RETRY_BACKOFF
    last: dict[str, Any] = {}
    tid = task["id"]

    for attempt in range(retries + 1):
        async with sem:
            start = time.perf_counter()
            if attempt > 0:
                delay = delays[min(attempt - 1, len(delays) - 1)]
                bus.publish(
                    TaskRetrying(
                        job_id=job_id,
                        task_id=tid,
                        attempt=attempt + 1,
                        reason=last.get("output", ""),
                    )
                )
                await asyncio.sleep(delay)

            try:
                agent, prompt = _create_sub_agent(task, provider=provider)
                res = await asyncio.wait_for(
                    agent.execute(prompt), timeout=task["timeout"]
                )
                duration = time.perf_counter() - start
                output = res.text or "(no output)"
                if len(output) > config.delegate_max_output_chars:
                    output = (
                        output[: config.delegate_max_output_chars] + "\n\n[truncated]"
                    )

                return {
                    "id": tid,
                    "status": TaskStatus.COMPLETED,
                    "output": output,
                    "model": res.model,
                    "duration": duration,
                    "tokens": res.usage,
                }
            except TimeoutError:
                return {
                    "id": tid,
                    "status": TaskStatus.TIMEOUT,
                    "output": (
                        f"Error: Task exceeded execution limit of {task['timeout']}s"
                    ),
                    "duration": time.perf_counter() - start,
                }
            except Exception as e:
                last = {
                    "id": tid,
                    "status": TaskStatus.ERROR,
                    "output": f"Error [{type(e).__name__}]: {e}",
                    "duration": time.perf_counter() - start,
                }
                if attempt >= retries:
                    return last
    return last


def _inject_dependency_context(
    task: dict[str, Any], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if not task["depends_on"]:
        return task
    parts = []
    for dep_id in task["depends_on"]:
        res = results.get(dep_id)
        if not res:
            continue
        label = (
            f"'{dep_id}'"
            if res["status"] == TaskStatus.COMPLETED
            else f"'{dep_id}' [{str(res['status']).upper()}]"
        )
        parts.append(f"--- Output from {label} ---\n{res.get('output', '')}")

    if not parts:
        return task
    enriched = task.copy()
    ctx = "DEPENDENCY OUTPUTS:\n" + "\n\n".join(parts)
    enriched["context"] = (
        f"{enriched['context']}\n\n{ctx}" if enriched["context"] else ctx
    )
    return enriched


def _should_skip(
    task: dict[str, Any], results: dict[str, dict[str, Any]]
) -> str | None:
    if task.get("tolerate_failures"):
        return None
    for dep_id in task["depends_on"]:
        res = results.get(dep_id)
        if res and res["status"] != TaskStatus.COMPLETED:
            return f"Dependency '{dep_id}' {res['status']}"
    return None


def _compute_priority(tid: str, tasks: list[dict[str, Any]]) -> int:
    adj: dict[str, list[str]] = {t["id"]: [] for t in tasks}
    for t in tasks:
        for dep in t["depends_on"]:
            adj[dep].append(t["id"])

    visited, queue, count = set(), deque([tid]), 0
    while queue:
        curr = queue.popleft()
        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                count += 1
                queue.append(neighbor)
    return count


async def _execute_dag(
    tasks: list[dict[str, Any]],
    sem: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    from ..index import get_provider

    results: dict[str, dict[str, Any]] = {}
    done: dict[str, asyncio.Event] = {t["id"]: asyncio.Event() for t in tasks}
    priorities = {t["id"]: _compute_priority(t["id"], tasks) for t in tasks}
    pools: dict[str, LLMProvider] = {}

    async def _gate(task: dict[str, Any]) -> None:
        for dep_id in task["depends_on"]:
            await done[dep_id].wait()

        if priorities[task["id"]] < max(priorities.values(), default=0):
            await asyncio.sleep(0)

        tid, label = task["id"], task.get("agent") or "custom"
        skip = _should_skip(task, results)
        if skip:
            results[tid] = {
                "id": tid,
                "status": TaskStatus.SKIPPED,
                "output": f"Skipped: {skip}",
                "duration": 0.0,
            }
            await mark_task_skipped(job_id, tid, skip)
            bus.publish(TaskSkipped(job_id=job_id, task_id=tid, reason=skip))
            done[tid].set()
            return

        start = time.monotonic()
        await mark_task_started(job_id, tid)
        bus.publish(
            TaskStarted(
                job_id=job_id,
                task_id=tid,
                agent_label=label,
                started_at=start,
                timeout=task["timeout"],
            )
        )

        model = task["model"]
        if model not in pools:
            pools[model] = get_provider(shared=False, model_name=model, pool_size=limit)

        res = await _run_sub_agent(
            _inject_dependency_context(task, results),
            sem,
            bus,
            job_id,
            provider=pools[model],
        )
        results[tid], duration = res, res["duration"]

        if res["status"] == TaskStatus.COMPLETED:
            await mark_task_completed(job_id, tid, res.get("output", ""), duration)
            bus.update_task_state(
                tid, {"status": "DONE", "agent_label": label, "duration": duration}
            )
            bus.publish(
                TaskCompleted(
                    job_id=job_id,
                    task_id=tid,
                    duration=duration,
                    output=res.get("output", ""),
                    model=res.get("model", ""),
                )
            )
        elif res["status"] == TaskStatus.TIMEOUT:
            await mark_task_timed_out(job_id, tid, "Task timed out", duration)
            bus.update_task_state(
                tid, {"status": "TIMEOUT", "agent_label": label, "duration": duration}
            )
            bus.publish(TaskTimedOut(job_id=job_id, task_id=tid, duration=duration))
        else:
            out = res.get("output", "Unknown error")
            await mark_task_failed(job_id, tid, out, duration)
            bus.update_task_state(
                tid, {"status": "FAILED", "agent_label": label, "duration": duration}
            )
            bus.publish(
                TaskFailed(job_id=job_id, task_id=tid, duration=duration, error=out)
            )

        done[tid].set()

    await asyncio.gather(*[_gate(t) for t in tasks])
    return [results[t["id"]] for t in tasks if t["id"] in results]


async def _run_heartbeat(
    bus: DelegationBus, job_id: str, interval: int = DELEGATE_HEARTBEAT_INTERVAL
) -> None:
    while bus_exists(job_id):
        await asyncio.sleep(interval)
        if not bus_exists(job_id):
            return

        now, snap = time.monotonic(), bus.get_snapshot()
        bus.publish(
            HeartbeatTick(job_id=job_id, milestone=bus.milestone, snapshot=snap)
        )

        for tid, s in snap.items():
            if s.get("status") != "RUNNING":
                continue
            start, to = s.get("started_at"), s.get("timeout")
            if (
                start
                and to
                and (now - start) >= DELEGATE_ANOMALY_THRESHOLD * to
                and not s.get("anomaly_fired")
            ):
                bus.publish(
                    TaskAnomaly(
                        job_id=job_id,
                        task_id=tid,
                        agent_label=s.get("agent_label", "?"),
                        elapsed_seconds=now - start,
                        timeout=to,
                    )
                )
                bus.update_task_state(tid, {**s, "anomaly_fired": True})


async def _run_milestone(
    milestone: str,
    tasks: list[dict[str, Any]],
    sem: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    limit: int,
    hb: asyncio.Task[None] | None = None,
) -> None:
    start = time.perf_counter()
    try:
        results = await _execute_dag(tasks, sem, bus, job_id, limit)
        wall = time.perf_counter() - start
        bus.publish(MilestoneCompleted(job_id, milestone, results, wall))
    except asyncio.CancelledError:
        wall = time.perf_counter() - start
        bus.publish(MilestoneCancelled(job_id, milestone, wall))
        raise
    finally:
        await mark_capsule_completed(job_id)
        if hb:
            hb.cancel()
        bus.close()
        remove_bus(job_id)
        _JOBS.pop(job_id, None)


async def delegate_tasks(
    milestone: str,
    tasks: str,
    max_parallel: int = 3,
    notify_per_task: bool = True,
) -> str:
    """Orchestrates background sub-agents to complete a complex milestone."""
    try:
        validated = _validate_tasks(tasks)
        limit = max(1, min(max_parallel, config.delegate_max_hard_cap))
        sem, job_id = asyncio.Semaphore(limit), str(uuid.uuid4())[:DELEGATE_JOB_ID_LEN]

        bus = get_bus(job_id, milestone)
        bus.notify_per_task = notify_per_task
        if _ON_START:
            _ON_START(job_id)

        await create_capsule(
            job_id=job_id, milestone=milestone, tasks=validated, parallel_cap=limit
        )
        bus.publish(
            MilestoneStarted(
                job_id=job_id,
                milestone=milestone,
                task_ids=[t["id"] for t in validated],
                parallel_cap=limit,
            )
        )

        hb = asyncio.create_task(_run_heartbeat(bus, job_id))
        _BACKGROUND.add(hb)
        hb.add_done_callback(_BACKGROUND.discard)

        task = asyncio.create_task(
            _run_milestone(milestone, validated, sem, bus, job_id, limit, hb)
        )
        _JOBS[job_id] = task
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)

        plan = [
            f"Delegation started for milestone: **{milestone}**",
            f"Job ID: `{job_id}`",
            "\n| # | Task ID | Agent | Dependencies |",
            "|---|---------|-------|--------------|",
        ]
        for i, t in enumerate(validated, 1):
            deps = ", ".join(t["depends_on"]) or "none"
            plan.append(
                f"| {i} | {t['id']} | {t['agent'] or 'custom'} | after: {deps} |"
            )
        plan.append(f"\nSub-agents are running with parallel limit: {limit}")
        plan.append("\nResults will be automatically posted back here when complete.")
        plan.append(
            f"You can also use `check_delegation(job_id='{job_id}')` for a snapshot."
        )
        return "\n".join(plan)
    except Exception as e:
        logger.error(f"delegate_tasks failed: {e}")
        return f"Error: {e}"


async def check_delegation(job_id: str) -> str:
    """Non-blocking snapshot of the current job state."""
    if not bus_exists(job_id):
        return f"Job `{job_id}` is unknown or already complete."
    bus = get_bus(job_id)
    snap = bus.get_snapshot()
    if not snap:
        return f"Job `{job_id}` has no tasks tracked yet."

    rows = ["| Task ID | Agent | Status | Elapsed | Timeout |", "|---|---|---|---|---|"]
    for tid, s in snap.items():
        start = s.get("started_at")
        elapsed = f"{time.monotonic() - start:.1f}s" if start else "-"
        row = (
            f"| {tid} | {s.get('agent_label', 'custom')} | "
            f"{s.get('status', 'PENDING')} | "
            f"{elapsed} | "
            f"{s.get('timeout', 0)}s |"
        )
        rows.append(row)
    return "\n".join(rows)


async def cancel_delegation(job_id: str) -> str:
    """Cancels an active delegation job."""
    if job_id not in _JOBS:
        if bus_exists(job_id):
            remove_bus(job_id)
        return f"Job '{job_id}' unknown or complete."

    _JOBS.pop(job_id).cancel()
    await mark_capsule_cancelled(job_id)
    return f"Cancellation requested for '{job_id}'."


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
        if task_tokens := r.get("tokens", {}).get("total_tokens"):
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
        if token_total := r.get("tokens", {}).get("total_tokens"):
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
