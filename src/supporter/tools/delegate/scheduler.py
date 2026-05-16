import asyncio
import time
from typing import Any

from ...config import DELEGATE_ANOMALY_THRESHOLD, DELEGATE_HEARTBEAT_INTERVAL
from ...logger import logger
from ...types import (
    HeartbeatTick,
    MilestoneCancelled,
    MilestoneCompleted,
    TaskAnomaly,
    TaskCompleted,
    TaskFailed,
    TaskSkipped,
    TaskStarted,
    TaskStatus,
    TaskTimedOut,
)
from .agents import run_sub_agent
from .bus import DelegationBus, bus_exists, remove_bus
from .capsule import (
    extract_task_capsule_fields,
    mark_capsule_cancelled,
    mark_capsule_completed,
    mark_task_completed,
    mark_task_failed,
    mark_task_skipped,
    mark_task_started,
    mark_task_timed_out,
)

BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
JOB_TASKS: dict[str, asyncio.Task[Any]] = {}


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


async def _execute_dag(
    tasks: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    parallel_limit: int,
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    task_done: dict[str, asyncio.Event] = {t["id"]: asyncio.Event() for t in tasks}

    async def _run_with_gate(task: dict[str, Any]) -> None:
        for dep_id in task["depends_on"]:
            await task_done[dep_id].wait()

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
            await mark_task_skipped(job_id, task_id, skip_reason)
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
        await mark_task_started(
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

        result = await run_sub_agent(enriched, semaphore, bus, job_id)
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
            await mark_task_completed(
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
            await mark_task_timed_out(
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
            await mark_task_failed(
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


async def run_heartbeat(
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


async def run_milestone(
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
        await mark_capsule_completed(job_id)
        bus.publish(MilestoneCompleted(job_id, milestone, results, total_wall))
    except asyncio.CancelledError:
        total_wall = time.perf_counter() - start_wall
        logger.info(f"Milestone '{milestone}' (job={job_id}) cancelled")
        await mark_capsule_cancelled(job_id)
        bus.publish(MilestoneCancelled(job_id, milestone, total_wall))
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        bus.close()
        remove_bus(job_id)
        JOB_TASKS.pop(job_id, None)
