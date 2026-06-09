import asyncio
import time
from typing import Any

from ...config import DELEGATE_ANOMALY_THRESHOLD, DELEGATE_HEARTBEAT_INTERVAL, config
from ...decision_log import log_decision
from ...logger import logger
from ...prompts import DELEGATION_REPAIR_REQUEST, DELEGATION_RESULT_CONTRACT
from ...types import (
    HeartbeatTick,
    MilestoneCancelled,
    MilestoneCompleted,
    MilestoneStarted,
    TaskAnomaly,
    TaskCompleted,
    TaskFailed,
    TaskSkipped,
    TaskStarted,
    TaskStatus,
    TaskTimedOut,
)
from .agents import delegate_allowed_tool_names, run_sub_agent
from .bus import DelegationBus, bus_exists, get_bus, remove_bus
from .capsule import (
    extract_task_capsule_fields,
    load_capsule,
    mark_capsule_cancelled,
    mark_capsule_completed,
    mark_task_completed,
    mark_task_failed,
    mark_task_skipped,
    mark_task_started,
    mark_task_timed_out,
    status_value,
    validate_delegation_payload,
)
from .metrics import subscribe_metrics
from .project_memory import load_project_memory, memory_context_block, record_learnings
from .qa_gate import run_qa_gate

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


def _inject_memory_context(task: dict[str, Any], memory_block: str) -> dict[str, Any]:
    """Inject project memory block into task context. No-op when block is empty."""
    if not memory_block:
        return task
    enriched = task.copy()
    base = enriched["context"]
    enriched["context"] = f"{base}\n\n{memory_block}" if base else memory_block
    return enriched


async def _record_milestone_learnings(job_id: str) -> None:
    """Extract a completed capsule's key_findings into project memory.

    Never raises: a memory failure must not break milestone completion.
    """
    try:
        capsule = load_capsule(job_id)
        synthesis = capsule.get("synthesis", {})
        if not isinstance(synthesis, dict):
            return
        key_findings = synthesis.get("key_findings", [])
        if not isinstance(key_findings, list):
            return
        insights = [str(kf) for kf in key_findings if isinstance(kf, str)]
        await record_learnings(insights, job_id)
    except Exception as exc:
        logger.warning(f"Failed to record project memory for job {job_id}: {exc}")


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


async def _repair_or_rerequest(
    task: dict[str, Any],
    result: dict[str, Any],
    global_semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> dict[str, Any]:
    """One bounded re-request when a delegated result is off-schema (SPEC §8).

    A malformed structured block is repaired by asking the SAME agent to re-emit
    just the JSON block exactly once. The re-request never turns a COMPLETED task
    into a failure and never crashes the scheduler: on any miss the original
    result is kept untouched.
    """
    if (
        not config.delegate_result_repair
        or result.get("status") != TaskStatus.COMPLETED
        or not task.get("result_contract", True)
    ):
        return result
    if validate_delegation_payload(result.get("output", "")):
        return result

    log_decision(
        site="scheduler.repair_retry",
        chosen="re-request",
        options=("keep_original", "re-request"),
        reason="delegated result failed schema validation",
        correlation_id=f"{job_id}:{task['id']}",
    )
    truncated = result.get("output", "")[: config.delegate_max_output_chars]
    followup = task.copy()
    followup["id"] = f"{task['id']}__repair"
    followup["task"] = (
        DELEGATION_REPAIR_REQUEST + truncated + DELEGATION_RESULT_CONTRACT
    )
    followup["max_retries"] = 0
    followup["result_contract"] = True

    try:
        repaired = await run_sub_agent(followup, global_semaphore, bus, job_id)
    except Exception as exc:
        logger.warning(
            f"capsule repair failed, using original for task {task['id']}: {exc}"
        )
        return result

    if repaired.get("status") == TaskStatus.COMPLETED and validate_delegation_payload(
        repaired.get("output", "")
    ):
        result["output"] = repaired.get("output", "")
        logger.info(f"capsule repaired for task {task['id']}")
    else:
        logger.warning(f"capsule repair failed, using original for task {task['id']}")
    return result


async def _execute_dag(
    tasks: list[dict[str, Any]],
    job_semaphore: asyncio.Semaphore,
    global_semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    seed_results: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = dict(seed_results) if seed_results else {}
    task_done: dict[str, asyncio.Event] = {t["id"]: asyncio.Event() for t in tasks}
    # Load project memory once for all tasks
    memory = await load_project_memory()
    memory_block = memory_context_block(memory)
    # Seeded (already-settled) tasks resolve immediately so dependents' gates open
    # and the settled output is available as context -- even when only unfinished
    # tasks are passed in `tasks` (the resume path) and a dep lives only in seeds.
    for seeded_id in results:
        task_done.setdefault(seeded_id, asyncio.Event()).set()

    async def _run_with_gate(task: dict[str, Any]) -> None:
        task_id = task["id"]
        if task_id in results:
            # Settled in a prior run (seeded on resume) -- never re-execute.
            task_done[task_id].set()
            return

        for dep_id in task["depends_on"]:
            await task_done[dep_id].wait()

        agent_label = task.get("agent") or "custom"

        skip_reason = _should_skip(task, results)
        if skip_reason:
            log_decision(
                site="scheduler.skip",
                chosen="skip",
                options=("run", "skip"),
                reason=skip_reason,
                correlation_id=f"{job_id}:{task_id}",
            )
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
        enriched = _inject_memory_context(enriched, memory_block)
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

        async with job_semaphore:
            result = await run_sub_agent(enriched, global_semaphore, bus, job_id)
            result = await run_qa_gate(enriched, result, global_semaphore, bus, job_id)
            result = await _repair_or_rerequest(
                enriched, result, global_semaphore, bus, job_id
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
                    tokens=result.get("tokens") or {},
                    step_count=int(result.get("step_count", 0)),
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
    ordered = [results[t["id"]] for t in tasks if t["id"] in results]
    seen = {t["id"] for t in tasks}
    ordered.extend(v for k, v in results.items() if k not in seen)
    return ordered


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
                    log_decision(
                        site="scheduler.anomaly",
                        chosen="flag_anomaly",
                        options=("wait", "flag_anomaly"),
                        reason=(
                            f"elapsed {elapsed:.0f}s >= "
                            f"{DELEGATE_ANOMALY_THRESHOLD:.0%} of "
                            f"timeout {task_timeout:.0f}s"
                        ),
                        correlation_id=f"{job_id}:{task_id}",
                    )
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
    job_semaphore: asyncio.Semaphore,
    global_semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    heartbeat_task: asyncio.Task[None] | None = None,
    seed_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    start_wall = time.perf_counter()
    metrics_task = subscribe_metrics(bus, job_id)
    try:
        results = await _execute_dag(
            tasks, job_semaphore, global_semaphore, bus, job_id, seed_results
        )
        total_wall = time.perf_counter() - start_wall
        await mark_capsule_completed(job_id)
        await _record_milestone_learnings(job_id)
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
        await asyncio.gather(metrics_task, return_exceptions=True)
        remove_bus(job_id)
        JOB_TASKS.pop(job_id, None)


def _task_to_seed_result(task: dict[str, Any]) -> dict[str, Any]:
    """Convert a capsule task record to a result record for seeding."""
    status = task.get("status", TaskStatus.PENDING.value)
    return {
        "id": task["id"],
        "status": status,
        "output": task.get("output", ""),
        "duration": float(task.get("duration", 0.0)),
        "model": task.get("model", ""),
        "tokens": task.get("tokens", {}),
        "step_count": int(task.get("step_count", 0)),
    }


_BUS_STATUS_BY_TERMINAL = {
    TaskStatus.COMPLETED.value: "DONE",
    TaskStatus.TIMEOUT.value: "TIMEOUT",
    TaskStatus.ERROR.value: "FAILED",
    TaskStatus.SKIPPED.value: "SKIPPED",
}


def _register_capsule_tasks_on_bus(
    bus: DelegationBus,
    tasks_by_id: dict[str, dict[str, Any]],
    terminal_statuses: set[str],
) -> None:
    """Seed the bus snapshot with each capsule task's last-known state."""
    for task_id, task_record in tasks_by_id.items():
        if not isinstance(task_record, dict):
            continue
        task_status = status_value(task_record.get("status", ""))
        if task_status in terminal_statuses:
            bus.update_task_state(
                task_id,
                {
                    "status": _BUS_STATUS_BY_TERMINAL[task_status],
                    "agent_label": task_record.get("agent", "custom"),
                    "task_goal": task_record.get("goal", ""),
                    "duration": float(task_record.get("duration", 0.0)),
                    "summary": str(task_record.get("summary", "")),
                },
            )
        else:
            bus.update_task_state(
                task_id,
                {
                    "status": "PENDING",
                    "agent_label": task_record.get("agent", "custom"),
                    "task_goal": task_record.get("goal", ""),
                    "duration": 0.0,
                },
            )


async def find_resumable_jobs() -> list[str]:
    """Find job IDs for interrupted milestones that can be auto-resumed.

    Scans capsule files and returns job_ids whose effective_status is
    "interrupted_by_restart".
    """
    from .capsule import effective_status, load_capsule_safe
    from .capsule_query import capsule_files

    job_ids: list[str] = []
    for path in capsule_files():
        try:
            capsule = load_capsule_safe(path.stem)
            if effective_status(capsule) == "interrupted_by_restart":
                job_ids.append(str(capsule.get("job_id", "")))
        except Exception as exc:
            logger.debug(
                f"Skipping corrupted capsule during resume scan "
                f"[path={path}, error={type(exc).__name__}]"
            )
    return [jid for jid in job_ids if jid]


async def resume_milestone(job_id: str) -> bool:
    """Resume an interrupted milestone from its persisted capsule.

    Loads the capsule, identifies unfinished tasks, rebuilds seed_results
    from settled tasks, and runs only the unfinished tasks through the
    existing gate/QA/repair pipeline.

    Returns True if the milestone was resumed, False if it was skipped
    (e.g., already has a live entry in JOB_TASKS).
    """
    from .capsule import load_capsule

    # Skip if already running -- a live in-process task or a registered bus.
    existing = JOB_TASKS.get(job_id)
    if existing is not None and not existing.done():
        logger.info(f"Job {job_id} already running, skipping resume")
        return False
    if bus_exists(job_id):
        logger.info(f"Job {job_id} already has live bus, skipping resume")
        return False

    capsule = load_capsule(job_id)
    milestone = str(capsule.get("milestone", ""))
    parallel_cap = int(capsule.get("parallel_cap", config.delegate_max_hard_cap))
    tasks_by_id: dict[str, dict[str, Any]] = capsule.get("tasks", {})

    if not tasks_by_id:
        logger.warning(f"Job {job_id} has no tasks in capsule, skipping")
        return False

    # Classify tasks by status
    terminal_statuses = {
        TaskStatus.COMPLETED.value,
        TaskStatus.TIMEOUT.value,
        TaskStatus.ERROR.value,
        TaskStatus.SKIPPED.value,
    }
    settled: dict[str, dict[str, Any]] = {}
    unfinished: list[dict[str, Any]] = []

    for task_id, task_record in tasks_by_id.items():
        if not isinstance(task_record, dict):
            continue
        task_status = status_value(task_record.get("status", ""))
        if task_status in terminal_statuses:
            settled[task_id] = _task_to_seed_result(task_record)
        else:
            # Reconstruct unfinished task dict
            unfinished.append(
                {
                    "id": task_id,
                    "task": task_record.get("goal", ""),
                    "agent": task_record.get("agent") or "custom",
                    "backend": "gemini",
                    "tools": set(delegate_allowed_tool_names(task_record.get("agent"))),
                    "model": task_record.get("model", config.gemini_model),
                    "persona": config.delegate_default_persona,
                    "context": "",
                    "timeout": (
                        task_record.get("timeout") or config.delegate_default_timeout
                    ),
                    "max_retries": 0,
                    "depends_on": list(task_record.get("depends_on", [])),
                    "tolerate_failures": bool(
                        task_record.get("tolerate_failures", False)
                    ),
                }
            )

    # Build seed_results from settled records
    seed_results = settled.copy()

    # The capsule on disk already holds correct per-task terminal state; do not
    # recreate it -- that would reset settled tasks to pending and lose their
    # outputs. The first resumed task's status write refreshes `updated_at`,
    # clearing the interrupted-by-restart flag.
    log_decision(
        site="scheduler.resume_milestone",
        chosen="resume",
        options=("resume", "skip"),
        reason=f"auto-resume interrupted milestone {job_id}",
        correlation_id=job_id,
    )

    # Run only unfinished tasks
    if unfinished:
        # Create the bus only when there is live work; seed its snapshot with
        # every task's last-known state so the resumed run renders in full.
        bus = get_bus(job_id, milestone)
        bus.notify_per_task = True
        _register_capsule_tasks_on_bus(bus, tasks_by_id, terminal_statuses)

        job_semaphore = asyncio.Semaphore(parallel_cap)
        hb_task = asyncio.create_task(run_heartbeat(bus, job_id))
        BACKGROUND_TASKS.add(hb_task)
        hb_task.add_done_callback(BACKGROUND_TASKS.discard)

        milestone_task = asyncio.create_task(
            run_milestone(
                milestone,
                unfinished,
                job_semaphore,
                _get_global_semaphore(),
                bus,
                job_id,
                hb_task,
                seed_results,
            )
        )
        JOB_TASKS[job_id] = milestone_task
        BACKGROUND_TASKS.add(milestone_task)
        milestone_task.add_done_callback(BACKGROUND_TASKS.discard)

        # Publish MilestoneStarted for the resumed job
        bus.publish(
            MilestoneStarted(
                job_id=job_id,
                milestone=milestone,
                task_ids=[t["id"] for t in unfinished],
                parallel_cap=parallel_cap,
            )
        )
    else:
        # All tasks already settled - just mark completed
        await mark_capsule_completed(job_id)
        await _record_milestone_learnings(job_id)

    return True


async def resume_interrupted_jobs() -> list[str]:
    """Find and resume all interrupted jobs.

    Auto-resume entry point - finds resumable jobs and resumes each.
    Logs the decision via decision_log.log_decision.

    Returns list of job_ids that were resumed.
    """
    resumed: list[str] = []
    for job_id in await find_resumable_jobs():
        if await resume_milestone(job_id):
            resumed.append(job_id)
    if resumed:
        log_decision(
            site="scheduler.resume_interrupted_jobs",
            chosen="resume",
            options=("resume", "skip"),
            reason=f"auto-resumed {len(resumed)} interrupted milestone(s)",
            correlation_id="startup",
        )
    return resumed


# Module-level global semaphore for concurrency control
_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for delegation concurrency control."""
    global _GLOBAL_SEMAPHORE
    if _GLOBAL_SEMAPHORE is None:
        _GLOBAL_SEMAPHORE = asyncio.Semaphore(config.delegate_max_hard_cap)
    return _GLOBAL_SEMAPHORE
