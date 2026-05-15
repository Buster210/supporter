import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

from ...config import DELEGATE_JOB_ID_LEN, config
from ...logger import logger
from ...types import MilestoneStarted
from ..base import ToolError
from .bus import bus_exists, get_bus
from .capsule import create_capsule
from .capsule_query import serialize_capsule_result as _serialize_capsule_result
from .scheduler import (
    BACKGROUND_TASKS,
    JOB_TASKS,
    run_heartbeat,
    run_milestone,
)
from .validation import validate_tasks

_on_delegation_start: Callable[[str], None] | None = None

_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None


def _get_global_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SEMAPHORE
    if _GLOBAL_SEMAPHORE is None:
        _GLOBAL_SEMAPHORE = asyncio.Semaphore(config.delegate_max_hard_cap)
    return _GLOBAL_SEMAPHORE


def reset_global_semaphore() -> None:
    global _GLOBAL_SEMAPHORE
    _GLOBAL_SEMAPHORE = None


def set_delegation_start_callback(cb: Callable[[str], None] | None) -> None:
    global _on_delegation_start
    _on_delegation_start = cb


def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    return _serialize_capsule_result(job_id)


async def delegate_tasks(
    milestone: str,
    tasks: str,
    max_parallel: int = 3,
    notify_per_task: bool = True,
) -> str:
    """Orchestrates background sub-agents to complete a complex milestone.

    Args:
        milestone: A brief label for the overall objective.
        tasks: A JSON string representing a list of task objects.
            EACH task object MUST include:
            - id: A unique string identifier (e.g., "t1", "analyze_file").
            - task: Detailed instructions for the sub-agent.
            - agent: (Optional) Role from the roster (e.g., "scout", "code_writer").
            - depends_on: (Optional) List of task IDs to wait for.
            - tolerate_failures: (Optional) If true, run even when deps failed/
              timed-out/skipped; their outputs are injected with a status tag.
            Example: '[{"id": "t1", "agent": "scout", "task": "map src/app.py"}]'
        max_parallel: Max number of agents to run at once (Default: 3).
        notify_per_task: If true, compact completed/failed task signals are
            fed back to the orchestrator so it can query details and adapt
            while siblings still run. Default true.

    Returns:
        A job confirmation message with a JOB_ID.

    Raises:
        ValueError: Cycle or invalid JSON.
        KeyError: Missing id/agent/task in task object.
    """
    logger.info(f"Tool: delegate_tasks -- milestone='{milestone}'")
    try:
        validated_tasks = validate_tasks(tasks)
        parallel_cap = max(1, min(max_parallel, config.delegate_max_hard_cap))
        semaphore = _get_global_semaphore()
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
        await create_capsule(job_id, milestone, validated_tasks, parallel_cap)
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

        hb_task = asyncio.create_task(run_heartbeat(bus, job_id))
        BACKGROUND_TASKS.add(hb_task)
        hb_task.add_done_callback(BACKGROUND_TASKS.discard)

        milestone_task = asyncio.create_task(
            run_milestone(
                milestone,
                validated_tasks,
                semaphore,
                bus,
                job_id,
                parallel_cap,
                hb_task,
            )
        )
        JOB_TASKS[job_id] = milestone_task
        BACKGROUND_TASKS.add(milestone_task)
        milestone_task.add_done_callback(BACKGROUND_TASKS.discard)

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
    task = JOB_TASKS.get(job_id)
    if task is None or task.done():
        return f"Job `{job_id}` is unknown or already complete."

    task.cancel()
    return f"Cancellation requested for job `{job_id}`."
