from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from ..logger import logger


class MessageInjector(Protocol):
    def __call__(self, message: str) -> None: ...


class ProgressUpdater(Protocol):
    async def __call__(self, job_id: str, bus: Any) -> None: ...


def format_completed_task_signal(job_id: str, task_id: str) -> str:
    return (
        "<br/>\n\n"
        f"Delegation task completed — job_id: `{job_id}` | task_id: `{task_id}`\n"
        "\n<br/>"
    )


def format_delegation_progress(job_id: str, bus: Any) -> str:
    rows = []
    for task_id, state in bus.get_snapshot().items():
        status = _display_task_status(str(state.get("status", "PENDING")))
        agent = str(state.get("agent_label", "?"))
        duration = state.get("duration")
        duration_text = ""
        if isinstance(duration, int | float) and duration > 0:
            duration_text = f"{duration:.2f}s"
        rows.append(
            "| "
            + " | ".join(
                [
                    task_id,
                    agent,
                    status,
                    duration_text,
                ]
            )
            + " |"
        )

    table = [
        f"Job `{job_id}`",
        "",
        "| Task | Agent | Status | Time |",
        "| ---- | ----- | ------ | ---- |",
        *rows,
    ]
    return "\n".join(table)


def _display_task_status(status: str) -> str:
    normalized = status.lower()
    if normalized == "running":
        return "working"
    if normalized == "done":
        return "completed"
    return normalized


def _format_task_signal(job_id: str, kind: str, task_id: str, bus: Any) -> str:
    state = bus.get_snapshot().get(task_id, {})
    payload = {
        "job_id": job_id,
        "task_id": task_id,
        "agent": str(state.get("agent_label", "?")),
        "assigned_task": str(state.get("task_goal", "")).strip(),
    }
    status = kind.upper()
    return f"DELEGATION_TASK_{status}: {json.dumps(payload, ensure_ascii=False)}"


class DelegationListener:
    def __init__(
        self,
        inject_message: MessageInjector,
        upsert_progress: ProgressUpdater,
        drop_progress: Callable[[str], None],
    ) -> None:
        self._inject_message = inject_message
        self._upsert_progress = upsert_progress
        self._drop_progress = drop_progress

    async def listen(self, job_id: str) -> None:
        from ..tools.delegate.api import serialize_capsule_result
        from ..tools.delegate.bus import get_bus
        from ..tools.delegate.scheduler import serialize_results
        from ..types import (
            MilestoneCancelled,
            MilestoneCompleted,
            MilestoneStarted,
            TaskAnomaly,
            TaskCompleted,
            TaskFailed,
            TaskSkipped,
            TaskStarted,
            TaskTimedOut,
        )

        try:
            bus = get_bus(job_id)
            queue = bus.subscribe()
            while True:
                event = await queue.get()
                if event is None:
                    break

                if isinstance(event, (MilestoneStarted, TaskStarted)):
                    pass

                elif isinstance(event, TaskAnomaly):
                    msg = (
                        f"AGENT ALERT: Task `{event.task_id}` [{event.agent_label}] "
                        f"has used {event.elapsed_seconds:.0f}s of its "
                        f"{event.timeout:.0f}s limit and may be hung."
                    )
                    self._inject_message(msg)

                elif isinstance(event, TaskCompleted):
                    sys_body = format_completed_task_signal(
                        job_id=job_id, task_id=event.task_id
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "DONE",
                        event.task_id,
                        sys_body=sys_body,
                    )

                elif isinstance(event, TaskFailed):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "FAIL",
                        event.task_id,
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, TaskTimedOut):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "TIMEOUT",
                        event.task_id,
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, TaskSkipped):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "SKIP",
                        event.task_id,
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, MilestoneCompleted):
                    try:
                        payload = serialize_capsule_result(job_id)
                    except Exception as e:
                        logger.warning(
                            "Falling back to legacy delegation result for "
                            f"{job_id} [{type(e).__name__}]: {e}"
                        )
                        has_failures = any(
                            str(result.get("status")) in {"error", "timeout", "skipped"}
                            for result in event.results
                        )
                        payload = serialize_results(
                            event.milestone,
                            event.results,
                            event.total_duration,
                            job_id,
                            status="completed_with_failures"
                            if has_failures
                            else "completed",
                        )
                    self._inject_capsule_result(payload)
                    self._drop_progress(job_id)
                    break

                elif isinstance(event, MilestoneCancelled):
                    try:
                        payload = serialize_capsule_result(job_id)
                    except Exception as e:
                        logger.warning(
                            "Falling back to legacy cancellation result for "
                            f"{job_id} [{type(e).__name__}]: {e}"
                        )
                        payload = {
                            "job_id": job_id,
                            "milestone": event.milestone,
                            "status": "cancelled",
                            "total_duration": round(event.total_duration, 2),
                        }
                    self._inject_capsule_result(payload)
                    self._drop_progress(job_id)
                    break

        except Exception as e:
            logger.error(f"Delegation listener failed for {job_id}: {e}")

    def _inject_capsule_result(self, payload: dict[str, Any]) -> None:
        msg = (
            "DELEGATION_CAPSULE_RESULT (json):\n```json\n"
            f"{json.dumps(payload, indent=2)}\n```"
        )
        self._inject_message(msg)

    async def _emit_task_event(
        self,
        bus: Any,
        job_id: str,
        kind: str,
        task_id: str,
        sys_extra: str = "",
        sys_body: str | None = None,
    ) -> None:
        await self._upsert_progress(job_id, bus)
        if bus.notify_per_task:
            if sys_body:
                message = sys_body
            else:
                message = _format_task_signal(job_id, kind, task_id, bus)
                if sys_extra:
                    message += sys_extra
            self._inject_message(message)


__all__ = [
    "DelegationListener",
    "format_completed_task_signal",
    "format_delegation_progress",
]
