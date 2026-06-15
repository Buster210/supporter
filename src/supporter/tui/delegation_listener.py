from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from ..logger import logger

# Maximum tail length for streaming output (last ~500 chars or last 3 lines)
_OUTPUT_TAIL_MAX_CHARS = 500
_OUTPUT_TAIL_MAX_LINES = 3


def _truncate_output_tail(text: str) -> str:
    """Keep a bounded rolling tail of streamed output (char + line caps)."""
    if len(text) <= _OUTPUT_TAIL_MAX_CHARS:
        return text
    # Prefer the last N lines, then hard-cap to the char budget so a single
    # very long line (no newlines) still can't grow the widget unbounded.
    tail = "\n".join(text.splitlines()[-_OUTPUT_TAIL_MAX_LINES:])
    if len(tail) > _OUTPUT_TAIL_MAX_CHARS:
        tail = tail[-_OUTPUT_TAIL_MAX_CHARS:]
    return tail


class MessageInjector(Protocol):
    def __call__(self, message: str) -> None: ...


class ProgressUpdater(Protocol):
    async def __call__(self, job_id: str, bus: Any) -> None: ...


_KIND_LABELS = {
    "DONE": "completed",
    "FAIL": "failed",
    "TIMEOUT": "timed out",
    "SKIP": "skipped",
}


def format_task_signal(job_id: str, kind: str, task_id: str, bus: Any) -> str:
    state = bus.get_snapshot().get(task_id, {})
    agent = str(state.get("agent_label", "?"))
    goal = str(state.get("task_goal", "")).strip()
    if len(goal) > 80:
        goal = goal[:77] + "..."
    label = _KIND_LABELS.get(kind.upper(), kind.lower())
    detail = f" — {goal}" if goal else ""
    return f"<br/>\n\nDelegation task {label} — `{task_id}` [{agent}]{detail}\n\n<br/>"


def format_delegation_progress(job_id: str, bus: Any) -> str:
    rows = []
    for task_id, state in bus.get_snapshot().items():
        status = _display_task_status(str(state.get("status", "PENDING")))
        agent = str(state.get("agent_label", "?"))
        duration = state.get("duration")
        duration_text = ""
        if isinstance(duration, int | float) and duration > 0:
            duration_text = f"{duration:.2f}s"
        output_tail = state.get("output_tail", "")
        if output_tail:
            # Show tail indicator for running tasks
            duration_text += f" [{output_tail.splitlines()[0][:40]}]"
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


class DelegationListener:
    def __init__(
        self,
        inject_message: MessageInjector,
        upsert_progress: ProgressUpdater,
        drop_progress: Callable[[str], None],
        render_signal: Callable[[str], None],
    ) -> None:
        self._inject_message = inject_message
        self._upsert_progress = upsert_progress
        self._drop_progress = drop_progress
        self._render_signal = render_signal
        # Per-task rolling output tails (bounded)
        self._output_tails: dict[str, str] = {}

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
            TaskOutputChunk,
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

                elif isinstance(event, TaskOutputChunk):
                    # Append to the bounded per-task tail and MERGE it into the
                    # task's existing state — never replace, which would wipe the
                    # scheduler-owned status/agent fields for a running task.
                    current_tail = self._output_tails.get(event.task_id, "")
                    truncated = _truncate_output_tail(current_tail + event.chunk)
                    self._output_tails[event.task_id] = truncated
                    state = bus.get_snapshot().get(event.task_id, {})
                    bus.update_task_state(
                        event.task_id, {**state, "output_tail": truncated}
                    )
                    # Coalesce: only re-render on a newline boundary.
                    if "\n" in event.chunk:
                        await self._upsert_progress(job_id, bus)

                elif isinstance(event, TaskAnomaly):
                    msg = (
                        f"AGENT ALERT: Task `{event.task_id}` [{event.agent_label}] "
                        f"has used {event.elapsed_seconds:.0f}s of its "
                        f"{event.timeout:.0f}s limit and may be hung."
                    )
                    self._render_signal(msg)

                elif isinstance(event, TaskCompleted):
                    self._clear_task_tail(bus, event.task_id)
                    await self._emit_task_event(bus, job_id, "DONE", event.task_id)

                elif isinstance(event, TaskFailed):
                    self._clear_task_tail(bus, event.task_id)
                    await self._emit_task_event(bus, job_id, "FAIL", event.task_id)

                elif isinstance(event, TaskTimedOut):
                    self._clear_task_tail(bus, event.task_id)
                    await self._emit_task_event(bus, job_id, "TIMEOUT", event.task_id)

                elif isinstance(event, TaskSkipped):
                    self._clear_task_tail(bus, event.task_id)
                    await self._emit_task_event(bus, job_id, "SKIP", event.task_id)

                elif isinstance(event, MilestoneCompleted):
                    # Clear all stored tails
                    self._output_tails.clear()
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
                    # Clear all stored tails
                    self._output_tails.clear()
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

    def _clear_task_tail(self, bus: Any, task_id: str) -> None:
        """Drop a task's rolling tail (local + bus state) on terminal status.

        Merges the tail out of the existing state rather than replacing it, so
        the scheduler-owned status/agent fields survive for the final render.
        """
        self._output_tails.pop(task_id, None)
        state = bus.get_snapshot().get(task_id, {})
        if "output_tail" in state:
            bus.update_task_state(
                task_id, {k: v for k, v in state.items() if k != "output_tail"}
            )

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
    ) -> None:
        await self._upsert_progress(job_id, bus)
        self._render_signal(format_task_signal(job_id, kind, task_id, bus))


__all__ = [
    "_OUTPUT_TAIL_MAX_CHARS",
    "_OUTPUT_TAIL_MAX_LINES",
    "DelegationListener",
    "_truncate_output_tail",
    "format_delegation_progress",
    "format_task_signal",
]
