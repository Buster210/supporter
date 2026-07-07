"""Delegation event listener with dispatch-map event loop.

ponytail: Removed the buffering system (flush-when-both-ready between
delegation completion and answer-bubble streaming). Delegation updates
now stream directly to the UI.  Consolidated format_task_signal +
format_delegation_progress into format_delegation_update.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from ..logger import logger
from ..tools.delegate.formatting import format_delegation_table

# Maximum tail length for streaming output (last ~500 chars or last 3 lines)
_OUTPUT_TAIL_MAX_CHARS = 500
_OUTPUT_TAIL_MAX_LINES = 3


def _truncate_output_tail(text: str) -> str:
    """Keep a bounded rolling tail of streamed output (char + line caps)."""
    lines = text.splitlines(keepends=True)
    tail = "".join(lines[-_OUTPUT_TAIL_MAX_LINES:])
    if len(tail) > _OUTPUT_TAIL_MAX_CHARS:
        tail = tail[-_OUTPUT_TAIL_MAX_CHARS:]
    return tail


class MessageInjector(Protocol):
    """Callback to inject a message into the chat."""

    def __call__(self, message: str) -> None: ...


class ProgressUpdater(Protocol):
    """Async callback to update delegation progress on the UI."""

    async def __call__(self, job_id: str, bus: Any) -> None: ...


class PlanBubbleInjector(Protocol):
    """Injects a formatted plan as a visible bubble in the chat."""

    def __call__(self, markdown: str) -> None: ...


_KIND_LABELS = {
    "DONE": "completed",
    "FAIL": "failed",
    "TIMEOUT": "timed out",
    "SKIP": "skipped",
}


def format_delegation_update(
    job_id: str,
    bus: Any,
    *,
    task_id: str | None = None,
    status: str | None = None,
) -> str:
    """Consolidated delegation formatter — single-line signal or full table.

    When *task_id* is given, emit a one-line human-readable status for that
    task (the old ``format_task_signal`` path).  *status* is the explicit
    status label (e.g. ``"DONE"``); if omitted, falls back to the bus
    snapshot.  Otherwise emit the full progress table (the old
    ``format_delegation_progress`` path).

    ponytail: merged two similar functions that both read bus snapshots
    and emit markdown; single entry point for all delegation display.
    """
    snapshot = bus.get_snapshot()

    if task_id is not None:
        state = snapshot.get(task_id, {})
        agent = str(state.get("agent_label", "?"))
        kind = (status or str(state.get("status", "PENDING"))).upper()
        label = _KIND_LABELS.get(kind, kind.lower())
        return f"Task {task_id} ({agent}) {label}"

    headers = ["Task", "Agent", "Status", "Time"]
    data_rows = []
    for tid, state in snapshot.items():
        st = _display_task_status(str(state.get("status", "PENDING")))
        agent = str(state.get("agent_label", "?"))
        duration = state.get("duration")
        duration_text = ""
        if isinstance(duration, int | float) and duration > 0:
            duration_text = f"{duration:.2f}s"
        output_tail = state.get("output_tail", "")
        if output_tail:
            duration_text += f" [{output_tail.splitlines()[0][:40]}]"
        data_rows.append([tid, agent, st, duration_text])
    table = format_delegation_table(headers, data_rows)
    return f"Job `{job_id}`\n\n{table}"


def _display_task_status(status: str) -> str:
    """Convert task status codes to natural-language labels (AC3)."""
    normalized = status.lower()
    status_map = {
        "running": "working",
        "done": "completed",
        "pending": "waiting",
        "failed": "failed",
        "error": "failed",
        "timeout": "timed out",
        "skipped": "skipped",
    }
    return status_map.get(normalized, normalized)


def format_delegation_summary(job_id: str, bus: Any) -> str:
    """AC4: Compact one-line completion summary (what was done + result)."""
    snapshot = bus.get_snapshot()
    if not snapshot:
        return f"Job {job_id} completed"
    completed = sum(
        1 for s in snapshot.values() if str(s.get("status")).upper() == "DONE"
    )
    failed = sum(
        1
        for s in snapshot.values()
        if str(s.get("status")).upper() in ("FAIL", "ERROR", "TIMEOUT", "SKIPPED")
    )
    total = len(snapshot)
    result = "completed" if failed == 0 else "completed with issues"
    return f"Job {job_id}: {completed}/{total} tasks {result}"


class DelegationListener:
    """Listens for delegation events and renders task progress + signals to the UI."""

    def __init__(
        self,
        inject_message: MessageInjector,
        drop_progress: Callable[[str], None],
        render_signal: Callable[[str], None],
        render_progress_live: Callable[[str, str], None] | None = None,
        render_summary: Callable[[str, str], None] | None = None,
        plan_bubble_injector: PlanBubbleInjector | None = None,
        plan_storer: Callable[[str, str], None] | None = None,
    ) -> None:
        self._inject_message = inject_message
        self._drop_progress = drop_progress
        self._render_signal = render_signal
        self._render_progress_live = render_progress_live
        self._render_summary = render_summary
        self._plan_bubble_injector = plan_bubble_injector
        self._plan_storer = plan_storer
        # Per-task rolling output tails (bounded)
        self._output_tails: dict[str, str] = {}

    # ── event handlers (one per event type) ──────────────────────────────

    def _on_task_output_chunk(self, event: Any, job_id: str, bus: Any) -> None:
        """Append to bounded tail and re-render on newline boundaries."""
        current_tail = self._output_tails.get(event.task_id, "")
        truncated = _truncate_output_tail(current_tail + event.chunk)
        self._output_tails[event.task_id] = truncated
        state = bus.get_snapshot().get(event.task_id, {})
        bus.update_task_state(event.task_id, {**state, "output_tail": truncated})
        if "\n" in event.chunk and self._render_progress_live is not None:
            self._render_progress_live(job_id, format_delegation_update(job_id, bus))

    def _on_task_anomaly(self, event: Any) -> None:
        msg = (
            f"AGENT ALERT: Task `{event.task_id}` "
            f"[{event.agent_label}] "
            f"has used {event.elapsed_seconds:.0f}s of its "
            f"{event.timeout:.0f}s limit and may be hung."
        )
        self._render_signal(msg)

    def _on_task_update_sent(self, event: Any) -> None:
        """Handle TaskUpdateSent event — show one line to TUI."""
        msg = f"Update sent to task {event.task_id}: {event.message}"
        self._render_signal(msg)

    def _on_task_terminal(self, event: Any, job_id: str, bus: Any, kind: str) -> None:
        """Handle TaskCompleted/Failed/TimedOut/Skipped."""
        self._clear_task_tail(bus, event.task_id)
        self._render_signal(
            format_delegation_update(job_id, bus, task_id=event.task_id, status=kind)
        )

    def _serialize_milestone_result(
        self,
        event: Any,
        job_id: str,
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Serialize milestone result with fallback to legacy format."""
        from ..tools.delegate.api import serialize_capsule_result
        from ..tools.delegate.scheduler import serialize_results

        try:
            return serialize_capsule_result(job_id)
        except Exception as e:
            logger.warning(
                f"Falling back to legacy delegation result for "
                f"{job_id} [{type(e).__name__}]: {e}"
            )
            if hasattr(event, "results"):
                has_failures = any(
                    str(r.get("status")) in {"error", "timeout", "skipped"}
                    for r in event.results
                )
                return serialize_results(
                    event.milestone,
                    event.results,
                    event.total_duration,
                    job_id,
                    status=("completed_with_failures" if has_failures else "completed"),
                )
            return {
                "job_id": job_id,
                "milestone": event.milestone,
                "status": status,
                "total_duration": round(event.total_duration, 2),
            }

    def _on_milestone_terminal(self, event: Any, job_id: str, bus: Any) -> bool:
        """Handle MilestoneCompleted/Cancelled. Returns True to break loop."""
        self._output_tails.clear()
        # MilestoneCancelled has no .results; MilestoneCompleted does.
        status = "cancelled" if not hasattr(event, "results") else "completed"
        payload = self._serialize_milestone_result(event, job_id, status=status)
        self._inject_capsule_result(payload)
        if self._render_summary is not None:
            if status == "cancelled":
                summary = f"Job {job_id} cancelled"
            else:
                summary = format_delegation_summary(job_id, bus)
            self._render_summary(job_id, summary)
        self._drop_progress(job_id)
        return True

    # ── main event loop ──────────────────────────────────────────────────

    async def listen(self, job_id: str) -> None:
        """Listen for delegation events on job_id and render to UI in real time.

        Dispatches events (task start/complete/fail, anomaly, milestone) to
        appropriate handler methods which render signals and progress updates.
        """
        from ..tools.delegate.bus import get_bus
        from ..types import (
            MilestoneCancelled,
            MilestoneCompleted,
            TaskAnomaly,
            TaskCompleted,
            TaskFailed,
            TaskOutputChunk,
            TaskSkipped,
            TaskTimedOut,
            TaskUpdateSent,
        )

        # ponytail: dispatch map replaces 10+ isinstance branches.
        terminal_tasks = (
            TaskCompleted,
            TaskFailed,
            TaskTimedOut,
            TaskSkipped,
        )
        task_kinds = {
            TaskCompleted: "DONE",
            TaskFailed: "FAIL",
            TaskTimedOut: "TIMEOUT",
            TaskSkipped: "SKIP",
        }
        milestone_events = (MilestoneCompleted, MilestoneCancelled)

        try:
            bus = get_bus(job_id)
            queue = bus.subscribe()
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break

                    etype = type(event)

                    if etype is TaskOutputChunk:
                        self._on_task_output_chunk(event, job_id, bus)
                        if "\n" in event.chunk:  # type: ignore[attr-defined]
                            await self._upsert_progress_live(job_id, bus)
                    elif etype in terminal_tasks:
                        self._on_task_terminal(event, job_id, bus, task_kinds[etype])
                        await self._upsert_progress_live(job_id, bus)
                    elif etype is TaskAnomaly:
                        self._on_task_anomaly(event)
                    elif etype is TaskUpdateSent:
                        self._on_task_update_sent(event)
                    elif etype in milestone_events:
                        self._on_milestone_terminal(event, job_id, bus)
                        break
            finally:
                bus.unsubscribe(queue)
        except Exception as e:
            logger.error(f"Delegation listener failed for {job_id}: {e}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _clear_task_tail(self, bus: Any, task_id: str) -> None:
        """Drop a task's rolling tail on terminal status."""
        self._output_tails.pop(task_id, None)
        state = bus.get_snapshot().get(task_id, {})
        if "output_tail" in state:
            bus.update_task_state(
                task_id,
                {k: v for k, v in state.items() if k != "output_tail"},
            )

    async def _upsert_progress_live(self, job_id: str, bus: Any) -> None:
        """Render live progress table to UI."""
        if self._render_progress_live is not None:
            self._render_progress_live(job_id, format_delegation_update(job_id, bus))

    def _inject_capsule_result(self, payload: dict[str, Any]) -> None:
        """Inject capsule JSON as system message; mount plan bubble."""
        agent = payload.get("agent", "")
        if agent == "planner":
            if self._plan_bubble_injector is not None:
                from ..tools.delegate.capsule_view import (
                    format_plan_capsule,
                )

                try:
                    plan_md = format_plan_capsule(payload)
                    if plan_md:
                        self._plan_bubble_injector(plan_md)
                except Exception as exc:
                    logger.warning(f"Failed to render plan bubble: {exc}")
            if self._plan_storer is not None:
                objective = payload.get("milestone", "")
                plan_text = json.dumps(payload, indent=2)
                self._plan_storer(objective, plan_text)
        msg = (
            "DELEGATION_CAPSULE_RESULT (json):\n```json\n"
            f"{json.dumps(payload, indent=2)}\n```"
        )
        self._inject_message(msg)


__all__ = [
    "_OUTPUT_TAIL_MAX_CHARS",
    "_OUTPUT_TAIL_MAX_LINES",
    "DelegationListener",
    "MessageInjector",
    "PlanBubbleInjector",
    "ProgressUpdater",
    "_truncate_output_tail",
    "format_delegation_summary",
    "format_delegation_update",
]
