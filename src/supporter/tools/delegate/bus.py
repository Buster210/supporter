import asyncio
import contextlib
from typing import Any

from ...types import DelegationEvent, MilestoneCompleted, TaskUpdateSent

_REGISTRY: dict[str, DelegationBus] = {}


class DelegationBus:
    def __init__(self, milestone: str) -> None:
        self.milestone = milestone
        self._subscribers: list[asyncio.Queue[DelegationEvent | None]] = []
        self._final_event: MilestoneCompleted | None = None
        self._task_states: dict[str, dict[str, Any]] = {}
        self._task_updates: dict[str, list[dict[str, Any]]] = {}
        self._task_update_cursors: dict[str, int] = {}

    def publish(self, event: DelegationEvent) -> None:
        if isinstance(event, MilestoneCompleted):
            self._final_event = event
        for q in self._subscribers:
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[DelegationEvent | None]:
        q: asyncio.Queue[DelegationEvent | None] = asyncio.Queue()
        if self._final_event:
            q.put_nowait(self._final_event)
            q.put_nowait(None)
        else:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue[DelegationEvent | None]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    def close(self) -> None:
        for q in self._subscribers:
            q.put_nowait(None)
        self._subscribers.clear()
        self._task_updates.clear()
        self._task_update_cursors.clear()

    def update_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        self._task_states[task_id] = state

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._task_states)

    def send_task_update(
        self, task_id: str, message: str, job_id: str | None = None
    ) -> None:
        """Orchestrator pushes an update to a running task.

        Updates are stored in-memory and retrieved via poll_task_updates.
        Each update is a dict with seq (monotonic counter), source, and message.
        """
        if task_id not in self._task_updates:
            self._task_updates[task_id] = []
            self._task_update_cursors[task_id] = 0

        seq = len(self._task_updates[task_id])
        self._task_updates[task_id].append(
            {
                "seq": seq,
                "source": "orchestrator",
                "message": message,
            }
        )

        # Publish event for TUI to show update was sent
        if job_id:
            self.publish(
                TaskUpdateSent(job_id=job_id, task_id=task_id, message=message)
            )

    def poll_task_updates(self, task_id: str) -> list[str]:
        """Sub-agent polls for updates since last poll.

        Returns a list of update messages (strings) not yet seen by this task.
        Advances the per-task read cursor to mark them as consumed.
        """
        if task_id not in self._task_update_cursors:
            self._task_update_cursors[task_id] = 0

        updates = self._task_updates.get(task_id, [])
        cursor = self._task_update_cursors[task_id]
        new_updates = [u["message"] for u in updates[cursor:]]
        self._task_update_cursors[task_id] = len(updates)
        return new_updates


def get_bus(job_id: str, milestone: str = "") -> DelegationBus:
    bus = _REGISTRY.get(job_id)
    if bus is None:
        bus = DelegationBus(milestone)
        _REGISTRY[job_id] = bus
    return bus


def remove_bus(job_id: str) -> None:
    bus = _REGISTRY.pop(job_id, None)
    if bus is not None:
        bus.close()


def bus_exists(job_id: str) -> bool:
    return job_id in _REGISTRY
