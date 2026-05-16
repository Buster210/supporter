import asyncio
from typing import Any

from ...types import DelegationEvent, MilestoneCompleted

_REGISTRY: dict[str, DelegationBus] = {}


class DelegationBus:
    def __init__(self, milestone: str) -> None:
        self.milestone = milestone
        self.notify_per_task = False
        self._subscribers: list[asyncio.Queue[DelegationEvent | None]] = []
        self._final_event: MilestoneCompleted | None = None
        self._task_states: dict[str, dict[str, Any]] = {}

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

    def close(self) -> None:
        for q in self._subscribers:
            q.put_nowait(None)
        self._subscribers.clear()

    def update_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        self._task_states[task_id] = state

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._task_states)


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
