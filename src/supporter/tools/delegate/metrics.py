"""Unified reliability metrics for delegated runs (SPEC §8).

A single bus subscriber aggregates one delegation job's events into per-outcome
reliability counters and timing totals, then emits one structured summary line.
It is read-only over the bus -- it never mutates capsules or task results, so
existing capsule/serialize flows are unaffected.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ...logger import logger
from ...types import (
    DelegationEvent,
    MilestoneCompleted,
    TaskCompleted,
    TaskFailed,
    TaskRetrying,
    TaskSkipped,
    TaskTimedOut,
)
from .backends import QA_REJECTION_MARKER
from .bus import DelegationBus


@dataclass
class JobMetrics:
    """Per-job reliability counters, aggregated from bus events."""

    job_id: str
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    skipped: int = 0
    retries: int = 0
    qa_rejections: int = 0
    task_duration: float = 0.0
    milestone_duration: float = 0.0

    def record(self, event: DelegationEvent) -> None:
        if isinstance(event, TaskCompleted):
            self.completed += 1
            self.task_duration += event.duration
        elif isinstance(event, TaskFailed):
            self.failed += 1
            self.task_duration += event.duration
            if QA_REJECTION_MARKER in event.error:
                self.qa_rejections += 1
        elif isinstance(event, TaskTimedOut):
            self.timed_out += 1
            self.task_duration += event.duration
        elif isinstance(event, TaskSkipped):
            self.skipped += 1
        elif isinstance(event, TaskRetrying):
            self.retries += 1
        elif isinstance(event, MilestoneCompleted):
            self.milestone_duration = event.total_duration

    def summary(self) -> dict[str, Any]:
        attempted = self.completed + self.failed + self.timed_out
        success_rate = self.completed / attempted if attempted else 0.0
        return {
            "job_id": self.job_id,
            "completed": self.completed,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "skipped": self.skipped,
            "retries": self.retries,
            "qa_rejections": self.qa_rejections,
            "success_rate": round(success_rate, 3),
            "task_duration": round(self.task_duration, 2),
            "milestone_duration": round(self.milestone_duration, 2),
        }


async def drain_metrics(
    queue: asyncio.Queue[DelegationEvent | None], job_id: str
) -> JobMetrics:
    """Aggregate events from a pre-subscribed bus queue until the close sentinel.

    The queue must be obtained via ``bus.subscribe()`` BEFORE the job starts
    publishing, so no early events are missed. ``bus.close()`` enqueues ``None``,
    which ends the drain and triggers the summary log.
    """
    metrics = JobMetrics(job_id=job_id)
    while True:
        event = await queue.get()
        if event is None:
            break
        metrics.record(event)
    logger.info(f"Delegation metrics: {metrics.summary()}")
    return metrics


def subscribe_metrics(bus: DelegationBus, job_id: str) -> asyncio.Task[JobMetrics]:
    """Subscribe to the bus synchronously and start draining in the background.

    Subscribing before returning guarantees the metrics queue is registered
    ahead of any publish, regardless of when the returned task is first run.
    """
    queue = bus.subscribe()
    return asyncio.create_task(drain_metrics(queue, job_id))
