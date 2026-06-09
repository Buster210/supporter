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

# Approximate USD pricing per 1M tokens, as (input, output), matched by the
# model-name family substring. Rates are published approximations centralized
# here so they can be updated in one place as pricing changes.
_PRICE_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "pro": (1.25, 5.00),
    "flash": (0.30, 2.50),
    "gemma": (0.05, 0.15),
}
_FALLBACK_PRICE_PER_1M_TOKENS: tuple[float, float] = (0.30, 2.50)
_TOKENS_PER_PRICE_UNIT = 1_000_000


def _price_for_model(model: str) -> tuple[float, float]:
    name = model.lower()
    for family, price in _PRICE_PER_1M_TOKENS.items():
        if family in name:
            return price
    return _FALLBACK_PRICE_PER_1M_TOKENS


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for one model call from its token counts."""
    input_rate, output_rate = _price_for_model(model)
    return (
        prompt_tokens * input_rate + completion_tokens * output_rate
    ) / _TOKENS_PER_PRICE_UNIT


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
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens_total: int = 0
    total_steps: int = 0
    cost_usd: float = 0.0

    def record(self, event: DelegationEvent) -> None:
        if isinstance(event, TaskCompleted):
            self.completed += 1
            self.task_duration += event.duration
            tokens = event.tokens or {}
            prompt_tokens = int(tokens.get("prompt_tokens", 0) or 0)
            completion_tokens = int(tokens.get("completion_tokens", 0) or 0)
            self.prompt_tokens_total += prompt_tokens
            self.completion_tokens_total += completion_tokens
            self.total_tokens_total += int(tokens.get("total_tokens", 0) or 0)
            self.total_steps += int(event.step_count or 0)
            self.cost_usd += estimate_cost_usd(
                event.model, prompt_tokens, completion_tokens
            )
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
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens_total": self.total_tokens_total,
            "total_steps": self.total_steps,
            "cost_usd": round(self.cost_usd, 6),
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
