import asyncio

import pytest

from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.metrics import (
    JobMetrics,
    drain_metrics,
    estimate_cost_usd,
    subscribe_metrics,
)
from supporter.types import (
    DelegationEvent,
    MilestoneCompleted,
    TaskCompleted,
    TaskFailed,
    TaskRetrying,
    TaskSkipped,
    TaskTimedOut,
)


def _completed(duration: float = 1.0) -> TaskCompleted:
    return TaskCompleted(
        job_id="j", task_id="t", duration=duration, output="", model="m"
    )


class TestJobMetricsRecord:
    def test_counts_each_outcome(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(_completed(1.0))
        m.record(TaskFailed(job_id="j", task_id="t2", duration=0.5, error="boom"))
        m.record(TaskTimedOut(job_id="j", task_id="t3", duration=2.0))
        m.record(TaskSkipped(job_id="j", task_id="t4", reason="dep"))
        m.record(TaskRetrying(job_id="j", task_id="t", attempt=2, reason="flake"))
        assert (m.completed, m.failed, m.timed_out, m.skipped, m.retries) == (
            1,
            1,
            1,
            1,
            1,
        )
        assert m.task_duration == pytest.approx(3.5)

    def test_qa_rejection_is_detected_from_failure_marker(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(
            TaskFailed(
                job_id="j",
                task_id="t",
                duration=1.0,
                error="QA gate rejected after 3 correction rounds. tier-2 rejected",
            )
        )
        assert m.failed == 1
        assert m.qa_rejections == 1

    def test_plain_failure_is_not_a_qa_rejection(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(TaskFailed(job_id="j", task_id="t", duration=1.0, error="boom"))
        assert m.qa_rejections == 0

    def test_milestone_duration_recorded(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(MilestoneCompleted("j", "ms", [], 9.0))
        assert m.milestone_duration == 9.0

    def test_empty_tokens_and_step_count_default_to_zero(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(_completed(1.0))
        assert m.prompt_tokens_total == 0
        assert m.completion_tokens_total == 0
        assert m.total_tokens_total == 0
        assert m.total_steps == 0

    def test_tokens_and_step_count_aggregate_across_completions(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(
            TaskCompleted(
                job_id="j",
                task_id="t1",
                duration=1.0,
                output="",
                model="m",
                tokens={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                step_count=3,
            )
        )
        m.record(
            TaskCompleted(
                job_id="j",
                task_id="t2",
                duration=2.0,
                output="",
                model="m",
                tokens={"prompt_tokens": 4, "total_tokens": 7},
                step_count=2,
            )
        )
        assert m.prompt_tokens_total == 14
        assert m.completion_tokens_total == 5
        assert m.total_tokens_total == 22
        assert m.total_steps == 5


class TestSummary:
    def test_success_rate_excludes_skipped(self) -> None:
        m = JobMetrics(job_id="j", completed=3, failed=1, timed_out=0, skipped=5)
        assert m.summary()["success_rate"] == 0.75

    def test_success_rate_zero_when_nothing_attempted(self) -> None:
        assert JobMetrics(job_id="j").summary()["success_rate"] == 0.0

    def test_summary_includes_efficiency_fields(self) -> None:
        m = JobMetrics(
            job_id="j",
            completed=1,
            prompt_tokens_total=12,
            completion_tokens_total=3,
            total_tokens_total=15,
            total_steps=4,
        )
        s = m.summary()
        assert s["prompt_tokens_total"] == 12
        assert s["completion_tokens_total"] == 3
        assert s["total_tokens_total"] == 15
        assert s["total_steps"] == 4


class TestCost:
    def test_flash_rate_applied(self) -> None:
        # flash = (0.30, 2.50) per 1M tokens
        cost = estimate_cost_usd("gemini-3.1-flash-live-preview", 1000, 2000)
        assert cost == pytest.approx((1000 * 0.30 + 2000 * 2.50) / 1_000_000)

    def test_pro_rate_applied(self) -> None:
        cost = estimate_cost_usd("gemini-pro", 1000, 1000)
        assert cost == pytest.approx((1000 * 1.25 + 1000 * 5.00) / 1_000_000)

    def test_unknown_model_uses_fallback_rate(self) -> None:
        cost = estimate_cost_usd("mystery-model", 1_000_000, 0)
        assert cost == pytest.approx(0.30)

    def test_cost_accumulates_across_completions(self) -> None:
        m = JobMetrics(job_id="j")
        m.record(
            TaskCompleted(
                job_id="j",
                task_id="t1",
                duration=1.0,
                output="",
                model="gemini-flash",
                tokens={"prompt_tokens": 1000, "completion_tokens": 1000},
            )
        )
        m.record(
            TaskCompleted(
                job_id="j",
                task_id="t2",
                duration=1.0,
                output="",
                model="gemini-pro",
                tokens={"prompt_tokens": 1000, "completion_tokens": 1000},
            )
        )
        expected = (1000 * 0.30 + 1000 * 2.50 + 1000 * 1.25 + 1000 * 5.00) / 1_000_000
        assert m.summary()["cost_usd"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_drain_metrics_stops_on_sentinel() -> None:
    queue: asyncio.Queue[DelegationEvent | None] = asyncio.Queue()
    queue.put_nowait(_completed(1.0))
    queue.put_nowait(_completed(2.0))
    queue.put_nowait(None)
    metrics = await drain_metrics(queue, "j")
    assert metrics.completed == 2
    assert metrics.task_duration == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_subscribe_metrics_captures_events_published_after_subscribe() -> None:
    bus = DelegationBus("ms")
    task = subscribe_metrics(bus, "j")
    bus.publish(_completed(1.0))
    bus.publish(TaskFailed(job_id="j", task_id="t2", duration=0.5, error="boom"))
    bus.close()
    metrics = await task
    assert metrics.completed == 1
    assert metrics.failed == 1
