from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import supporter.tools.delegate.capsule as capsule_store
from supporter.config import config
from supporter.tools.delegate.agents import _cache
from supporter.tools.delegate.bus import DelegationBus, get_bus, remove_bus
from supporter.tools.delegate.capsule import create_capsule
from supporter.tools.delegate.scheduler import _execute_dag, run_heartbeat
from supporter.tools.delegate.validation import validate_tasks
from supporter.types import (
    HeartbeatTick,
    SubtaskVerificationResult,
    TaskAnomaly,
    TaskCompleted,
    TaskFailed,
    TaskSkipped,
    TaskStatus,
    TaskTimedOut,
)


@pytest.fixture(autouse=True)
def isolate_delegation_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    monkeypatch.setattr("supporter.pool.get_provider", lambda **_kwargs: object())
    _cache.clear()
    capsule_store._CAPSULE_LOCKS.clear()


def _task(
    task_id: str,
    *,
    depends_on: Iterable[str] = (),
    tolerate_failures: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "task": f"run {task_id}",
        "agent": "explorer",
        "tools": {"read_file"},
        "model": "gemini-test",
        "persona": "persona",
        "context": "",
        "timeout": timeout,
        "max_retries": 0,
        "depends_on": list(depends_on),
        "tolerate_failures": tolerate_failures,
    }


def _result(
    task_id: str,
    status: TaskStatus,
    output: str | None = None,
    *,
    duration: float = 0.1,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "status": status,
        "output": output or f"{task_id} {status.value}",
        "duration": duration,
        "model": "gemini-test",
        "tokens": {"total_tokens": 1} if status == TaskStatus.COMPLETED else {},
    }


async def _run_dag(
    tasks: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    *,
    parallel_limit: int = 4,
) -> tuple[list[dict[str, Any]], DelegationBus, list[str], list[Any]]:
    bus = DelegationBus("dag")
    queue = bus.subscribe()
    started: list[str] = []

    async def fake_run(
        task: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        started.append(task["id"])
        return outcomes[task["id"]]

    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=fake_run),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("job-dag", "dag", tasks, parallel_limit)
        sem = asyncio.Semaphore(parallel_limit)
        results, _verifications = await _execute_dag(tasks, sem, sem, bus, "job-dag")
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return results, bus, started, events


def _by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {result["id"]: result for result in results}


def _published(events: list[Any], event_type: type[Any]) -> list[Any]:
    return [event for event in events if isinstance(event, event_type)]


@pytest.mark.asyncio
async def test_tolerated_upstream_failure_allows_downstream_with_context() -> None:
    tasks = [
        _task("analyze"),
        _task("repair", depends_on=["analyze"], tolerate_failures=True),
    ]
    outcomes = {
        "analyze": _result("analyze", TaskStatus.ERROR, "analysis crashed"),
        "repair": _result("repair", TaskStatus.COMPLETED, "repair used context"),
    }

    results, _, started, events = await _run_dag(tasks, outcomes)

    assert _by_id(results)["repair"]["status"] == TaskStatus.COMPLETED
    assert started == ["analyze", "repair"]
    completed = _published(events, TaskCompleted)
    assert [event.task_id for event in completed] == ["repair"]


@pytest.mark.asyncio
async def test_failed_dependency_skips_downstream_but_independent_task_continues() -> (
    None
):
    tasks = [
        _task("root"),
        _task("blocked", depends_on=["root"]),
        _task("independent"),
    ]
    outcomes = {
        "root": _result("root", TaskStatus.ERROR, "root failed"),
        "independent": _result("independent", TaskStatus.COMPLETED, "kept going"),
    }

    results, _, started, events = await _run_dag(tasks, outcomes)
    indexed = _by_id(results)

    assert indexed["blocked"]["status"] == TaskStatus.SKIPPED
    assert "Dependency 'root'" in indexed["blocked"]["output"]
    assert indexed["independent"]["status"] == TaskStatus.COMPLETED
    assert "blocked" not in started
    assert {event.task_id for event in _published(events, TaskSkipped)} == {"blocked"}


@pytest.mark.asyncio
async def test_timeout_propagates_to_dependent_skip() -> None:
    tasks = [_task("slow"), _task("after", depends_on=["slow"])]
    outcomes = {
        "slow": _result("slow", TaskStatus.TIMEOUT, "too slow", duration=30.0),
    }

    results, _, started, events = await _run_dag(tasks, outcomes)
    indexed = _by_id(results)

    assert indexed["slow"]["status"] == TaskStatus.TIMEOUT
    assert indexed["after"]["status"] == TaskStatus.SKIPPED
    assert "after" not in started
    assert [event.task_id for event in _published(events, TaskTimedOut)] == ["slow"]


@pytest.mark.asyncio
async def test_diamond_dependency_waits_for_both_parents() -> None:
    tasks = [
        _task("root"),
        _task("left", depends_on=["root"]),
        _task("right", depends_on=["root"]),
        _task("join", depends_on=["left", "right"]),
    ]
    outcomes = {
        "root": _result("root", TaskStatus.COMPLETED, "root done"),
        "left": _result("left", TaskStatus.COMPLETED, "left done"),
        "right": _result("right", TaskStatus.COMPLETED, "right done"),
        "join": _result("join", TaskStatus.COMPLETED, "join done"),
    }

    results, _, started, _ = await _run_dag(tasks, outcomes)
    indexed = _by_id(results)

    assert indexed["join"]["status"] == TaskStatus.COMPLETED
    assert started.index("join") > started.index("left")
    assert started.index("join") > started.index("right")


@pytest.mark.asyncio
async def test_wide_fan_out_skips_only_failed_branch() -> None:
    tasks = [
        _task("root"),
        *[_task(f"leaf{i}", depends_on=["root"]) for i in range(5)],
        _task("independent"),
    ]
    outcomes = {
        "root": _result("root", TaskStatus.COMPLETED, "root done"),
        "leaf0": _result("leaf0", TaskStatus.ERROR, "leaf failed"),
        "leaf1": _result("leaf1", TaskStatus.COMPLETED),
        "leaf2": _result("leaf2", TaskStatus.COMPLETED),
        "leaf3": _result("leaf3", TaskStatus.COMPLETED),
        "leaf4": _result("leaf4", TaskStatus.COMPLETED),
        "independent": _result("independent", TaskStatus.COMPLETED),
    }

    results, _, _, events = await _run_dag(tasks, outcomes, parallel_limit=6)
    indexed = _by_id(results)

    assert indexed["leaf0"]["status"] == TaskStatus.ERROR
    assert indexed["leaf1"]["status"] == TaskStatus.COMPLETED
    assert indexed["leaf4"]["status"] == TaskStatus.COMPLETED
    assert indexed["independent"]["status"] == TaskStatus.COMPLETED
    failed = _published(events, TaskFailed)
    assert [event.task_id for event in failed] == ["leaf0"]


@pytest.mark.asyncio
async def test_run_sub_agent_retry_exhaustion_becomes_error() -> None:
    task = validate_tasks('[{"id": "retry", "task": "retry task", "max_retries": 1}]')[
        0
    ]
    bus = DelegationBus("retry")
    provider = MagicMock()

    async def fail(_: str) -> Any:
        raise RuntimeError("still failing")

    agent = MagicMock()
    agent.execute = fail
    with (
        patch(
            "supporter.tools.delegate.agents._create_sub_agent",
            return_value=(agent, "prompt"),
        ),
        patch("supporter.tools.delegate.agents.asyncio.sleep", new=AsyncMock()),
    ):
        from supporter.tools.delegate.agents import run_sub_agent

        result = await run_sub_agent(
            task, asyncio.Semaphore(1), bus, "retry-job", provider=provider
        )

    assert result["status"] == TaskStatus.ERROR
    assert "still failing" in result["output"]


@pytest.mark.asyncio
async def test_heartbeat_anomaly_event_marks_state_once() -> None:
    job_id = "hb-dag"
    bus = get_bus(job_id, "heartbeat")
    bus.update_task_state(
        "slow",
        {
            "status": "RUNNING",
            "agent_label": "explorer",
            "started_at": 0.0,
            "timeout": 10.0,
            "anomaly_fired": False,
        },
    )
    queue = bus.subscribe()

    with (
        patch(
            "supporter.tools.delegate.scheduler.bus_exists",
            side_effect=[True, True, False],
        ),
        patch("supporter.tools.delegate.scheduler.time.monotonic", return_value=9.0),
        patch("supporter.tools.delegate.scheduler.asyncio.sleep", new=AsyncMock()),
        patch("supporter.tools.delegate.scheduler.DELEGATE_ANOMALY_THRESHOLD", 0.8),
    ):
        await run_heartbeat(bus, job_id, interval=0)

    remove_bus(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert any(isinstance(event, HeartbeatTick) for event in events)
    anomalies = [event for event in events if isinstance(event, TaskAnomaly)]
    assert len(anomalies) == 1
    assert anomalies[0].task_id == "slow"
    assert bus.get_snapshot()["slow"]["anomaly_fired"] is True


@pytest.mark.asyncio
async def test_per_milestone_cap_is_effective_even_with_higher_global_cap() -> None:
    """SPEC §10: the per-milestone parallel_cap is the actual effective
    concurrency bound; the global hard cap is only an outer ceiling.

    With five independent tasks, a configured max_parallel=2 must mean no
    more than two sub-agents ever run concurrently even though the global
    hard cap is 5.
    """
    tasks = [_task(f"t{i}") for i in range(5)]
    bus = DelegationBus("cap-test")

    in_flight = 0
    peak = 0
    release = asyncio.Event()

    async def slow_run(
        task: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await release.wait()
            return _result(task["id"], TaskStatus.COMPLETED)
        finally:
            in_flight -= 1

    # Per-milestone cap (configured max_parallel) is the inner bound.
    # Global hard cap is the outer bound and is intentionally higher.
    job_sem = asyncio.Semaphore(2)
    global_sem = asyncio.Semaphore(5)

    with (
        patch(
            "supporter.tools.delegate.scheduler.run_sub_agent",
            side_effect=slow_run,
        ),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("cap-job", "cap-test", tasks, 2)
        milestone_task = asyncio.create_task(
            _execute_dag(tasks, job_sem, global_sem, bus, "cap-job")
        )
        # Let asyncio dispatch all 5 task coroutines and let the per-job
        # semaphore settle to its steady state (2 in flight, 3 blocked).
        for _ in range(1000):
            if peak >= 2:
                break
            await asyncio.sleep(0.001)
        release.set()
        results, _verifications = await milestone_task

    assert peak == 2, f"Expected peak concurrency 2, observed {peak}"
    assert len(results) == 5
    for result in results:
        assert result["status"] == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# P3 Item 4 — narrow job-semaphore hold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_task_starts_sub_agent_while_first_is_in_qa() -> None:
    """job_semaphore released after run_sub_agent; QA does not hold a job slot.

    Two tasks, job_semaphore=1: if job_semaphore were held across QA the two
    tasks would form a deadlock (task-1 QA waits for task-2 sub-agent which
    waits for the semaphore task-1 holds). With the narrow hold, task-2's
    sub-agent starts while task-1 is still in QA, and both complete.
    """
    tasks = [_task("t1"), _task("t2")]
    outcomes = {
        "t1": _result("t1", TaskStatus.COMPLETED, "t1 done"),
        "t2": _result("t2", TaskStatus.COMPLETED, "t2 done"),
    }

    task2_sub_started = asyncio.Event()
    task1_qa_gate = asyncio.Event()

    async def controlled_run(
        task: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        tid = task["id"]
        if tid == "t1":
            task1_qa_gate.set()  # signal: t1 sub-agent done, entering QA soon
        elif tid == "t2":
            task2_sub_started.set()  # signal: t2 sub-agent running
        return outcomes[tid]

    async def blocking_qa(
        task: dict[str, Any], result: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> SubtaskVerificationResult:
        if task["id"] == "t1":
            # QA for t1 waits until t2's sub-agent has started.
            # If job_semaphore is still held here, t2 can never start → deadlock.
            await asyncio.wait_for(task2_sub_started.wait(), timeout=2.0)
        return SubtaskVerificationResult(
            task_id=task["id"], passed=True, marker="[QA gate: PASSED]"
        )

    bus = DelegationBus("narrow-sem-test")
    job_sem = asyncio.Semaphore(1)
    global_sem = asyncio.Semaphore(4)

    with (
        patch(
            "supporter.tools.delegate.scheduler.run_sub_agent",
            side_effect=controlled_run,
        ),
        patch(
            "supporter.tools.delegate.scheduler.run_qa_gate_verify_only",
            side_effect=blocking_qa,
        ),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("narrow-job", "narrow", tasks, 1)
        results, _verifications = await _execute_dag(
            tasks, job_sem, global_sem, bus, "narrow-job"
        )

    indexed = _by_id(results)
    assert indexed["t1"]["status"] == TaskStatus.COMPLETED
    assert indexed["t2"]["status"] == TaskStatus.COMPLETED


def test_plan_gate_injects_planner_as_root_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "router_enabled", True)
    tasks = [
        {"id": "t1", "task": "implement thing", "agent": "custom"},
        {"id": "t2", "task": "implement other", "agent": None, "depends_on": ["t1"]},
    ]
    validated = validate_tasks(json.dumps(tasks))

    planners = [t for t in validated if t["agent"] == "planner"]
    assert len(planners) == 1
    plan_id = planners[0]["id"]

    by_id = {t["id"]: t for t in validated}
    assert by_id["t1"]["depends_on"] == [plan_id]
    # t2 already had a dep (on t1), so it is left alone -- waits transitively
    assert by_id["t2"]["depends_on"] == ["t1"]


def test_plan_gate_respects_existing_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "router_enabled", True)
    tasks = [
        {"id": "p1", "task": "plan it", "agent": "planner"},
        {
            "id": "t1",
            "task": "implement thing",
            "agent": "custom",
            "depends_on": ["p1"],
        },
    ]
    validated = validate_tasks(json.dumps(tasks))

    assert len(validated) == 2
    assert [t["agent"] for t in validated] == ["planner", "custom"]


def test_plan_gate_skips_recon_only_milestone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "router_enabled", True)
    tasks = [
        {"id": "e1", "task": "explore repo", "agent": "explorer"},
        {"id": "r1", "task": "review diff", "agent": "code_reviewer"},
    ]
    validated = validate_tasks(json.dumps(tasks))

    assert len(validated) == 2
    assert not any(t["agent"] == "planner" for t in validated)


def test_plan_gate_id_collision_avoidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "router_enabled", True)
    tasks = [
        {"id": "__plan__", "task": "implement thing", "agent": "custom"},
        {"id": "__plan___1", "task": "implement other", "agent": "custom"},
    ]
    validated = validate_tasks(json.dumps(tasks))

    planners = [t for t in validated if t["agent"] == "planner"]
    assert len(planners) == 1
    assert planners[0]["id"] == "__plan___2"


def test_plan_gate_augmented_graph_passes_cycle_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "router_enabled", True)
    tasks = [
        {"id": "t1", "task": "implement thing", "agent": "custom"},
        {
            "id": "t2",
            "task": "implement other",
            "agent": "custom",
            "depends_on": ["t1"],
        },
        {"id": "t3", "task": "implement more", "agent": "custom"},
    ]
    # should not raise despite the new source node fanning out to both roots
    validated = validate_tasks(json.dumps(tasks))
    assert len(validated) == 4


def test_plan_gate_rootless_milestone_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A milestone with no root (every task carries deps) cannot exist as a
    # valid DAG. Injection may fan the plan out to nothing, but the downstream
    # cycle / dep-existence checks must reject it, so no orphaned planner is
    # ever returned. This locks that safety net.
    monkeypatch.setattr(config, "router_enabled", True)
    cyclic = [
        {"id": "t1", "task": "a", "agent": "custom", "depends_on": ["t2"]},
        {"id": "t2", "task": "b", "agent": "custom", "depends_on": ["t1"]},
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_tasks(json.dumps(cyclic))

    external = [
        {"id": "t1", "task": "a", "agent": "custom", "depends_on": ["ext"]},
    ]
    with pytest.raises(ValueError, match="does not exist"):
        validate_tasks(json.dumps(external))


def test_plan_gate_off_by_default_unaffected() -> None:
    assert config.router_enabled is False
    tasks = [{"id": "t1", "task": "implement thing", "agent": "custom"}]
    validated = validate_tasks(json.dumps(tasks))
    assert len(validated) == 1
    assert validated[0]["agent"] == "custom"
