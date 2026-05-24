from __future__ import annotations

import asyncio
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

    with patch(
        "supporter.tools.delegate.scheduler.run_sub_agent", side_effect=fake_run
    ):
        await create_capsule("job-dag", "dag", tasks, parallel_limit)
        results = await _execute_dag(
            tasks,
            asyncio.Semaphore(parallel_limit),
            bus,
            "job-dag",
            parallel_limit,
        )
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
