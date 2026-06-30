from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import supporter.tools.delegate.capsule as capsule_store
from supporter.config import config
from supporter.tools.delegate.bus import DelegationBus, get_bus, remove_bus
from supporter.tools.delegate.capsule import create_capsule
from supporter.tools.delegate.scheduler import (
    JOB_TASKS,
    _execute_dag,
    _task_to_seed_result,
    find_resumable_jobs,
    resume_milestone,
)
from supporter.types import TaskStatus


@pytest.fixture(autouse=True)
def isolate_delegation_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    monkeypatch.setattr("supporter.pool.get_provider", lambda **_kwargs: object())
    capsule_store._CAPSULE_LOCKS.clear()
    capsule_store._CAPSULE_CACHE.clear()
    capsule_store._CAPSULE_DIRTY_COUNT.clear()
    JOB_TASKS.clear()


def _task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
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
        "depends_on": depends_on or [],
        "tolerate_failures": tolerate_failures,
    }


def _seed_result(
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


@pytest.mark.asyncio
async def test_seed_results_runs_only_unfinished_tasks(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With seed_results, settled tasks are skipped and unfinished tasks run."""
    bus = DelegationBus("seed-test")
    ran_tasks: list[str] = []

    async def fake_run(
        task: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        ran_tasks.append(task["id"])
        return _seed_result(task["id"], TaskStatus.COMPLETED, f"{task['id']} done")

    tasks = [_task("t1"), _task("t2"), _task("t3")]
    # t1 and t2 are already settled, only t3 should run
    seed_results = {
        "t1": _seed_result("t1", TaskStatus.COMPLETED, "already done"),
        "t2": _seed_result("t2", TaskStatus.COMPLETED, "also done"),
    }

    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=fake_run),
        patch.object(config, "delegate_result_repair", False),
    ):
        sem = asyncio.Semaphore(4)
        await create_capsule("seed-job", "seed-test", tasks, 4)
        results, _ = await _execute_dag(tasks, sem, sem, bus, "seed-job", seed_results)

    # Only t3 should have run
    assert ran_tasks == ["t3"]
    # All results should be present
    assert len(results) == 3
    assert {r["id"] for r in results} == {"t1", "t2", "t3"}
    # t1 and t2 should have their seeded output
    by_id = {r["id"]: r for r in results}
    assert by_id["t1"]["output"] == "already done"
    assert by_id["t2"]["output"] == "also done"
    assert by_id["t3"]["output"] == "t3 done"


@pytest.mark.asyncio
async def test_settled_dependency_output_available_to_resumed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resumed task receives output from its settled dependency."""
    bus = DelegationBus("dep-test")

    async def fake_run(
        task: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        # Check that context was injected
        ctx = task.get("context", "")
        assert "t1" in ctx or "already done" in ctx
        return _seed_result(task["id"], TaskStatus.COMPLETED, f"{task['id']} done")

    tasks = [_task("t1"), _task("t2", depends_on=["t1"])]
    seed_results = {
        "t1": _seed_result("t1", TaskStatus.COMPLETED, "already done"),
    }

    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=fake_run),
        patch.object(config, "delegate_result_repair", False),
    ):
        sem = asyncio.Semaphore(4)
        await create_capsule("dep-job", "dep-test", tasks, 4)
        results, _ = await _execute_dag(tasks, sem, sem, bus, "dep-job", seed_results)

    # t2 should have run and received t1's output in context
    assert {r["id"] for r in results} == {"t1", "t2"}
    assert results[1]["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_pending_task_with_failed_dependency_is_skipped_on_resume(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On resume, a pending task with failed dependency is SKIPPED."""
    bus = DelegationBus("skip-test")

    ran_tasks: list[str] = []

    async def fake_run(
        task: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        ran_tasks.append(task["id"])
        return _seed_result(task["id"], TaskStatus.COMPLETED, "ran")

    tasks = [_task("t1"), _task("t2", depends_on=["t1"])]
    # t1 has failed, t2 should be skipped
    seed_results = {
        "t1": _seed_result("t1", TaskStatus.ERROR, "failed"),
    }

    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=fake_run),
        patch.object(config, "delegate_result_repair", False),
    ):
        sem = asyncio.Semaphore(4)
        await create_capsule("skip-job", "skip-test", tasks, 4)
        results, _ = await _execute_dag(tasks, sem, sem, bus, "skip-job", seed_results)

    by_id = {r["id"]: r for r in results}
    assert by_id["t1"]["status"] == TaskStatus.ERROR
    assert by_id["t2"]["status"] == TaskStatus.SKIPPED
    assert "Dependency 't1'" in by_id["t2"]["output"]
    # t2 should NOT have run
    assert ran_tasks == []


@pytest.mark.asyncio
async def test_task_to_seed_result_converts_task_record(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_task_to_seed_result correctly converts capsule task to result format."""
    task_record = {
        "id": "test-task",
        "status": "completed",
        "output": "test output",
        "duration": 1.5,
        "model": "gemini-pro",
        "tokens": {"total_tokens": 100},
        "step_count": 3,
    }
    result = _task_to_seed_result(task_record)
    assert result["id"] == "test-task"
    assert result["status"] == "completed"
    assert result["output"] == "test output"
    assert result["duration"] == 1.5
    assert result["model"] == "gemini-pro"
    assert result["tokens"] == {"total_tokens": 100}
    assert result["step_count"] == 3


@pytest.mark.asyncio
async def test_find_resumable_jobs_finds_interrupted_capsules(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_resumable_jobs identifies interrupted milestones correctly."""
    # Create a capsule directory with an interrupted capsule
    caps_dir = tmp_path / ".supporter" / "delegations"
    caps_dir.mkdir(parents=True)

    # Create an interrupted capsule (pending status with old updated_at)
    interrupted_capsule = caps_dir / "interrupted-job.json"
    old_time = "2020-01-01T00:00:00Z"
    interrupted_capsule.write_text(
        '{"job_id": "interrupted-job", "status": "pending", "milestone": "test", '
        '"updated_at": "' + old_time + '", "tasks": {}}'
    )

    # Create a completed capsule (should not be found)
    completed_capsule = caps_dir / "completed-job.json"
    completed_capsule.write_text(
        '{"job_id": "completed-job", "status": "completed", "milestone": "test", '
        '"updated_at": "2024-01-01T00:00:00Z", "tasks": {}}'
    )

    monkeypatch.setattr(capsule_store, "delegations_dir", lambda: caps_dir)

    job_ids = await find_resumable_jobs()
    assert "interrupted-job" in job_ids
    assert "completed-job" not in job_ids


@pytest.mark.asyncio
async def test_resume_restores_live_and_pre_approved_commands(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume reconstructs unfinished tasks with live + pre_approved_commands."""
    captured: dict[str, Any] = {}

    async def fake_run_milestone(
        _milestone: str, unfinished: list[dict[str, Any]], *_args: Any, **_kwargs: Any
    ) -> None:
        captured["unfinished"] = unfinished

    async def fake_heartbeat(*_args: Any, **_kwargs: Any) -> None:
        return None

    task = _task("t1")
    task["live"] = True
    task["pre_approved_commands"] = ["pwd", "ls"]
    await create_capsule("resume-live", "resume-test", [task], 4)

    with (
        patch(
            "supporter.tools.delegate.scheduler.run_milestone",
            side_effect=fake_run_milestone,
        ),
        patch(
            "supporter.tools.delegate.scheduler.run_heartbeat",
            side_effect=fake_heartbeat,
        ),
    ):
        try:
            result = await resume_milestone("resume-live")
            await JOB_TASKS["resume-live"]
        finally:
            remove_bus("resume-live")

    assert result is True
    recon = {t["id"]: t for t in captured["unfinished"]}["t1"]
    assert recon["live"] is True
    assert recon["pre_approved_commands"] == ["pwd", "ls"]


@pytest.mark.asyncio
async def test_resume_milestone_skips_if_bus_already_exists(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume_milestone returns False if job already has a live bus."""
    job_id = "live-job"
    # Pre-create a bus entry to simulate live job
    get_bus(job_id, "test")
    try:
        result = await resume_milestone(job_id)
        assert result is False
    finally:
        remove_bus(job_id)


@pytest.mark.asyncio
async def test_resume_milestone_skips_if_job_task_live(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resume_milestone returns False if a live JOB_TASKS entry exists."""
    job_id = "running-job"

    async def _never() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(_never())
    JOB_TASKS[job_id] = task
    try:
        result = await resume_milestone(job_id)
        assert result is False
    finally:
        task.cancel()
        JOB_TASKS.pop(job_id, None)
