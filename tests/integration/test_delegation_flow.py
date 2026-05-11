import asyncio
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from supporter.config import config
from supporter.tools.base import ToolError
from supporter.tools.delegate import (
    cancel_delegation,
    check_delegation,
    delegate_tasks,
    get_bus,
    remove_bus,
)
from supporter.types import MilestoneCancelled, MilestoneCompleted, TaskStatus


@pytest.fixture(autouse=True)
def isolate_delegation_capsules(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_delegate_tasks_returns_plan_immediately(self) -> None:
        tasks_json = json.dumps(
            [{"id": "t1", "task": "task 1"}, {"id": "t2", "task": "task 2"}]
        )
        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.side_effect = [
                {
                    "id": "t1",
                    "status": "completed",
                    "output": "out1",
                    "duration": 1.0,
                    "model": "m",
                },
                {
                    "id": "t2",
                    "status": "completed",
                    "output": "out2",
                    "duration": 1.0,
                    "model": "m",
                },
            ]
            plan = await delegate_tasks("Test", tasks_json, max_parallel=2)

        assert "Delegation started" in plan
        assert "Job ID:" in plan
        assert "t1" in plan
        assert "t2" in plan
        assert "check_delegation" in plan

    @pytest.mark.asyncio
    async def test_check_delegation_returns_snapshot(self) -> None:
        tasks_json = json.dumps(
            [{"id": "t1", "task": "task 1"}, {"id": "t2", "task": "task 2"}]
        )
        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.side_effect = [
                {
                    "id": "t1",
                    "status": "completed",
                    "output": "out1",
                    "duration": 1.0,
                    "model": "m",
                },
                {
                    "id": "t2",
                    "status": "completed",
                    "output": "out2",
                    "duration": 1.0,
                    "model": "m",
                },
            ]
            plan = await delegate_tasks("Test", tasks_json, max_parallel=2)
            job_id = next(line for line in plan.splitlines() if "Job ID:" in line)
            job_id = job_id.split("`")[1]
            snapshot = await check_delegation(job_id)
        assert "Test" in snapshot or job_id in snapshot

    @pytest.mark.asyncio
    async def test_check_delegation_invalid_job(self) -> None:
        result = await check_delegation("nonexistent")
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_delegate_tasks_milestone_completes(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first"},
                {"id": "b", "task": "second", "depends_on": ["a"]},
            ]
        )
        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.side_effect = [
                {
                    "id": "a",
                    "status": "completed",
                    "output": "out_a",
                    "duration": 1.0,
                    "model": "m",
                },
                {
                    "id": "b",
                    "status": "completed",
                    "output": "out_b",
                    "duration": 1.0,
                    "model": "m",
                },
            ]
            plan = await delegate_tasks("DAG Test", tasks_json, max_parallel=2)
            job_id = next(line for line in plan.splitlines() if "Job ID:" in line)
            job_id = job_id.split("`")[1]

            bus = get_bus(job_id)
            queue = bus.subscribe()
            completed_event = None
            for _ in range(50):
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
                if event is None:
                    break
                if isinstance(event, MilestoneCompleted):
                    completed_event = event
                    break

        assert completed_event is not None
        assert "after: a" in plan
        results = completed_event.results
        completed = [r for r in results if r["status"] == TaskStatus.COMPLETED]
        assert len(completed) == 2


class TestCancelDelegation:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self) -> None:
        result = await cancel_delegation("nonexistent")
        assert "nonexistent" in result
        assert "unknown" in result.lower() or "complete" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_running_job_publishes_cancelled_event(self) -> None:
        async def slow_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(5.0)
            return {
                "id": "t1",
                "status": TaskStatus.COMPLETED,
                "output": "x",
                "duration": 5.0,
                "model": "m",
            }

        tasks_json = json.dumps([{"id": "t1", "task": "slow"}])
        with patch("supporter.tools.delegate._run_sub_agent", side_effect=slow_run):
            plan = await delegate_tasks("Cancel Test", tasks_json, max_parallel=1)
            job_id = next(
                line for line in plan.splitlines() if "Job ID:" in line
            ).split("`")[1]

            bus = get_bus(job_id)
            queue = bus.subscribe()
            await asyncio.sleep(0.1)

            confirm = await cancel_delegation(job_id)
            assert "Cancellation requested" in confirm

            cancelled_event = None
            for _ in range(20):
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
                if event is None:
                    break
                if isinstance(event, MilestoneCancelled):
                    cancelled_event = event
                    break

        assert cancelled_event is not None
        assert cancelled_event.milestone == "Cancel Test"


@pytest.mark.asyncio
async def test_delegate_tasks_start_callback_and_error_path() -> None:
    called: list[str] = []

    from supporter.tools.delegate import set_delegation_start_callback

    set_delegation_start_callback(lambda job: called.append(job))
    plan = await delegate_tasks("cb", json.dumps([{"id": "t1", "task": "x"}]), 1)
    assert called
    assert "Job ID:" in plan

    with (
        patch(
            "supporter.tools.delegate._validate_tasks",
            side_effect=ValueError("bad"),
        ),
        pytest.raises(ToolError, match="Delegation failed"),
    ):
        await delegate_tasks("bad", "[]", 1)


@pytest.mark.asyncio
async def test_check_delegation_table_rows_for_running_and_done() -> None:
    bus = get_bus("check1", "milestone")
    bus.update_task_state(
        "run",
        {
            "status": "RUNNING",
            "agent_label": "a",
            "started_at": time.monotonic() - 2,
            "timeout": 30,
        },
    )
    bus.update_task_state(
        "done",
        {
            "status": "DONE",
            "agent_label": "b",
            "duration": 1.2,
        },
    )
    table = await check_delegation("check1")
    remove_bus("check1")
    assert "| Task | Status | Agent | Elapsed |" in table
    assert "`run`" in table and "`done`" in table
