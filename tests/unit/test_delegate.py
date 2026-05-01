import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.config import config
from supporter.tools.delegate import (
    _build_tool_registry,
    _create_sub_agent,
    _execute_dag,
    _format_results,
    _inject_dependency_context,
    _resolve_agent_profile,
    _run_sub_agent,
    _should_skip,
    _validate_tasks,
    cancel_delegation,
    check_delegation,
    delegate_tasks,
    serialize_results,
)
from supporter.types import LLMResult, TaskStatus


class TestAgentRoster:
    def test_resolve_roster_agent(self) -> None:
        task = {"agent": "security_auditor", "task": "audit"}
        profile = _resolve_agent_profile(task)
        assert (
            profile["persona"]
            == config.delegate_agent_roster["security_auditor"]["persona"]
        )
        assert profile["tools"] == {"read_file", "execute_bash"}

    def test_resolve_roster_with_overrides(self) -> None:
        task = {
            "agent": "security_auditor",
            "task": "audit",
            "persona": "Custom persona",
            "tools": "read_file",
        }
        profile = _resolve_agent_profile(task)
        assert profile["persona"] == "Custom persona"
        assert profile["tools"] == "read_file"

    def test_resolve_custom_agent(self) -> None:
        task = {"agent": "custom", "task": "do stuff", "persona": "My persona"}
        profile = _resolve_agent_profile(task)
        assert profile["persona"] == "My persona"

    def test_resolve_unknown_agent_falls_back_to_custom(self) -> None:
        task = {"agent": "unknown_role", "task": "do stuff"}
        profile = _resolve_agent_profile(task)
        assert profile["persona"] == config.delegate_default_persona

    def test_resolve_no_agent_field(self) -> None:
        task = {"task": "do stuff"}
        profile = _resolve_agent_profile(task)
        assert profile["persona"] == config.delegate_default_persona

    def test_validate_tasks_with_roster_agent(self) -> None:
        tasks_json = json.dumps(
            [{"id": "t1", "agent": "test_engineer", "task": "run tests"}]
        )
        validated = _validate_tasks(tasks_json)
        assert (
            validated[0]["persona"]
            == config.delegate_agent_roster["test_engineer"]["persona"]
        )
        assert validated[0]["tools"] == {"read_file", "execute_bash"}


class TestValidation:
    def test_valid_json(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "t1", "task": "do something", "tools": "read_file,execute_bash"},
                {
                    "id": "t2",
                    "task": "do another",
                    "persona": "custom",
                    "context": "ctx",
                },
            ]
        )
        validated = _validate_tasks(tasks_json)
        assert len(validated) == 2
        assert validated[0]["id"] == "t1"
        assert validated[0]["tools"] == {"read_file", "execute_bash"}
        assert validated[1]["persona"] == "custom"
        assert validated[1]["context"] == "ctx"
        assert validated[1]["tools"] == config.delegate_allowed_tools

    def test_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            _validate_tasks("not json")

    def test_not_array(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON array"):
            _validate_tasks('{"id": "t1"}')

    def test_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="missing a valid string 'id'"):
            _validate_tasks('[{"task": "no id"}]')
        with pytest.raises(ValueError, match="missing a valid string 'task'"):
            _validate_tasks('[{"id": "no task"}]')

    def test_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="Duplicate task ID"):
            _validate_tasks('[{"id": "t1", "task": "a"}, {"id": "t1", "task": "b"}]')

    def test_too_many(self) -> None:
        tasks = [
            {"id": f"t{i}", "task": "task"}
            for i in range(config.delegate_max_tasks + 1)
        ]
        with pytest.raises(ValueError, match="Too many tasks"):
            _validate_tasks(json.dumps(tasks))

    def test_timeout_parsing(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "t1", "task": "a", "timeout": 10},
                {"id": "t2", "task": "b", "timeout": 1000},
                {"id": "t3", "task": "c"},
            ]
        )
        validated = _validate_tasks(tasks_json)
        assert validated[0]["timeout"] == 10
        assert validated[1]["timeout"] == 600
        assert validated[2]["timeout"] == 180


class TestDependencyValidation:
    def test_valid_depends_on(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first"},
                {"id": "b", "task": "second", "depends_on": ["a"]},
            ]
        )
        validated = _validate_tasks(tasks_json)
        assert validated[0]["depends_on"] == []
        assert validated[1]["depends_on"] == ["a"]

    def test_depends_on_nonexistent(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first"},
                {"id": "b", "task": "second", "depends_on": ["nonexistent"]},
            ]
        )
        with pytest.raises(ValueError, match="does not exist"):
            _validate_tasks(tasks_json)

    def test_depends_on_self(self) -> None:
        tasks_json = json.dumps([{"id": "a", "task": "first", "depends_on": ["a"]}])
        with pytest.raises(ValueError, match="cannot depend on itself"):
            _validate_tasks(tasks_json)

    def test_cycle_detection(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first", "depends_on": ["b"]},
                {"id": "b", "task": "second", "depends_on": ["a"]},
            ]
        )
        with pytest.raises(ValueError, match="cycle"):
            _validate_tasks(tasks_json)

    def test_depends_on_string_format(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first"},
                {"id": "b", "task": "second", "depends_on": "a"},
            ]
        )
        validated = _validate_tasks(tasks_json)
        assert validated[1]["depends_on"] == ["a"]

    def test_complex_dag_valid(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "t"},
                {"id": "b", "task": "t"},
                {"id": "c", "task": "t", "depends_on": ["a"]},
                {"id": "d", "task": "t", "depends_on": ["b"]},
                {"id": "e", "task": "t", "depends_on": ["c", "d"]},
            ]
        )
        validated = _validate_tasks(tasks_json)
        assert len(validated) == 5
        assert validated[4]["depends_on"] == ["c", "d"]


class TestToolRegistry:
    def test_scoping(self) -> None:
        registry = _build_tool_registry({"read_file", "execute_bash", "unknown_tool"})
        assert "read_file" in registry
        assert "execute_bash" in registry
        assert "write_file" not in registry
        assert "unknown_tool" not in registry

    def test_no_recursion(self) -> None:
        registry = _build_tool_registry({"read_file", "delegate_tasks"})
        assert "read_file" in registry
        assert "delegate_tasks" not in registry


class TestSubAgentFactory:
    @patch("supporter.index.get_provider")
    @patch("supporter.tools.delegate.ChatAgent")
    def test_create_sub_agent(
        self, mock_agent_class: Any, mock_get_provider: Any
    ) -> None:
        task = {
            "id": "t1",
            "task": "my task",
            "persona": "my persona",
            "tools": {"read_file"},
            "model": "my-model",
            "context": "my context",
        }
        _agent, prompt = _create_sub_agent(task)
        mock_get_provider.assert_called_once()
        mock_agent_class.assert_called_once()
        assert "my task" in prompt
        assert "my context" in prompt


class TestSubAgentRunner:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        task = {
            "id": "t1",
            "task": "task",
            "tools": {"read_file"},
            "model": "m",
            "persona": "p",
            "context": "c",
            "timeout": 10,
            "max_retries": 0,
            "depends_on": [],
        }
        semaphore = asyncio.Semaphore(1)
        mock_result = LLMResult(text="Done", model="m", duration=1.0)
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=mock_result)

        from supporter.tools.event_bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch(
            "supporter.tools.delegate._create_sub_agent",
            return_value=(mock_agent, "prompt"),
        ):
            result = await _run_sub_agent(task, semaphore, mock_bus, "job1")

        assert result["id"] == "t1"
        assert result["status"] == TaskStatus.COMPLETED
        assert result["output"] == "Done"

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        task = {
            "id": "t1",
            "task": "task",
            "tools": {"read_file"},
            "model": "m",
            "persona": "p",
            "context": "c",
            "timeout": 0.01,
            "max_retries": 0,
            "depends_on": [],
        }
        semaphore = asyncio.Semaphore(1)

        from supporter.tools.event_bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch("supporter.tools.delegate._create_sub_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.execute = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_factory.return_value = (mock_agent, "prompt")
            result = await _run_sub_agent(task, semaphore, mock_bus, "job1")

        assert result["status"] == TaskStatus.TIMEOUT
        assert "0.01s" in result["output"]

    @pytest.mark.asyncio
    async def test_error(self) -> None:
        task = {
            "id": "t1",
            "task": "task",
            "tools": {"read_file"},
            "model": "m",
            "persona": "p",
            "context": "c",
            "timeout": 10,
            "max_retries": 0,
            "depends_on": [],
        }
        semaphore = asyncio.Semaphore(1)

        from supporter.tools.event_bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch("supporter.tools.delegate._create_sub_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.execute = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_factory.return_value = (mock_agent, "prompt")
            result = await _run_sub_agent(task, semaphore, mock_bus, "job1")

        assert result["status"] == TaskStatus.ERROR
        assert "Boom" in result["output"]


class TestDAGExecution:
    @pytest.mark.asyncio
    async def test_parallel_no_deps(self) -> None:
        tasks = [
            {
                "id": "a",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": [],
            },
            {
                "id": "b",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": [],
            },
        ]
        semaphore = asyncio.Semaphore(5)

        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.side_effect = [
                {"id": "a", "status": "completed", "output": "out_a", "duration": 1.0},
                {"id": "b", "status": "completed", "output": "out_b", "duration": 1.0},
            ]
            from supporter.tools.event_bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            results = await _execute_dag(tasks, semaphore, mock_bus, "test_job", 5)

        assert len(results) == 2
        assert mock_run.call_count == 2
        assert results[0]["status"] == TaskStatus.COMPLETED
        assert results[1]["status"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_sequential_chain(self) -> None:
        tasks = [
            {
                "id": "a",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": [],
            },
            {
                "id": "b",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": ["a"],
            },
        ]
        semaphore = asyncio.Semaphore(5)

        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.side_effect = [
                {"id": "a", "status": "completed", "output": "out_a", "duration": 1.0},
                {"id": "b", "status": "completed", "output": "out_b", "duration": 1.0},
            ]
            from supporter.tools.event_bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            results = await _execute_dag(tasks, semaphore, mock_bus, "test_job", 5)

        assert len(results) == 2
        b_call_args = mock_run.call_args_list[1]
        enriched_task = b_call_args[0][0]
        assert "out_a" in enriched_task["context"]

    @pytest.mark.asyncio
    async def test_failure_propagation(self) -> None:
        tasks = [
            {
                "id": "a",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": [],
            },
            {
                "id": "b",
                "task": "t",
                "tools": {"read_file"},
                "model": "m",
                "persona": "p",
                "context": "",
                "timeout": 10,
                "depends_on": ["a"],
            },
        ]
        semaphore = asyncio.Semaphore(5)

        with patch("supporter.tools.delegate._run_sub_agent") as mock_run:
            mock_run.return_value = {
                "id": "a",
                "status": "error",
                "output": "crashed",
                "duration": 0.5,
            }
            from supporter.tools.event_bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            results = await _execute_dag(tasks, semaphore, mock_bus, "test_job", 5)

        assert results[0]["status"] == TaskStatus.ERROR
        assert results[1]["status"] == TaskStatus.SKIPPED
        assert "Dependency 'a'" in results[1]["output"]
        assert mock_run.call_count == 1


class TestFormatResults:
    def test_mixed_results(self) -> None:
        results = [
            {
                "id": "t1",
                "status": "completed",
                "output": "ok",
                "model": "m1",
                "duration": 1.5,
            },
            {"id": "t2", "status": "error", "output": "fail", "duration": 0.5},
            {
                "id": "t3",
                "status": "skipped",
                "output": "Skipped: dep failed",
                "duration": 0.0,
            },
        ]
        report = _format_results("M1", results, 2.0)
        assert "MILESTONE REPORT: M1" in report
        assert "1/3 completed" in report
        assert "1 skipped" in report
        assert "Task: t1" in report
        assert "Task: t3" in report


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
        from supporter.tools.event_bus import get_bus
        from supporter.types import MilestoneCompleted

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


class TestSerializeResults:
    def test_serializes_summary_and_per_task(self) -> None:
        results = [
            {
                "id": "t1",
                "status": TaskStatus.COMPLETED,
                "output": "ok",
                "model": "m1",
                "duration": 1.5,
                "tokens": {"total_tokens": 100},
            },
            {
                "id": "t2",
                "status": TaskStatus.ERROR,
                "output": "boom",
                "duration": 0.5,
                "tokens": {},
            },
            {
                "id": "t3",
                "status": TaskStatus.SKIPPED,
                "output": "Skipped: dep failed",
                "duration": 0.0,
            },
            {
                "id": "t4",
                "status": TaskStatus.TIMEOUT,
                "output": "Error: timed out",
                "duration": 60.0,
            },
        ]
        payload = serialize_results("M1", results, 62.0, "abc12345")

        assert payload["job_id"] == "abc12345"
        assert payload["milestone"] == "M1"
        assert payload["status"] == "completed"
        assert payload["total_duration"] == 62.0
        assert payload["totals"] == {
            "completed": 1,
            "failed": 1,
            "skipped": 1,
            "timed_out": 1,
            "tokens": 100,
        }
        assert len(payload["tasks"]) == 4
        t1, t2, t3, t4 = payload["tasks"]
        assert t1["output"] == "ok"
        assert t1["tokens"] == 100
        assert t1["model"] == "m1"
        assert t2["error"] == "boom"
        assert "output" not in t2
        assert t3["output"] == "Skipped: dep failed"
        assert t4["output"] == "Error: timed out"

    def test_status_field_overrides_default(self) -> None:
        payload = serialize_results("M1", [], 1.23, "j1", status="cancelled")
        assert payload["status"] == "cancelled"
        assert payload["tasks"] == []
        assert payload["totals"]["completed"] == 0


class TestToleratesFailures:
    def test_should_skip_when_dep_failed(self) -> None:
        task = {"depends_on": ["t1"], "tolerate_failures": False}
        results = {"t1": {"status": TaskStatus.ERROR, "output": "boom"}}
        assert _should_skip(task, results) is not None

    def test_tolerate_failures_bypasses_skip(self) -> None:
        task = {"depends_on": ["t1"], "tolerate_failures": True}
        results = {"t1": {"status": TaskStatus.ERROR, "output": "boom"}}
        assert _should_skip(task, results) is None

    def test_no_skip_when_dep_completed(self) -> None:
        task = {"depends_on": ["t1"], "tolerate_failures": False}
        results = {"t1": {"status": TaskStatus.COMPLETED, "output": "ok"}}
        assert _should_skip(task, results) is None

    def test_validation_accepts_tolerate_failures_field(self) -> None:
        tasks_json = json.dumps(
            [{"id": "t1", "task": "do it", "tolerate_failures": True}]
        )
        validated = _validate_tasks(tasks_json)
        assert validated[0]["tolerate_failures"] is True

    def test_validation_defaults_tolerate_failures_false(self) -> None:
        tasks_json = json.dumps([{"id": "t1", "task": "do it"}])
        validated = _validate_tasks(tasks_json)
        assert validated[0]["tolerate_failures"] is False


class TestDependencyContextStatus:
    def test_no_tag_for_completed_dep(self) -> None:
        task = {"depends_on": ["t1"], "context": ""}
        results = {
            "t1": {"status": TaskStatus.COMPLETED, "output": "all good"},
        }
        enriched = _inject_dependency_context(task, results)
        assert "[COMPLETED]" not in enriched["context"]
        assert "Output from 't1'" in enriched["context"]
        assert "all good" in enriched["context"]

    def test_injects_failed_dep_outputs(self) -> None:
        task = {"depends_on": ["t1", "t2"], "context": "ctx"}
        results = {
            "t1": {"status": TaskStatus.COMPLETED, "output": "ok"},
            "t2": {"status": TaskStatus.ERROR, "output": "boom"},
        }
        enriched = _inject_dependency_context(task, results)
        assert "[COMPLETED]" not in enriched["context"]
        assert "[ERROR]" in enriched["context"]
        assert "boom" in enriched["context"]
        assert enriched["context"].startswith("ctx")

    def test_no_change_without_dependencies(self) -> None:
        task = {"depends_on": [], "context": "original"}
        enriched = _inject_dependency_context(task, {})
        assert enriched is task


class TestCancelDelegation:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(self) -> None:
        result = await cancel_delegation("nonexistent")
        assert "nonexistent" in result
        assert "unknown" in result.lower() or "complete" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_running_job_publishes_cancelled_event(self) -> None:
        from supporter.tools.event_bus import get_bus
        from supporter.types import MilestoneCancelled

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
