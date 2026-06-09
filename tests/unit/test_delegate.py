import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import supporter.tools.delegate.capsule as capsule_store
from supporter.config import config
from supporter.tools.catalog import build_tool_catalog, select_delegate_tools
from supporter.tools.delegate.agents import (
    _cache,
    _cache_key,
    _create_sub_agent,
    _rotated_keys_for_role,
    _truncate_delegate_output,
    run_sub_agent,
)
from supporter.tools.delegate.capsule import create_capsule
from supporter.tools.delegate.opencode_backend import run_opencode
from supporter.tools.delegate.scheduler import (
    _execute_dag,
    _inject_dependency_context,
    _should_skip,
    run_heartbeat,
    run_milestone,
    serialize_results,
)
from supporter.tools.delegate.validation import _resolve_agent_profile, validate_tasks
from supporter.types import LLMResult, TaskCompleted, TaskOutputChunk, TaskStatus


@pytest.fixture(autouse=True)
def isolate_delegation_capsules(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    capsule_store._CAPSULE_LOCKS.clear()
    _cache.clear()


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
        validated = validate_tasks(tasks_json)
        assert (
            validated[0]["persona"]
            == config.delegate_agent_roster["test_engineer"]["persona"]
        )
        assert validated[0]["tools"] == {
            "read_file",
            "execute_bash",
        }


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
        validated = validate_tasks(tasks_json)
        assert len(validated) == 2
        assert validated[0]["id"] == "t1"
        assert validated[0]["tools"] == {
            "read_file",
            "execute_bash",
        }
        assert validated[1]["persona"] == "custom"
        assert validated[1]["context"] == "ctx"
        assert validated[1]["tools"] == set(
            select_delegate_tools(build_tool_catalog(), "all")
        )

    def test_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_tasks("not json")

    def test_not_array(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON array"):
            validate_tasks('{"id": "t1"}')

    def test_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="missing a valid string 'id'"):
            validate_tasks('[{"task": "no id"}]')
        with pytest.raises(ValueError, match="missing a valid string 'task'"):
            validate_tasks('[{"id": "no task"}]')

    def test_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="Duplicate task ID"):
            validate_tasks('[{"id": "t1", "task": "a"}, {"id": "t1", "task": "b"}]')

    def test_too_many(self) -> None:
        tasks = [
            {"id": f"t{i}", "task": "task"}
            for i in range(config.delegate_max_tasks + 1)
        ]
        with pytest.raises(ValueError, match="Too many tasks"):
            validate_tasks(json.dumps(tasks))

    def test_backend_defaults_to_gemini(self) -> None:
        validated = validate_tasks('[{"id": "t1", "task": "a"}]')
        assert validated[0]["backend"] == "gemini"

    def test_backend_explicit_is_case_insensitive(self) -> None:
        validated = validate_tasks('[{"id": "t1", "task": "a", "backend": "GEMINI"}]')
        assert validated[0]["backend"] == "gemini"

    def test_backend_opencode_accepted(self) -> None:
        validated = validate_tasks('[{"id": "t1", "task": "a", "backend": "opencode"}]')
        assert validated[0]["backend"] == "opencode"

    def test_backend_unknown_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown backend 'bogus'"):
            validate_tasks('[{"id": "t1", "task": "a", "backend": "bogus"}]')

    def test_backend_non_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-string 'backend'"):
            validate_tasks('[{"id": "t1", "task": "a", "backend": 123}]')

    def test_timeout_parsing(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "t1", "task": "a", "timeout": 10},
                {"id": "t2", "task": "b", "timeout": 1000},
                {"id": "t3", "task": "c"},
            ]
        )
        validated = validate_tasks(tasks_json)
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
        validated = validate_tasks(tasks_json)
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
            validate_tasks(tasks_json)

    def test_depends_on_self(self) -> None:
        tasks_json = json.dumps([{"id": "a", "task": "first", "depends_on": ["a"]}])
        with pytest.raises(ValueError, match="cannot depend on itself"):
            validate_tasks(tasks_json)

    def test_cycle_detection(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first", "depends_on": ["b"]},
                {"id": "b", "task": "second", "depends_on": ["a"]},
            ]
        )
        with pytest.raises(ValueError, match="cycle"):
            validate_tasks(tasks_json)

    def test_depends_on_string_format(self) -> None:
        tasks_json = json.dumps(
            [
                {"id": "a", "task": "first"},
                {"id": "b", "task": "second", "depends_on": "a"},
            ]
        )
        validated = validate_tasks(tasks_json)
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
        validated = validate_tasks(tasks_json)
        assert len(validated) == 5
        assert validated[4]["depends_on"] == ["c", "d"]


class TestToolRegistry:
    def test_scoping(self) -> None:
        registry = select_delegate_tools(
            build_tool_catalog(), {"read_file", "execute_bash", "unknown_tool"}
        )
        assert "read_file" in registry
        assert "execute_bash" in registry
        assert "write_file" not in registry
        assert "unknown_tool" not in registry

    def test_no_recursion(self) -> None:
        registry = select_delegate_tools(
            build_tool_catalog(), {"read_file", "delegate_tasks"}
        )
        assert "read_file" in registry
        assert "delegate_tasks" not in registry

    def test_registry_only_includes_explicit_allowed_tools(self) -> None:
        requested = {
            "read_file",
            "write_file",
            "execute_bash",
            "google_search",
            "delegate_tasks",
            "check_delegation",
            "query_delegation",
            "unknown_tool",
        }
        registry = select_delegate_tools(build_tool_catalog(), requested)
        assert set(registry) == {
            "read_file",
            "write_file",
            "execute_bash",
            "google_search",
        }


class TestSubAgentFactory:
    @patch("supporter.pool.get_provider")
    @patch("supporter.tools.delegate.agents.ChatAgent")
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
        assert '"confidence"' in prompt

    @patch("supporter.pool.get_provider")
    @patch("supporter.tools.delegate.agents.ChatAgent")
    def test_create_sub_agent_suppresses_result_contract(
        self, mock_agent_class: Any, mock_get_provider: Any
    ) -> None:
        task = {
            "id": "t1",
            "task": "my task",
            "persona": "my persona",
            "tools": {"read_file"},
            "model": "my-model",
            "context": "my context",
            "result_contract": False,
        }
        _agent, prompt = _create_sub_agent(task)
        assert '"confidence"' not in prompt


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

        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch(
            "supporter.tools.delegate.agents._create_sub_agent",
            return_value=(mock_agent, "prompt"),
        ):
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

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

        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch("supporter.tools.delegate.agents._create_sub_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.execute = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_factory.return_value = (mock_agent, "prompt")
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

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

        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch("supporter.tools.delegate.agents._create_sub_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.execute = AsyncMock(side_effect=RuntimeError("Boom"))
            mock_factory.return_value = (mock_agent, "prompt")
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

        assert result["status"] == TaskStatus.ERROR
        assert "Boom" in result["output"]

    @pytest.mark.asyncio
    async def test_opencode_backend_dispatch(self) -> None:
        task = {
            "id": "t1",
            "task": "task",
            "backend": "opencode",
            "tools": {"read_file"},
            "model": "m",
            "persona": "p",
            "context": "c",
            "timeout": 10,
            "max_retries": 0,
            "depends_on": [],
        }
        semaphore = asyncio.Semaphore(1)

        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)

        with patch(
            "supporter.tools.delegate.agents.run_opencode",
            new=AsyncMock(return_value=("changed a.py", "google/x", {})),
        ) as mock_oc:
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

        mock_oc.assert_awaited_once()
        assert result["status"] == TaskStatus.COMPLETED
        assert result["output"] == "changed a.py"
        assert result["model"] == "google/x"

    def test_build_spec_includes_result_contract(self) -> None:
        from supporter.tools.delegate.opencode_backend import _build_spec

        spec = _build_spec({"task": "do it", "context": "ctx"})
        assert "do it" in spec
        assert "ctx" in spec
        assert '"confidence"' in spec

    def test_build_spec_suppresses_result_contract(self) -> None:
        from supporter.tools.delegate.opencode_backend import _build_spec

        spec = _build_spec({"task": "do it", "result_contract": False})
        assert '"confidence"' not in spec


class TestOpenCodeStreaming:
    @pytest.mark.asyncio
    async def test_on_chunk_invoked_multiple_times(self) -> None:
        """Test that on_chunk is invoked multiple times for incremental output."""
        task = {
            "id": "t1",
            "task": "task",
            "timeout": 10,
        }

        chunks_received: list[str] = []

        # Create a fake process with mock stdout that returns data in chunks
        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(
            side_effect=[
                b"Line 1\n",
                b"Line 2\n",
                b"Line 3\n",
                b"",
            ]
        )
        fake_proc.returncode = 0
        fake_proc.wait = AsyncMock(return_value=None)
        fake_proc.kill = MagicMock()

        with (
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_binary",
                return_value="/fake/opencode",
            ),
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_repo",
                return_value="/repo",
            ),
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
        ):
            output, _model, _tokens = await run_opencode(
                task, on_chunk=chunks_received.append
            )

        assert len(chunks_received) == 3
        assert chunks_received[0] == "Line 1\n"
        assert chunks_received[1] == "Line 2\n"
        assert chunks_received[2] == "Line 3\n"
        assert output == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_no_callback_path_lossless(self) -> None:
        """Test that without on_chunk, output matches communicate() behavior."""
        task = {
            "id": "t1",
            "task": "task",
            "timeout": 10,
        }

        # Create a fake process with single output (simulating communicate)
        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(side_effect=[b"Single output line\n", b""])
        fake_proc.returncode = 0
        fake_proc.wait = AsyncMock(return_value=None)
        fake_proc.kill = MagicMock()

        with (
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_binary",
                return_value="/fake/opencode",
            ),
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_repo",
                return_value="/repo",
            ),
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
        ):
            _output, _model, _tokens = await run_opencode(task, on_chunk=None)

        assert _output == "Single output line"

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self) -> None:
        """Test that timeout kills and reaps the process."""
        task = {
            "id": "t1",
            "task": "task",
            "timeout": 0.01,
        }

        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()

        # Make read block forever so the timeout watchdog fires
        async def _hang(*_args: Any, **_kwargs: Any) -> bytes:
            await asyncio.sleep(10)
            return b""

        fake_proc.stdout.read = _hang
        fake_proc.kill = MagicMock()
        fake_proc.wait = AsyncMock(return_value=1)

        with (
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_binary",
                return_value="/fake/opencode",
            ),
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_repo",
                return_value="/repo",
            ),
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
            pytest.raises(TimeoutError),
        ):
            await run_opencode(task, on_chunk=None)

        fake_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_runtime_error(self) -> None:
        """Test that non-zero exit raises RuntimeError."""
        task = {
            "id": "t1",
            "task": "task",
            "timeout": 10,
        }

        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(side_effect=[b"Error output\n", b""])
        fake_proc.kill = MagicMock()
        fake_proc.returncode = 1
        fake_proc.wait = AsyncMock(return_value=1)

        with (
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_binary",
                return_value="/fake/opencode",
            ),
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_repo",
                return_value="/repo",
            ),
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
            pytest.raises(RuntimeError, match="opencode exited 1"),
        ):
            await run_opencode(task, on_chunk=None)

    @pytest.mark.asyncio
    async def test_run_sub_agent_publishes_chunks_for_opencode(self) -> None:
        """Test that run_sub_agent publishes TaskOutputChunk for opencode backend."""
        task = {
            "id": "t1",
            "task": "task",
            "backend": "opencode",
            "tools": {"read_file"},
            "model": "m",
            "persona": "p",
            "context": "c",
            "timeout": 10,
            "max_retries": 0,
            "depends_on": [],
        }
        semaphore = asyncio.Semaphore(1)

        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)
        mock_bus.publish = MagicMock()

        # Create a fake process that emits multiple chunks
        fake_proc = MagicMock()
        fake_proc.stdout = MagicMock()
        fake_proc.stdout.read = AsyncMock(
            side_effect=[b"Chunk1\n", b"Chunk2\n", b"Chunk3\n", b""]
        )
        fake_proc.returncode = 0
        fake_proc.kill = MagicMock()
        fake_proc.wait = AsyncMock(return_value=0)

        with (
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_binary",
                return_value="/fake/opencode",
            ),
            patch(
                "supporter.tools.delegate.opencode_backend._resolve_repo",
                return_value="/repo",
            ),
            patch("asyncio.create_subprocess_exec", return_value=fake_proc),
        ):
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

        assert result["status"] == TaskStatus.COMPLETED

        # Verify TaskOutputChunk events were published (3 chunks)
        chunk_events = [
            call.args[0]
            for call in mock_bus.publish.call_args_list
            if isinstance(call.args[0], TaskOutputChunk)
        ]
        assert len(chunk_events) == 3  # 3 chunks
        # Verify seq ordering
        seqs = [e.seq for e in chunk_events]
        assert seqs == [1, 2, 3]


class TestOutputTailTruncation:
    def test_truncate_output_tail_short_text(self) -> None:
        from supporter.tui.delegation_listener import _truncate_output_tail

        text = "Short text"
        assert _truncate_output_tail(text) == text

    def test_truncate_output_tail_many_lines(self) -> None:
        from supporter.tui.delegation_listener import _truncate_output_tail

        # Lines long enough that the total exceeds the char budget, so the
        # last-N-lines tail genuinely kicks in.
        lines = [f"Line {i}: " + "x" * 100 for i in range(10)]
        text = "\n".join(lines)
        result = _truncate_output_tail(text)
        assert "Line 9" in result
        assert "Line 0" not in result

    def test_truncate_output_tail_long_single_line(self) -> None:
        from supporter.tui.delegation_listener import _truncate_output_tail

        text = "x" * 600
        result = _truncate_output_tail(text)
        assert len(result) <= 503  # Some buffer for the truncation logic

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

        with (
            patch("supporter.tools.delegate.scheduler.run_sub_agent") as mock_run,
            patch.object(config, "delegate_result_repair", False),
        ):
            mock_run.side_effect = [
                {"id": "a", "status": "completed", "output": "out_a", "duration": 1.0},
                {"id": "b", "status": "completed", "output": "out_b", "duration": 1.0},
            ]
            from supporter.tools.delegate.bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            await create_capsule("test_job", "test", tasks, 5)
            results = await _execute_dag(
                tasks, semaphore, semaphore, mock_bus, "test_job"
            )

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

        with (
            patch("supporter.tools.delegate.scheduler.run_sub_agent") as mock_run,
            patch.object(config, "delegate_result_repair", False),
        ):
            mock_run.side_effect = [
                {"id": "a", "status": "completed", "output": "out_a", "duration": 1.0},
                {"id": "b", "status": "completed", "output": "out_b", "duration": 1.0},
            ]
            from supporter.tools.delegate.bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            await create_capsule("test_job", "test", tasks, 5)
            results = await _execute_dag(
                tasks, semaphore, semaphore, mock_bus, "test_job"
            )

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

        with patch("supporter.tools.delegate.scheduler.run_sub_agent") as mock_run:
            mock_run.return_value = {
                "id": "a",
                "status": "error",
                "output": "crashed",
                "duration": 0.5,
            }
            from supporter.tools.delegate.bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            await create_capsule("test_job", "test", tasks, 5)
            results = await _execute_dag(
                tasks, semaphore, semaphore, mock_bus, "test_job"
            )

        assert results[0]["status"] == TaskStatus.ERROR
        assert results[1]["status"] == TaskStatus.SKIPPED
        assert "Dependency 'a'" in results[1]["output"]
        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_completed_event_includes_capsule_summary_fields(self) -> None:
        task_output = (
            "Mapped delegation internals.\n\n"
            "DELEGATION_RESULT:\n"
            "{\n"
            '  "summary": "Mapped delegate and tui flow",\n'
            '  "evidence": {\n'
            '    "files_read": ["src/supporter/tools/delegate.py"],\n'
            '    "files_changed": ["src/supporter/tui/__init__.py"],\n'
            '    "commands_run": ["pytest tests/unit/test_delegate.py"],\n'
            '    "sources": ["local_repo"]\n'
            "  },\n"
            '  "findings": ["Task completion bubbles should be compact"],\n'
            '  "handoff": "Render capsule fields in task completion UI",\n'
            '  "confidence": "high"\n'
            "}"
        )
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
            }
        ]
        semaphore = asyncio.Semaphore(2)

        with patch("supporter.tools.delegate.scheduler.run_sub_agent") as mock_run:
            mock_run.return_value = {
                "id": "a",
                "status": TaskStatus.COMPLETED,
                "output": task_output,
                "duration": 1.0,
                "model": "m",
            }
            from supporter.tools.delegate.bus import DelegationBus

            mock_bus = MagicMock(spec=DelegationBus)
            await create_capsule("test_job", "test", tasks, 2)
            await _execute_dag(tasks, semaphore, semaphore, mock_bus, "test_job")

        completion_events = [
            call.args[0]
            for call in mock_bus.publish.call_args_list
            if isinstance(call.args[0], TaskCompleted)
        ]
        assert len(completion_events) == 1
        event = completion_events[0]
        assert event.summary == "Mapped delegate and tui flow"
        assert event.confidence == "high"
        assert event.findings_count == 1
        assert event.evidence_counts == {
            "files_read": 1,
            "files_changed": 1,
            "commands_run": 1,
            "sources": 1,
        }
        assert event.handoff == "Render capsule fields in task completion UI"


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
        validated = validate_tasks(tasks_json)
        assert validated[0]["tolerate_failures"] is True

    def test_validation_defaults_tolerate_failures_false(self) -> None:
        tasks_json = json.dumps([{"id": "t1", "task": "do it"}])
        validated = validate_tasks(tasks_json)
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


def test_validate_tasks_edge_normalization_paths() -> None:
    tasks_json = json.dumps(
        [
            {
                "id": "t1",
                "task": "one",
                "depends_on": 123,
                "pre_approved_commands": "not-list",
            }
        ]
    )
    validated = validate_tasks(tasks_json)
    assert validated[0]["depends_on"] == []
    assert validated[0]["pre_approved_commands"] == []

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_tasks("[]")

    with pytest.raises(ValueError, match="must be an object"):
        validate_tasks(json.dumps(["bad"]))


def test_truncate_delegate_output_branch() -> None:
    text = "x" * (config.delegate_max_output_chars + 1)
    out = _truncate_delegate_output(text)
    assert out.endswith("[Output truncated...]")


def test_inject_dependency_context_skips_missing_deps() -> None:
    task = {"depends_on": ["missing"], "context": "ctx"}
    enriched = _inject_dependency_context(task, {})
    assert enriched is task


@pytest.mark.asyncio
async def test_execute_dag_timeout_branch_publishes_timeout() -> None:
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
        }
    ]
    semaphore = asyncio.Semaphore(1)
    with patch("supporter.tools.delegate.scheduler.run_sub_agent") as mock_run:
        mock_run.return_value = {
            "id": "a",
            "status": TaskStatus.TIMEOUT,
            "output": "timeout",
            "duration": 10.0,
            "model": "m",
            "tokens": {},
        }
        from supporter.tools.delegate.bus import DelegationBus

        mock_bus = MagicMock(spec=DelegationBus)
        await create_capsule("job", "test", tasks, 1)
        results = await _execute_dag(tasks, semaphore, semaphore, mock_bus, "job")

    assert results[0]["status"] == TaskStatus.TIMEOUT
    published = [
        call.args[0].__class__.__name__ for call in mock_bus.publish.call_args_list
    ]
    assert "TaskTimedOut" in published


@pytest.mark.asyncio
async def test_run_heartbeat_emits_anomaly() -> None:
    bus = MagicMock()
    bus.milestone = "M"
    state = {
        "status": "RUNNING",
        "agent_label": "a",
        "started_at": 0.0,
        "timeout": 10.0,
        "anomaly_fired": False,
    }
    bus.get_snapshot.return_value = {"t1": state}

    with (
        patch(
            "supporter.tools.delegate.scheduler.bus_exists",
            side_effect=[True, True, False],
        ),
        patch("supporter.tools.delegate.scheduler.time.monotonic", return_value=100.0),
        patch("supporter.tools.delegate.scheduler.asyncio.sleep", new=AsyncMock()),
        patch("supporter.tools.delegate.scheduler.DELEGATE_ANOMALY_THRESHOLD", 0.1),
    ):
        await run_heartbeat(bus, "hbjob", interval=0)

    published = [call.args[0].__class__.__name__ for call in bus.publish.call_args_list]
    assert "HeartbeatTick" in published
    assert "TaskAnomaly" in published
    bus.update_task_state.assert_called_once()


@pytest.mark.asyncio
async def test_run_sub_agent_retry_backoff_and_continue_branch() -> None:
    task = {
        "id": "r1",
        "task": "retry me",
        "agent": "custom",
        "tools": {"read_file"},
        "model": "m",
        "persona": "p",
        "context": "",
        "timeout": 10,
        "max_retries": 1,
    }
    bus = MagicMock()
    semaphore = asyncio.Semaphore(1)
    calls = {"n": 0}

    async def execute(_: str) -> LLMResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first fail")
        return LLMResult(text="ok", model="m")

    agent = MagicMock()
    agent.execute = execute
    with (
        patch(
            "supporter.tools.delegate.agents._create_sub_agent",
            return_value=(agent, "p"),
        ),
        patch("supporter.tools.delegate.scheduler.asyncio.sleep", new=AsyncMock()),
    ):
        result = await run_sub_agent(task, semaphore, bus, "job")
    assert result["status"] == TaskStatus.COMPLETED
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_run_sub_agent_negative_retries_returns_empty_last_result() -> None:
    task = {
        "id": "none",
        "task": "noop",
        "max_retries": -1,
    }
    result = await run_sub_agent(task, asyncio.Semaphore(1), MagicMock(), "job")
    assert result == {}


@pytest.mark.asyncio
async def test_run_heartbeat_early_stop_after_sleep() -> None:
    bus = MagicMock()
    with (
        patch(
            "supporter.tools.delegate.scheduler.bus_exists", side_effect=[True, False]
        ),
        patch("supporter.tools.delegate.scheduler.asyncio.sleep", new=AsyncMock()),
    ):
        await run_heartbeat(bus, "j1", interval=1)
    bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_run_heartbeat_skips_non_running_task_state() -> None:
    bus = MagicMock()
    bus.milestone = "M"
    bus.get_snapshot.return_value = {"t1": {"status": "DONE"}}
    with (
        patch(
            "supporter.tools.delegate.scheduler.bus_exists",
            side_effect=[True, True, False],
        ),
        patch("supporter.tools.delegate.scheduler.asyncio.sleep", new=AsyncMock()),
        patch("supporter.tools.delegate.scheduler.time.monotonic", return_value=10.0),
    ):
        await run_heartbeat(bus, "j2", interval=0)
    published = [call.args[0].__class__.__name__ for call in bus.publish.call_args_list]
    assert published == ["HeartbeatTick"]


@pytest.mark.asyncio
async def test_run_milestone_propagates_terminal_capsule_failure() -> None:
    bus = MagicMock()
    boom = RuntimeError("capsule terminal write failed")

    with (
        patch(
            "supporter.tools.delegate.scheduler._execute_dag",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "supporter.tools.delegate.scheduler.mark_capsule_completed",
            new=AsyncMock(side_effect=boom),
        ),
        patch("supporter.tools.delegate.scheduler.remove_bus"),
        pytest.raises(RuntimeError, match="capsule terminal write failed"),
    ):
        await run_milestone(
            "m",
            [],
            asyncio.Semaphore(1),
            asyncio.Semaphore(1),
            bus,
            "milestone_fail",
        )


class TestCacheKeyFunctions:
    def test_cache_key_with_valid_role(self) -> None:
        task = {
            "agent": "security_auditor",
            "model": "gemini-1.5-pro",
            "live": True,
            "task": "audit the code",
        }
        result = _cache_key(task)
        assert result == ("security_auditor", "gemini-1.5-pro", True)

        task["live"] = False
        result = _cache_key(task)
        assert result == ("security_auditor", "gemini-1.5-pro", False)

        del task["live"]
        result = _cache_key(task)
        assert result == ("security_auditor", "gemini-1.5-pro", False)

    def test_cache_key_with_invalid_role(self) -> None:
        td: dict[str, Any] = {"agent": "", "model": "gemini-1.5-pro", "task": "test"}
        result = _cache_key(td)
        assert result is None

        td = {"agent": None, "model": "gemini-1.5-pro", "task": "test"}
        result = _cache_key(td)
        assert result is None

        td["agent"] = "custom"
        result = _cache_key(td)
        assert result is None

        del td["agent"]
        result = _cache_key(td)
        assert result is None

    def test_rotated_keys_for_role_value_error_path(self) -> None:
        with (
            patch("supporter.config.config.gemini_api_keys", []),
            pytest.raises(ValueError, match="GEMINI_API_KEYS is missing/empty"),
        ):
            _rotated_keys_for_role("security_auditor")

    def test_rotated_keys_for_role_rotation_logic(self) -> None:
        _cache.role_offsets.clear()
        _cache.offset_counter = 0
        with patch("supporter.config.config.gemini_api_keys", ["key1", "key2", "key3"]):
            result_a = _rotated_keys_for_role("role_a")
            result_b = _rotated_keys_for_role("role_b")
            result_c = _rotated_keys_for_role("role_c")
            result_a2 = _rotated_keys_for_role("role_a")

        assert result_a == ["key1", "key2", "key3"]
        assert result_b == ["key2", "key3", "key1"]
        assert result_c == ["key3", "key1", "key2"]
        assert result_a2 == result_a

    def test_rotated_keys_for_role_different_roles(self) -> None:
        with patch("supporter.config.config.gemini_api_keys", ["key1", "key2"]):
            _cache.role_offsets.clear()
            _cache.offset_counter = 0

            result1 = _rotated_keys_for_role("security_auditor")
            result2 = _rotated_keys_for_role("test_engineer")

            assert len(result1) == 2
            assert len(result2) == 2
            assert set(result1) == {"key1", "key2"}
            assert set(result2) == {"key1", "key2"}


class TestCreateSubAgent:
    @patch("supporter.tools.delegate.agents._build_dedicated_provider")
    @patch("supporter.pool.get_provider")
    @patch("supporter.tools.delegate.agents.ChatAgent")
    def test_create_sub_agent_cache_hit_path(
        self,
        mock_agent_class: Any,
        mock_get_provider: Any,
        mock_build_provider: Any,
    ) -> None:
        task = {
            "id": "t1",
            "task": "my task",
            "agent": "security_auditor",
            "model": "gemini-1.5-pro",
            "tools": {"read_file"},
            "context": "my context",
            "persona": "test persona",
        }

        cache_key = _cache_key(task)
        assert cache_key is not None
        cached_agent = MagicMock()
        _cache.agents[cache_key] = cached_agent

        agent, prompt = _create_sub_agent(task)

        assert agent is cached_agent
        assert agent.history == []
        assert agent.current_interaction_id is None

        assert "my task" in prompt
        assert "my context" in prompt

        mock_agent_class.assert_not_called()
        mock_get_provider.assert_not_called()
        mock_build_provider.assert_not_called()

    @patch("supporter.tools.delegate.agents._build_dedicated_provider")
    @patch("supporter.pool.get_provider")
    @patch("supporter.tools.delegate.agents.ChatAgent")
    def test_create_sub_agent_cache_key_with_new_agent_path(
        self,
        mock_agent_class: Any,
        mock_get_provider: Any,
        mock_build_provider: Any,
    ) -> None:
        task = {
            "id": "t1",
            "task": "my task",
            "agent": "security_auditor",
            "model": "gemini-1.5-pro",
            "tools": {"read_file"},
            "context": "my context",
            "persona": "test persona",
        }

        mock_provider = MagicMock()
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_build_provider.return_value = mock_provider

        agent1, prompt1 = _create_sub_agent(task)

        assert agent1 is mock_agent
        cache_key = _cache_key(task)
        assert cache_key is not None
        assert _cache.agents[cache_key] is mock_agent
        assert cache_key in _cache.locks

        mock_build_provider.assert_called_once()

        agent2, prompt2 = _create_sub_agent(task)

        assert agent2 is mock_agent
        assert agent2 is agent1

        assert prompt1 == prompt2


class TestTruncateDelegateOutput:
    def test_truncate_delegate_output_truncation_path(self) -> None:
        long_text = "x" * (config.delegate_max_output_chars + 100)

        result = _truncate_delegate_output(long_text)

        assert len(result) <= config.delegate_max_output_chars + len(
            "\n\n[Output truncated...]"
        )
        assert result.endswith("[Output truncated...]")

        assert result.startswith("x" * config.delegate_max_output_chars)

        assert len(result) < len(long_text)

    def test_truncate_delegate_output_no_truncation(self) -> None:
        short_text = "x" * (config.delegate_max_output_chars - 10)

        result = _truncate_delegate_output(short_text)

        assert result == short_text
        assert not result.endswith("[Output truncated...]")


class TestRunSubAgentFinallyBlock:
    @pytest.mark.asyncio
    async def test_finally_block_close_path(self) -> None:
        task = {
            "id": "t1",
            "task": "test task",
            "agent": "custom",
            "live": True,
            "model": "gemini-1.5-pro",
            "tools": {"read_file"},
            "timeout": 10,
            "max_retries": 0,
        }

        semaphore = asyncio.Semaphore(1)
        mock_bus = MagicMock()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Done"
        mock_agent.execute = AsyncMock(return_value=mock_result)

        mock_provider = MagicMock()
        mock_provider.close = AsyncMock(side_effect=RuntimeError("Close failed"))
        mock_agent.provider = mock_provider

        with (
            patch("supporter.tools.delegate.agents._cache_key", return_value=None),
            patch(
                "supporter.tools.delegate.agents._create_sub_agent",
                return_value=(mock_agent, "prompt"),
            ),
        ):
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

            assert result["status"] == "completed"

            mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_finally_block_not_triggered(self) -> None:
        task = {
            "id": "t1",
            "task": "test task",
            "agent": "custom",
            "live": False,
            "model": "gemini-1.5-pro",
            "tools": {"read_file"},
            "timeout": 10,
            "max_retries": 0,
        }

        semaphore = asyncio.Semaphore(1)
        mock_bus = MagicMock()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Done"
        mock_agent.execute = AsyncMock(return_value=mock_result)

        with patch(
            "supporter.tools.delegate.agents._create_sub_agent",
            return_value=(mock_agent, "prompt"),
        ):
            result = await run_sub_agent(task, semaphore, mock_bus, "job1")

            assert result["status"] == "completed"

            mock_agent.provider.close.assert_not_called()


def test_global_semaphore_is_single_shared_object() -> None:
    """Regression: delegate and resume paths must share one global semaphore.

    Previously api.py held its own module-level _GLOBAL_SEMAPHORE while
    scheduler.py used a separate lazy singleton, letting concurrent
    resume + new-delegation reach 2x the global hard cap (SPEC §10).
    """
    from supporter.tools.delegate import api
    from supporter.tools.delegate.scheduler import _get_global_semaphore

    # __dict__ access avoids ruff B009 rewriting getattr and mypy's
    # no-implicit-reexport attr check on the re-imported private name.
    assert api.__dict__["_get_global_semaphore"] is _get_global_semaphore
    assert _get_global_semaphore() is _get_global_semaphore()
    assert getattr(api, "_GLOBAL_SEMAPHORE", None) is None
