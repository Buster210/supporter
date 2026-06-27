"""Tests for the objective tier-1 dispatcher and its config wiring."""

import asyncio
import json
import shutil
import stat
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.config import (
    DELEGATE_TIER1_COMMANDS,
    _cmd_list_env,
)
from supporter.config import (
    config as app_config,
)
from supporter.tools.bash import sandbox as bash_sandbox
from supporter.tools.delegate import qa_gate, tier1_checks
from supporter.tools.delegate.tier1_checks import (
    Tier1ToolUnavailable,
    _detect_node,
    _detect_python,
    resolve_tier1_commands,
    run_objective_tier1,
)


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestResolveTool:
    def test_prefers_repo_venv_binary(self, tmp_path: Path) -> None:
        _touch_executable(tmp_path / ".venv" / "bin" / "ruff")
        with patch.object(shutil, "which", return_value=None):
            resolved = tier1_checks._resolve_tool(tmp_path, "ruff")
        assert resolved is not None
        assert resolved[0] == str((tmp_path / ".venv" / "bin" / "ruff").resolve())
        assert len(resolved) == 1

    def test_falls_back_to_uv_run(self, tmp_path: Path) -> None:
        with patch.object(shutil, "which", return_value="/usr/bin/uv") as which:
            resolved = tier1_checks._resolve_tool(tmp_path, "mypy")
        assert resolved == ["uv", "run", "mypy"]
        which.assert_any_call("uv")

    def test_falls_back_to_host_tool(self, tmp_path: Path) -> None:
        def fake_which(name: str) -> str | None:
            return "/usr/bin/pytest" if name == "pytest" else None

        with patch.object(shutil, "which", side_effect=fake_which):
            resolved = tier1_checks._resolve_tool(tmp_path, "pytest")
        assert resolved == ["/usr/bin/pytest"]

    def test_returns_none_when_unavailable(self, tmp_path: Path) -> None:
        with patch.object(shutil, "which", return_value=None):
            assert tier1_checks._resolve_tool(tmp_path, "nope") is None


class TestDetectPython:
    def test_returns_venv_bin_paths_for_configured_tools(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.ruff]\n[tool.mypy]\n[tool.pytest.ini_options]\n")
        for tool in ("ruff", "mypy", "pytest"):
            _touch_executable(tmp_path / ".venv" / "bin" / tool)
        with patch.object(shutil, "which", return_value=None):
            commands = _detect_python(tmp_path)
        assert len(commands) == 3
        for cmd in commands:
            assert cmd[0].endswith(("/ruff", "/mypy", "/pytest"))
        # Verify each tool got its expected trailing args (resolving prefixes
        # can be more than the tool name when uv is used).
        joined = [" ".join(cmd) for cmd in commands]
        assert any("ruff check ." in j for j in joined)
        assert any("mypy ." in j for j in joined)
        assert any("pytest -q" in j for j in joined)

    def test_omits_unconfigured_tool(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
        _touch_executable(tmp_path / ".venv" / "bin" / "ruff")
        with patch.object(shutil, "which", return_value=None):
            commands = _detect_python(tmp_path)
        assert len(commands) == 1
        assert commands[0][0].endswith("/ruff")

    def test_top_level_mypy_section_recognized(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[mypy]\n")
        _touch_executable(tmp_path / ".venv" / "bin" / "mypy")
        with patch.object(shutil, "which", return_value=None):
            commands = _detect_python(tmp_path)
        assert len(commands) == 1

    def test_pytest_included_when_tests_dir_exists(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "tests").mkdir()
        _touch_executable(tmp_path / ".venv" / "bin" / "pytest")
        with patch.object(shutil, "which", return_value=None):
            commands = _detect_python(tmp_path)
        assert len(commands) == 1
        assert commands[0][-1] == "-q"

    def test_no_markers_and_empty_config_yields_empty(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        with patch.object(shutil, "which", return_value=None):
            assert _detect_python(tmp_path) == []

    def test_no_pyproject_yields_empty(self, tmp_path: Path) -> None:
        assert _detect_python(tmp_path) == []

    def test_invalid_toml_in_pyproject_yields_empty(self, tmp_path: Path) -> None:
        """TOMLDecodeError branch in _pyproject_has_section must not raise."""
        (tmp_path / "pyproject.toml").write_text("[[[ invalid")
        # No tests/ dir, no venv; invalid TOML must not raise, just return [].
        with patch.object(shutil, "which", return_value=None):
            result = _detect_python(tmp_path)
        assert result == []

    def test_drops_tools_that_cannot_be_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n[tool.mypy]\n")
        with patch.object(shutil, "which", return_value=None):
            assert _detect_python(tmp_path) == []


class TestDetectNode:
    def test_returns_defined_scripts_only(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            json.dumps(
                {
                    "scripts": {
                        "lint": "eslint .",
                        "test": "vitest run",
                        "build": "tsc",
                    }
                }
            )
        )
        _touch_executable(tmp_path / ".venv" / "bin" / "npm")
        with patch.object(shutil, "which", return_value=None):
            commands = _detect_node(tmp_path)
        assert len(commands) == 2
        joined = [" ".join(cmd) for cmd in commands]
        assert any("npm run lint" in j for j in joined)
        assert any("npm run test" in j for j in joined)

    def test_no_package_json(self, tmp_path: Path) -> None:
        assert _detect_node(tmp_path) == []

    def test_no_scripts_object(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(json.dumps({}))
        assert _detect_node(tmp_path) == []

    def test_invalid_json_in_package_json_yields_empty(self, tmp_path: Path) -> None:
        """Covers tier1_checks.py:105-106 — JSONDecodeError branch."""
        (tmp_path / "package.json").write_text("{ not valid json")
        assert _detect_node(tmp_path) == []

    def test_npm_not_resolvable_yields_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers tier1_checks.py:113 — npm not found branch."""
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"lint": "eslint ."}})
        )
        monkeypatch.setattr(tier1_checks, "_resolve_tool", lambda repo, tool: None)
        assert _detect_node(tmp_path) == []


class TestResolveTier1Commands:
    def test_explicit_config_override_returned_verbatim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override: list[list[str]] = [["echo", "hi"], ["pytest", "-q"]]
        monkeypatch.setattr(app_config, "delegate_tier1_commands", override)
        assert resolve_tier1_commands(tmp_path) == override
        # Detection must not have run -- no pyproject present, no .venv/bin.
        assert (tmp_path / "pyproject.toml").exists() is False

    def test_config_override_is_a_copy(self, tmp_path: Path) -> None:
        cmd: list[str] = ["echo", "x"]
        original: list[list[str]] = [cmd]
        with patch.object(app_config, "delegate_tier1_commands", original):
            out = resolve_tier1_commands(tmp_path)
        assert out[0] == ["echo", "x"]
        # Returned structure must be independent of the source list.
        out[0].append("mutated")
        assert cmd == ["echo", "x"]

    def test_get_commands_uses_config_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers tier1_checks.py:130 — config override bypasses auto-detection."""
        monkeypatch.setattr(app_config, "delegate_tier1_commands", [["pytest", "-v"]])
        result = resolve_tier1_commands(tmp_path)
        assert result == [["pytest", "-v"]]
        # No pyproject or package.json needed — detection must have been skipped.
        assert not (tmp_path / "pyproject.toml").exists()


class TestRunObjectiveTier1:
    @pytest.mark.asyncio
    async def test_all_zero_returns_true(self, tmp_path: Path) -> None:
        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"out", b""))
        fake_proc.returncode = 0
        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ) as wrap,
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc)
            ),
        ):
            ok, report = await run_objective_tier1(
                tmp_path, [["echo", "a"], ["echo", "b"]], timeout=5.0
            )
        assert ok is True
        assert "echo a" in report and "echo b" in report
        assert wrap.call_count == 2
        for call in wrap.call_args_list:
            assert call.kwargs["cwd"] == tmp_path
            assert call.kwargs["root"] == tmp_path

    @pytest.mark.asyncio
    async def test_first_nonzero_short_circuits(self, tmp_path: Path) -> None:
        async def make_proc(rc: int) -> Any:
            p = MagicMock()
            p.communicate = AsyncMock(return_value=(b"oops\n", b""))
            p.returncode = rc
            return p

        procs = [await make_proc(0), await make_proc(2), await make_proc(0)]
        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(side_effect=procs)
            ),
        ):
            ok, report = await run_objective_tier1(
                tmp_path,
                [["alpha"], ["beta"], ["gamma"]],
                timeout=5.0,
            )
        assert ok is False
        # The failing argv (beta) must appear; later commands must NOT have run.
        assert "beta" in report
        assert "gamma" not in report
        assert "exit 2" in report

    @pytest.mark.asyncio
    async def test_pytest_exit_5_counts_as_pass(self, tmp_path: Path) -> None:
        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b"no tests", b""))
        fake_proc.returncode = 5
        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc)
            ),
        ):
            ok, report = await run_objective_tier1(
                tmp_path, [["pytest", "-q"]], timeout=5.0
            )
        assert ok is True
        assert "pytest" in report

    @pytest.mark.asyncio
    async def test_timeout_kills_proc(self, tmp_path: Path) -> None:
        fake_proc = MagicMock()

        async def hang() -> tuple[bytes, bytes]:
            raise TimeoutError

        fake_proc.communicate = hang
        fake_proc.kill = MagicMock()
        fake_proc.wait = AsyncMock()

        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc)
            ),
        ):
            ok, report = await run_objective_tier1(tmp_path, [["slow"]], timeout=0.1)
        assert ok is False
        assert "timed out" in report
        fake_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_truncated_when_exceeds_max_chars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers tier1_checks.py:186 — long output is truncated with a marker."""
        monkeypatch.setattr(app_config, "delegate_max_output_chars", 10)
        long_output = b"A" * 200
        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(long_output, b""))
        fake_proc.returncode = 0
        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=fake_proc)
            ),
        ):
            ok, report = await run_objective_tier1(
                tmp_path, [["echo", "lots"]], timeout=5.0
            )
        assert ok is True
        assert "[Output truncated...]" in report

    @pytest.mark.asyncio
    async def test_sandbox_unavailable_raises_tool_unavailable(
        self, tmp_path: Path
    ) -> None:
        def boom(argv: list[str], cwd: Path, root: Path) -> list[str]:
            raise RuntimeError("Security Block: Sandbox tool not found")

        with (
            patch.object(bash_sandbox, "wrap_in_sandbox", side_effect=boom),
            pytest.raises(Tier1ToolUnavailable),
        ):
            await run_objective_tier1(tmp_path, [["ruff", "check"]], timeout=5.0)

    @pytest.mark.asyncio
    async def test_create_subprocess_exec_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        with (
            patch.object(
                bash_sandbox,
                "wrap_in_sandbox",
                side_effect=lambda argv, cwd, root: list(argv),
            ),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(side_effect=FileNotFoundError("nope")),
            ),
            pytest.raises(Tier1ToolUnavailable),
        ):
            await run_objective_tier1(
                tmp_path, [["tool-that-does-not-exist"]], timeout=5.0
            )


class TestTier1Dispatch:
    @pytest.mark.asyncio
    async def test_objective_path_does_not_invoke_sub_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            qa_gate, "resolve_tier1_commands", lambda _repo: [["echo", "ok"]]
        )

        async def fake_objective(
            repo: Path,
            commands: list[list[str]],
            timeout: float,  # noqa: ASYNC109
        ) -> tuple[bool, str]:
            return True, "objective report"

        monkeypatch.setattr(qa_gate, "run_objective_tier1", fake_objective)
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as sub:
            ok, out = await qa_gate._tier1(
                {"id": "t", "task": "x", "timeout": 5},
                0,
                asyncio.Semaphore(1),
                MagicMock(),
                "job",
            )
        sub.assert_not_awaited()
        assert ok is True
        assert out == "objective report"

    @pytest.mark.asyncio
    async def test_empty_commands_falls_back_to_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(qa_gate, "resolve_tier1_commands", lambda _repo: [])

        async def fake_llm(*_a: Any, **_k: Any) -> tuple[bool, str]:
            return True, "llm report"

        monkeypatch.setattr(qa_gate, "_tier1_llm", fake_llm)
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as sub:
            ok, out = await qa_gate._tier1(
                {"id": "t", "task": "x", "timeout": 5},
                0,
                asyncio.Semaphore(1),
                MagicMock(),
                "job",
            )
        sub.assert_not_awaited()
        assert ok is True
        assert out == "llm report"

    @pytest.mark.asyncio
    async def test_tool_unavailable_falls_back_to_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            qa_gate, "resolve_tier1_commands", lambda _repo: [["missing"]]
        )

        async def fake_objective(
            repo: Path,
            commands: list[list[str]],
            timeout: float,  # noqa: ASYNC109
        ) -> tuple[bool, str]:
            raise Tier1ToolUnavailable("no tool")

        async def fake_llm(*_a: Any, **_k: Any) -> tuple[bool, str]:
            return True, "llm fallback report"

        monkeypatch.setattr(qa_gate, "run_objective_tier1", fake_objective)
        monkeypatch.setattr(qa_gate, "_tier1_llm", fake_llm)
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as sub:
            ok, out = await qa_gate._tier1(
                {"id": "t", "task": "x", "timeout": 5},
                0,
                asyncio.Semaphore(1),
                MagicMock(),
                "job",
            )
        sub.assert_not_awaited()
        assert ok is True
        assert out == "llm fallback report"


class TestCmdListEnv:
    def test_default_returned_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("_TEST_CMD_LIST", raising=False)
        assert _cmd_list_env("_TEST_CMD_LIST", [["x"]]) == [["x"]]

    def test_default_returned_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_CMD_LIST", "")
        assert _cmd_list_env("_TEST_CMD_LIST", [["default"]]) == [["default"]]

    def test_parses_json_array_of_arrays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_CMD_LIST", '[["ruff","check","."],["pytest","-q"]]')
        assert _cmd_list_env("_TEST_CMD_LIST", DELEGATE_TIER1_COMMANDS) == [
            ["ruff", "check", "."],
            ["pytest", "-q"],
        ]

    def test_malformed_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_CMD_LIST", "[broken")
        with pytest.raises(ValueError, match="JSON array of string arrays"):
            _cmd_list_env("_TEST_CMD_LIST", DELEGATE_TIER1_COMMANDS)

    def test_wrong_shape_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_CMD_LIST", '["a", "b"]')
        with pytest.raises(ValueError, match="JSON array of string arrays"):
            _cmd_list_env("_TEST_CMD_LIST", DELEGATE_TIER1_COMMANDS)

    def test_non_string_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_TEST_CMD_LIST", '[["echo", 1]]')
        with pytest.raises(ValueError, match="JSON array of string arrays"):
            _cmd_list_env("_TEST_CMD_LIST", DELEGATE_TIER1_COMMANDS)
