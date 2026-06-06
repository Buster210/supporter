"""Objective tier-1 checks for the QA gate (SPEC §7).

Runs the project's configured build/type/lint/test commands with REAL exit
codes instead of an LLM self-report. The default path auto-detects the
ecosystem from ``pyproject.toml`` / ``package.json`` markers; an explicit
``config.delegate_tier1_commands`` overrides detection. On any unexecutable
tool (sandbox bind-mounts only the repo root on Linux nsjail), a sentinel
exception is raised so the dispatcher can fall back to the LLM tier-1
worker rather than reporting a false FAIL.
"""

import asyncio
import contextlib
import json
import shutil
import tomllib
from pathlib import Path

from ...config import config
from ...logger import logger
from ..bash import sandbox


# Raised when the sandbox cannot execute a configured tool (e.g. the repo's
# .venv/bin is missing, or the host tool is outside the nsjail bind-mount).
# The dispatcher catches it and falls back to the LLM tier-1 worker.
class Tier1ToolUnavailable(RuntimeError):  # noqa: N818 - "Unavailable" is the sentinel suffix
    """Tier-1 tool could not be executed; caller should fall back to LLM."""


def _resolve_tool(repo: Path, tool: str) -> list[str] | None:
    """Return an argv prefix to invoke ``tool`` inside the repo's sandbox.

    Preference order: repo's own ``.venv/bin/<tool>`` (the bind-mounted path),
    then ``uv run <tool>`` (so the local venv is used even when sandbox can't
    see the host Python), then whatever ``shutil.which`` finds on the host.
    ``None`` means the tool is not available and the caller should drop the
    corresponding command.
    """
    venv_bin = repo / ".venv" / "bin" / tool
    if venv_bin.is_file() and (venv_bin.stat().st_mode & 0o111):
        return [str(venv_bin.resolve())]
    if shutil.which("uv"):
        return ["uv", "run", tool]
    host = shutil.which(tool)
    if host:
        return [host]
    return None


def _pyproject_has_section(pyproject: Path, section: str) -> bool:
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except OSError, tomllib.TOMLDecodeError:
        return False
    if "tool" in data and isinstance(data["tool"], dict) and section in data["tool"]:
        return True
    # mypy is sometimes declared at the top level: [mypy]
    return section in data


def _detect_python(repo: Path) -> list[list[str]]:
    """Return ordered tier-1 argv lists for a Python repo, gated on real config.

    Empty ``pyproject.toml`` with no markers yields ``[]`` so we never emit a
    command just because the file exists. A tool is included only if the repo
    *configures* it: ``[tool.ruff]``, ``[tool.mypy]`` (or a top-level
    ``[mypy]``), and ``[tool.pytest.ini_options]`` or a ``tests/`` dir for
    pytest.
    """
    pyproject = repo / "pyproject.toml"
    candidates: list[tuple[str, list[str]]] = []

    if _pyproject_has_section(pyproject, "ruff"):
        candidates.append(("ruff", ["check", "."]))
    if _pyproject_has_section(pyproject, "mypy"):
        candidates.append(("mypy", ["."]))
    has_pytest_section = _pyproject_has_section(pyproject, "pytest")
    has_tests_dir = (repo / "tests").is_dir()
    if has_pytest_section or has_tests_dir:
        candidates.append(("pytest", ["-q"]))

    commands: list[list[str]] = []
    for tool, args in candidates:
        resolved = _resolve_tool(repo, tool)
        if resolved is None:
            continue
        commands.append([*resolved, *args])
    return commands


def _detect_node(repo: Path) -> list[list[str]]:
    """Return ordered tier-1 argv lists for a Node repo, gated on real config.

    Includes a script only if ``package.json`` defines it; missing scripts
    are silently skipped.
    """
    pkg = repo / "package.json"
    if not pkg.is_file():
        return []
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return []
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict):
        return []

    resolved_npm = _resolve_tool(repo, "npm")
    if resolved_npm is None:
        return []
    commands: list[list[str]] = []
    for script_name in ("lint", "test"):
        if script_name in scripts:
            commands.append([*resolved_npm, "run", script_name])
    return commands


def resolve_tier1_commands(repo: Path) -> list[list[str]]:
    """Resolve the ordered argv lists the harness will run for tier-1.

    Returns ``[]`` to signal "no objective checks possible; fall back to the
    LLM tier-1 worker". The explicit ``config.delegate_tier1_commands``
    override is returned verbatim (skipping detection) when non-empty.
    """
    if config.delegate_tier1_commands:
        return [list(cmd) for cmd in config.delegate_tier1_commands]
    return [*_detect_python(repo), *_detect_node(repo)]


async def run_objective_tier1(
    repo: Path,
    commands: list[list[str]],
    timeout: float,  # noqa: ASYNC109 - per-command timeout passed to wait_for
) -> tuple[bool, str]:
    """Run each argv in ``commands`` inside the sandbox and return a verdict.

    Returns ``(True, report)`` if every command exits 0, or ``(False, report)``
    on the first non-zero (short-circuit, do not run later commands). pytest
    exit code 5 ("no tests collected") counts as a pass because an empty test
    suite is not a regression. On timeout, the proc is killed and the report
    says so. ``FileNotFoundError`` / ``PermissionError`` from
    ``create_subprocess_exec`` are re-raised as :class:`Tier1ToolUnavailable`
    so the dispatcher falls back to the LLM tier-1 worker instead of
    reporting a false FAIL.
    """
    sections: list[str] = []
    max_chars = config.delegate_max_output_chars
    logger.info(f"objective tier-1: running {len(commands)} command(s) in {repo}")

    for argv in commands:
        try:
            wrapped = sandbox.wrap_in_sandbox(argv, cwd=repo, root=repo)
        except RuntimeError as exc:
            # wrap_in_sandbox raises RuntimeError ("Security Block: ...") when the
            # sandbox tool/profile/platform is unavailable. Fall back to the LLM
            # tier-1 worker (the pre-objective behavior) rather than crash the gate.
            raise Tier1ToolUnavailable(
                f"sandbox unavailable for {argv[0]!r}: {exc}"
            ) from exc
        try:
            proc = await asyncio.create_subprocess_exec(
                *wrapped,
                cwd=repo,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise Tier1ToolUnavailable(
                f"cannot exec {argv[0]!r} in sandbox: {exc}"
            ) from exc

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await asyncio.shield(proc.wait())
            return False, f"tier-1 {' '.join(argv)} timed out after {timeout}s"

        output = stdout.decode("utf-8", errors="replace")
        if len(output) > max_chars:
            output = output[:max_chars] + "\n\n[Output truncated...]"
        sections.append(f"$ {' '.join(argv)}\n{output}")

        is_pytest = "pytest" in argv[0] or any(piece == "pytest" for piece in argv)
        if proc.returncode == 0 or (is_pytest and proc.returncode == 5):
            continue
        return False, (
            f"tier-1 command failed (exit {proc.returncode}): "
            f"{' '.join(argv)}\n{output}"
        )

    return True, "\n\n".join(sections) if sections else "tier-1: no commands"


__all__ = [
    "Tier1ToolUnavailable",
    "resolve_tier1_commands",
    "run_objective_tier1",
]
