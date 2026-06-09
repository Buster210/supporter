"""opencode delegation backend.

Runs a delegated task on the `opencode` CLI as a headless print-and-exit
subprocess. opencode is not inheriting Ritesh's `~/.claude` standards, so the
harness injects a compact standards preamble into the prompt; the native QA gate
is the enforcement. Output is captured as plain text (`--format default`); the
structured-JSON result contract is layered on separately.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import Callable
from typing import Any

from ...config import config
from ...logger import logger
from ...prompts import DELEGATION_RESULT_CONTRACT

__all__ = ["run_opencode"]

OPENCODE_BIN = os.path.expanduser(os.getenv("OPENCODE_BIN", "~/.opencode/bin/opencode"))

_STANDARDS = (
    "You are a coding worker. BINDING STANDARDS: priority correctness > security "
    "> clarity > performance > brevity. Make surgical, minimal changes -- touch "
    "only what the task needs, no drive-by refactors. When changing code that "
    "already works, stay behaviorally lossless. Validate inputs at trust "
    "boundaries; never put secrets in code or logs. Match the surrounding code "
    "conventions; idiomatic to the language; prefer stdlib/maintained deps over "
    "hand-rolling."
)

_CONTRACT = "\n\n---\nMake only the changes the task requires. Stop when done."


def _build_spec(task: dict[str, Any]) -> str:
    parts = [_STANDARDS, f"\n\nTASK:\n{task['task']}"]
    if task.get("context"):
        parts.append(f"\n\nCONTEXT:\n{task['context']}")
    parts.append(_CONTRACT)
    if task.get("result_contract", True):
        parts.append(DELEGATION_RESULT_CONTRACT)
    return "".join(parts)


def _resolve_repo() -> str:
    dirs = config.allowed_directories
    return dirs[0] if dirs else os.getcwd()


def _resolve_binary() -> str | None:
    return OPENCODE_BIN if os.path.exists(OPENCODE_BIN) else shutil.which("opencode")


async def run_opencode(
    task: dict[str, Any],
    on_chunk: Callable[[str], None] | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """Run a task on the opencode CLI, returning (output, model, tokens).

    Raises TimeoutError if the run exceeds the task timeout (the process is
    killed), or RuntimeError if opencode is missing or exits non-zero. Uses
    argv-list subprocess (no shell) so the task spec cannot inject commands.

    If on_chunk is provided, it is called with each chunk of stdout as it arrives.
    """
    binary = _resolve_binary()
    if not binary:
        raise RuntimeError(
            "opencode CLI not found; set OPENCODE_BIN or install opencode"
        )

    repo = _resolve_repo()
    spec = _build_spec(task)
    model = os.getenv("OPENCODE_MODEL", "").strip() or None

    argv = [binary, "run", spec, "--format", "default", "--dir", repo]
    if model:
        argv += ["-m", model]

    logger.info(f"opencode backend: launching task '{task['id']}' in {repo}")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=repo,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout_buffer: list[bytes] = []

    async def _stream_stdout() -> None:
        while True:
            # Read in chunks to get incremental output
            # Use a large read size to avoid splitting lines unnecessarily
            chunk = await proc.stdout.read(8192)  # type: ignore[union-attr]
            if not chunk:
                break
            stdout_buffer.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk.decode("utf-8", errors="replace"))

    try:
        # Honor timeout across the entire read loop
        await asyncio.wait_for(_stream_stdout(), timeout=task["timeout"])
    except TimeoutError, asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await asyncio.shield(proc.wait())
        raise

    # Reap the process to get return code (already exited if we got all stdout)
    try:
        await proc.wait()
    except TimeoutError, asyncio.CancelledError:
        proc.kill()
        with contextlib.suppress(ProcessLookupError):
            await asyncio.shield(proc.wait())
        raise

    output = b"".join(stdout_buffer).decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"opencode exited {proc.returncode}: {output[-500:] or '(no output)'}"
        )
    return output, model, {}
