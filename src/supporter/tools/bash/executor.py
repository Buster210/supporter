import asyncio
import contextlib
import os
import re
import resource
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from ...logger import logger
from .. import resolved_project_root
from ..base import ToolError
from . import policy, sandbox
from .defs import (
    BLOCKED_BINARIES,
    CPU_LIMIT_SEC,
    EXECUTION_TIMEOUT_SEC,
    MEM_LIMIT_BYTES,
    MUTATING_BINARIES,
    OUTPUT_BUFFER_LIMIT,
    SECRET_KEYWORD_PATTERN,
    SECRET_KEYWORD_TRIGGERS,
    SECRET_LITERAL_PATTERNS,
    TIER_CONFIRM,
    TRUSTED_EXECUTABLE_PATH_PREFIXES,
    WRITE_REDIRECT_TOKENS,
)

_LITERAL_SECRET_PATTERNS = tuple(re.compile(p) for p in SECRET_LITERAL_PATTERNS)
_KEYWORD_SECRET_PATTERN = re.compile(SECRET_KEYWORD_PATTERN)


def _set_limits() -> None:
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SEC, CPU_LIMIT_SEC))
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))


def _redact_secrets(text: str) -> str:
    if not text:
        return text
    for pat in _LITERAL_SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    text_lower = text.lower()
    if any(k in text_lower for k in SECRET_KEYWORD_TRIGGERS):
        text = _KEYWORD_SECRET_PATTERN.sub("[REDACTED]", text)
    return text


async def execute_bash(command: str, working_directory: str | None = None) -> str:
    """Run shell command in sandbox. Avoid pipes, &&, ||, ;, $(), ``.

    Args:
        command: Safe, simple shell command.
        working_directory: Optional cwd (default: project root).

    Returns:
        stdout/stderr or error. Risky commands trigger UI confirmation.
    """
    logger.info(f"Tool: execute_bash — command='{command}'")

    def _sync_execute() -> str:
        if "\x00" in command:
            raise PermissionError(
                "Tier 3 BLOCK: Command contains null bytes. "
                "Remove null bytes and try again."
            )
        if not command.isascii():
            raise PermissionError(
                "Tier 3 BLOCK: Command contains non-ASCII characters. Use ASCII-only."
            )

        tokens = _parse_and_strip_env(command)
        if not tokens:
            return "Empty command"

        binary_name = tokens[0]
        if "/" in binary_name:
            raise PermissionError(
                "Tier 3 BLOCK: Command uses absolute path. "
                "Use command name only, e.g., 'ls' not '/bin/ls'."
            )

        binary_path = policy.verify_binary(binary_name)
        if binary_path.name in BLOCKED_BINARIES:
            raise PermissionError(
                f"Tier 3 BLOCK: Binary '{binary_path.name}' is not allowed "
                "by security policy."
            )

        root = resolved_project_root()
        cwd = root
        if working_directory:
            from ..file_ops import validate_path

            cwd = validate_path(working_directory)

        policy.check_execution_location(binary_path)
        policy.check_rm_nuclear(binary_path.name, tokens, cwd)
        policy.check_complex_syntax(command)

        tier = policy.apply_path_security(command, tokens, cwd, root)
        tier = policy.apply_policy_checks(command, tokens, binary_path.name, tier)
        tier = policy.apply_tier1_allowlist(tokens, binary_path.name, tier)

        logger.info(
            f"Security: binary={binary_path.name}, tier={tier}, tokens={tokens!r}"
        )
        policy.evaluate_final_tier(command, tokens, binary_path.name, tier, cwd)

        sed_in_place = binary_path.name == "sed" and any(
            tok.startswith("-i") for tok in tokens
        )
        is_mutation = (
            tier >= TIER_CONFIRM
            or binary_path.name in MUTATING_BINARIES
            or sed_in_place
            or any(tok in WRITE_REDIRECT_TOKENS for tok in tokens)
        )
        pre_names = _get_fs_names(cwd) if is_mutation else set()
        start_time = time.time() if is_mutation else 0.0

        return _execute_subprocess(
            binary_path, tokens, cwd, root, pre_names, start_time, is_mutation
        )

    try:
        return await asyncio.to_thread(_sync_execute)
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        raise ToolError(f"Bash execution failed: {e}") from e


def _parse_and_strip_env(command: str) -> list[str]:
    clean = command.strip()
    while match := re.match(r"^[A-Z_][A-Z0-9_]*=\S+\s+", clean):
        clean = clean[match.end() :].strip()
    return shlex.split(clean)


def _get_fs_names(target: Path) -> set[str]:
    from ...config import INTERNAL_BLACKLIST

    try:
        return {
            e.name
            for e in os.scandir(target)
            if e.is_file() and e.name not in INTERNAL_BLACKLIST
        }
    except Exception:
        return set()


def _get_fs_state(target: Path) -> dict[str, float]:
    from ...config import INTERNAL_BLACKLIST

    try:
        return {
            e.name: e.stat().st_mtime
            for e in os.scandir(target)
            if e.is_file() and e.name not in INTERNAL_BLACKLIST
        }
    except Exception:
        return {}


def _execute_subprocess(
    binary: Path,
    tokens: list[str],
    cwd: Path,
    root: Path,
    pre_names: set[str],
    start_time: float,
    is_mutation: bool = True,
) -> str:
    args = [str(binary), *tokens[1:]]
    cmd_tokens = sandbox.wrap_in_sandbox(args, cwd, root)

    env = {
        "PATH": ":".join(TRUSTED_EXECUTABLE_PATH_PREFIXES),
        "TERM": "dumb",
        "LANG": "en_US.UTF-8",
    }

    try:
        proc = subprocess.Popen(  # nosec B603 # noqa: S603
            cmd_tokens,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=sys.platform != "win32",
            preexec_fn=_set_limits if sys.platform != "win32" else None,
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=EXECUTION_TIMEOUT_SEC)
            returncode = proc.returncode  # noqa: F841
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            try:
                raw_out, raw_err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                raw_out, raw_err = proc.communicate()
            return f"Error: Command timed out ({EXECUTION_TIMEOUT_SEC}s limit)"

        out = raw_out[:OUTPUT_BUFFER_LIMIT].decode("utf-8", errors="replace")
        err = raw_err[:OUTPUT_BUFFER_LIMIT].decode("utf-8", errors="replace")

        combined = sandbox.ANSI_ESCAPE.sub("", out + err)
        output = _redact_secrets(combined)

        if is_mutation:
            post_state = _get_fs_state(cwd)
            post_names = set(post_state)
            added = post_names - pre_names
            deleted = pre_names - post_names
            modified = {
                n for n, m in post_state.items() if n in pre_names and m >= start_time
            }
            changed = added | deleted | modified
            if changed:
                names = ", ".join(sorted(changed))
                output += f"\n\n[WARNING] Files mutated in cwd: {names}"

        return output

    except Exception as e:
        if "Security Block" in str(e) and sandbox.bash_notification_callback:
            sandbox.bash_notification_callback(f"BASH TOOL FAILED: {e!s}")
        return f"Error executing command: {e!s}"
