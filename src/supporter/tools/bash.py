import ast
import asyncio
import fnmatch
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ..config import config
from ..logger import logger
from .bash_defs import (
    FILE_READING_BINS,
    HIGH_RISK_TIER2_BINARIES,
    INSTALL_CMDS,
    INTERPRETERS,
    NETWORK_BINARIES,
    PACKAGE_MANAGERS,
    RISKY_PYTHON_ATTRS,
    RISKY_PYTHON_MODULES,
    RISKY_PYTHON_NAMES,
    RM_NUCLEAR_PATHS,
    SECRET_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    SHELL_BINS,
    SHELL_METACHARACTERS,
    SYSTEM_DIRECTORIES,
    TEMP_DIRS,
    TIER1_GIT_SUBCOMMANDS,
    TIER3_BINARIES,
    TIER3_PYTHON_MODULES,
    TRUSTED_PREFIXES,
    UPLOAD_FLAGS,
)

_BASH_CONFIRMATION_CALLBACK: Callable[[list[str], str], bool] | None = None
_BASH_NOTIFICATION_CALLBACK: Callable[[str], None] | None = None

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _detect_sandbox() -> tuple[str | None, str | None]:
    if sys.platform == "darwin":
        bin_path = shutil.which("sandbox-exec")
        if bin_path:
            return "macos", bin_path
    elif sys.platform.startswith("linux"):
        bin_path = shutil.which("nsjail")
        if bin_path:
            return "linux", bin_path
    return None, None


_SB_TYPE, _SB_BIN = _detect_sandbox()


def _wrap_in_sandbox(tokens: list[str], cwd: Path, project_root: Path) -> list[str]:
    if not _SB_BIN:
        raise RuntimeError(
            "Security Block: Sandbox execution is required but no sandbox tool "
            "(sandbox-exec or nsjail) was found on this system."
        )

    if _SB_TYPE == "macos":
        sb_template_path = Path(__file__).parent / "supporter.sb"
        if not sb_template_path.exists():
            raise RuntimeError(
                f"Security Block: macOS sandbox profile missing: {sb_template_path}"
            )

        with open(sb_template_path) as f:
            profile_content = f.read()

        profile_content = profile_content.replace("{{PROJECT_ROOT}}", str(project_root))
        home_dir = os.environ.get("HOME", str(Path.home()))
        profile_content = profile_content.replace("{{HOME}}", home_dir)

        return [_SB_BIN, "-p", profile_content, *tokens]
    if _SB_TYPE == "linux":
        return [
            _SB_BIN,
            "-Mo",
            "--chroot",
            "/",
            "--cwd",
            str(cwd),
            "--bindmount",
            f"{project_root}:{project_root}",
            "--",
            *tokens,
        ]

    raise RuntimeError(
        f"Security Block: Unsupported sandbox configuration ({_SB_TYPE})"
    )


def set_bash_notification_callback(
    callback: Callable[[str], None] | None,
) -> None:
    global _BASH_NOTIFICATION_CALLBACK
    _BASH_NOTIFICATION_CALLBACK = callback


def _check_find_command(tokens: list[str]) -> bool:
    risky_flags = {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
    return any(token in risky_flags for token in tokens)


def check_bash_availability() -> bool:
    return _SB_BIN is not None


def notify_bash_unavailable() -> None:
    if _BASH_NOTIFICATION_CALLBACK:
        _BASH_NOTIFICATION_CALLBACK(
            "BASH TOOL DISABLED: Sandbox execution is required but no sandbox tool "
            "(sandbox-exec or nsjail) was found on this system."
        )


def set_bash_confirmation_callback(
    callback: Callable[[list[str], str], bool] | None,
) -> None:
    global _BASH_CONFIRMATION_CALLBACK
    _BASH_CONFIRMATION_CALLBACK = callback


def _check_network_egress(binary_name: str, tokens: list[str]) -> int:
    cmd_str = " ".join(tokens)
    if "|" in cmd_str or "<" in cmd_str:
        return 3

    for token in tokens:
        if token in UPLOAD_FLAGS:
            return 3
        if token.startswith(("-d@", "--data@", "-d @", "--data @")):
            return 3
        if token.startswith(("--data-binary @", "--data-urlencode @")):
            return 3
        if "@" in token and binary_name in ["http", "httpie"]:
            return 3

    for i in range(len(tokens) - 1):
        if tokens[i] in {
            "-d",
            "--data",
            "--data-binary",
            "--data-urlencode",
        } and tokens[i + 1].startswith("@"):
            return 3

    return 1


def _check_package_manager(binary_name: str, tokens: list[str]) -> int:
    cmd_str = " ".join(tokens)

    if any(
        x in cmd_str
        for x in [
            "--registry",
            "--index-url",
            "--extra-index-url",
            "git+",
            ".tar.gz",
            ".zip",
            "http://",
            "https://",
        ]
    ):
        return 3

    if "-g" in tokens or "--global" in tokens or "--user" in tokens:
        return 3

    is_install = any(t in INSTALL_CMDS for t in tokens)
    if not is_install:
        return 1

    return 2


def _check_rm_nuclear(binary_name: str, tokens: list[str], cwd: Path) -> None:
    if binary_name != "rm":
        return
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        try:
            p = Path(token).expanduser()
            resolved = (cwd / p).resolve() if not p.is_absolute() else p.resolve()
            if str(resolved) in RM_NUCLEAR_PATHS:
                raise PermissionError(
                    f"Tier 3 BLOCK: rm targeting system-critical path: {resolved}"
                )
        except PermissionError:
            raise
        except Exception as e:
            logger.debug(f"rm check failure: {e}")
            continue


def _gate_inner_shell_payload(inner_tokens: list[str], depth: int) -> int:
    if not inner_tokens:
        return 2
    base_cmd = inner_tokens[0]
    if "/" in base_cmd:
        return 3
    try:
        resolved = _verify_binary(base_cmd)
    except PermissionError:
        return 3
    binary_name = resolved.name
    if binary_name in TIER3_BINARIES:
        return 3
    try:
        _check_execution_location(resolved)
    except PermissionError:
        return 3
    inner_cmd = shlex.join(inner_tokens)
    try:
        _check_complex_syntax(inner_cmd)
    except PermissionError:
        return 3
    project_root = Path(config.allowed_directories[0]).expanduser().resolve()
    try:
        _apply_path_security(inner_cmd, inner_tokens, project_root, project_root)
    except PermissionError:
        return 3
    try:
        tier = _apply_policy_checks(inner_cmd, inner_tokens, binary_name, 1)
        if tier == 3:
            return 3
    except PermissionError:
        return 3
    if binary_name in INTERPRETERS:
        inner_tier = _inspect_interpreter_payload(binary_name, inner_tokens, depth + 1)
        if inner_tier == 3:
            return 3
    return 1


def _inspect_interpreter_payload(
    binary_name: str, tokens: list[str], depth: int = 0
) -> int:
    if depth > 1:
        return 3

    payload = None
    for i, token in enumerate(tokens):
        if token in ["-c", "-e"] and i + 1 < len(tokens):
            payload = tokens[i + 1]
            break

    if not payload:
        return 1

    if len(payload) > 500:
        return 3

    for pattern in SENSITIVE_FILE_PATTERNS:
        if fnmatch.fnmatch(payload, f"*{pattern}*"):
            return 3

    if binary_name in ["python", "python3"]:
        try:
            tree = ast.parse(payload)
            worst_tier = 1
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in RISKY_PYTHON_NAMES
                    ):
                        return 3
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in RISKY_PYTHON_NAMES
                    ):
                        return 3
                    if isinstance(node.func, ast.Name) and node.func.id == "open":
                        write_modes = {"w", "a", "x", "wb", "ab", "xb"}
                        mode_arg = None
                        if len(node.args) >= 2 and isinstance(
                            node.args[1], ast.Constant
                        ):
                            mode_arg = node.args[1].value
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode_arg = kw.value.value
                        if mode_arg and any(m in str(mode_arg) for m in write_modes):
                            worst_tier = max(worst_tier, 2)

                if isinstance(node, ast.Name) and node.id in RISKY_PYTHON_NAMES:
                    return 3

                if isinstance(node, ast.Attribute) and node.attr in RISKY_PYTHON_ATTRS:
                    return 3

                if isinstance(node, ast.Subscript):
                    if isinstance(node.slice, ast.Constant):
                        if str(node.slice.value) in RISKY_PYTHON_NAMES:
                            return 3
                    elif not isinstance(node.slice, ast.Constant):
                        return 3

                if isinstance(node, ast.Call) and isinstance(
                    node.func, (ast.Name, ast.Attribute)
                ):
                    func_name = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else node.func.attr
                    )
                    if func_name in {
                        "__import__",
                        "getattr",
                        "import_module",
                        "exec",
                        "eval",
                    }:
                        for arg in node.args:
                            if isinstance(arg, ast.BinOp) and isinstance(
                                arg.op, ast.Add
                            ):
                                return 3
                            if isinstance(arg, ast.JoinedStr):
                                return 3
                            if isinstance(arg, ast.Call):
                                return 3

                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [n.name for n in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module]
                    )
                    if any(name in TIER3_PYTHON_MODULES for name in names):
                        return 3
                    if any(name in RISKY_PYTHON_MODULES for name in names):
                        worst_tier = max(worst_tier, 2)
            return worst_tier

        except Exception as e:
            logger.debug(f"Python payload parse failure: {e}")
            return 3

    if binary_name in ["node", "js"]:
        risky_regex = (
            r"(require\((?!['\"][\w./\-@]+['\"])"
            r"|import\(|child_process"
            r"|fs\.(?:write|unlink|rm|rename|truncate)"
            r"|process\."
            r"|eval|Function\("
            r"|Buffer\.from\(.*'base64'\)|atob\(|btoa\("
            r"|(?:global|globalThis|process)\s*\[)"
        )
        if re.search(risky_regex, payload):
            return 3
        if "`" in payload and "${" in payload:
            return 3

    if binary_name in ["bash", "sh"]:
        if any(m in payload for m in [";", "&&", "||", "|", ">", "<", "`", "$("]):
            return 3
        try:
            inner_tokens = shlex.split(payload)
            return _gate_inner_shell_payload(inner_tokens, depth)
        except Exception as e:
            logger.debug(f"Bash payload split failure: {e}")
            return 3

    return 1


async def execute_bash(command: str, working_directory: str | None = None) -> str:
    """Executes a shell command in a sandboxed, restricted environment.

    IMPORTANT: Only use safe and simple commands. Unsafe or overly complex
    commands will be blocked by the security gate. Avoid piping, clubbing
    (&&, ||, ;), or command substitution ($(), ``).

    If a command is blocked, do not treat it as a task failure. Instead,
    try decomposing the operation into multiple simpler commands.

    Args:
        command: Shell command to execute (must be safe and simple).
        working_directory: Optional execution cwd (defaults to project root).
    Returns:
        Command stdout/stderr combined, or error message.
        Triggers UI confirmation for risky commands.
    """
    logger.info(f"Tool: execute_bash — command='{command}'")

    def _sync_execute() -> str:
        if "\x00" in command:
            raise ValueError("Null bytes not permitted in commands")
        if not command.isascii():
            raise ValueError("Non-ASCII characters not permitted")

        tokens = _parse_and_strip_env(command)
        if not tokens:
            return "Empty command"

        base_cmd = tokens[0]
        if "/" in base_cmd:
            raise PermissionError(
                f"Tier 3 BLOCK: Command invocation via full path prohibited: {base_cmd}"
            )

        resolved_binary_path = _verify_binary(base_cmd)
        binary_name = resolved_binary_path.name
        if binary_name in TIER3_BINARIES:
            raise PermissionError(f"Tier 3 BLOCK: Binary prohibited: {binary_name}")

        project_root = Path(config.allowed_directories[0]).expanduser().resolve()
        cwd = project_root
        if working_directory:
            from .file_ops import _validate_path

            cwd = _validate_path(working_directory)

        _check_execution_location(resolved_binary_path)

        _check_rm_nuclear(binary_name, tokens, cwd)

        _check_complex_syntax(command)

        security_tier = _apply_path_security(command, tokens, cwd, project_root)

        security_tier = _apply_policy_checks(
            command, tokens, binary_name, security_tier
        )
        logger.info(
            f"Security classification: binary={binary_name}, tier={security_tier}, "
            f"tokens={tokens!r}"
        )

        _evaluate_final_tier(command, tokens, binary_name, security_tier, cwd)

        is_mutation = security_tier >= 2 or any(
            m in command for m in SHELL_METACHARACTERS
        )
        pre_state = _get_fs_state(cwd) if is_mutation else {}

        return _execute_subprocess(
            resolved_binary_path, tokens, cwd, project_root, pre_state, is_mutation
        )

    try:
        return await asyncio.to_thread(_sync_execute)
    except Exception as e:
        logger.error(f"Tool Failure: execute_bash [{type(e).__name__}]: {e}")
        return f"Error: {e!s}"


def _parse_and_strip_env(command: str) -> list[str]:
    clean_command = command.strip()
    while True:
        match = re.match(r"^[A-Z_][A-Z0-9_]*=\S+\s+", clean_command)
        if match:
            clean_command = clean_command[match.end() :].strip()
        else:
            break
    return shlex.split(clean_command)


def _check_complex_syntax(command: str) -> None:
    if "$(" in command or "`" in command:
        raise PermissionError(
            "Tier 3 BLOCK: Command substitution ($() or backticks) prohibited"
        )

    if "|" in command:
        parts = command.split("|")
        for part in parts[1:]:
            sub_tokens = shlex.split(part.strip())
            if sub_tokens:
                try:
                    rhs_bin = _verify_binary(sub_tokens[0])
                    if rhs_bin.name in SHELL_BINS:
                        raise PermissionError(
                            f"Tier 3 BLOCK: Pipe-to-shell detected: {rhs_bin.name}"
                        )
                    if rhs_bin.name in NETWORK_BINARIES:
                        lhs_tokens = shlex.split(parts[0].strip())
                        if any(
                            t in FILE_READING_BINS or t.startswith(("/", ".", "~"))
                            for t in lhs_tokens
                        ):
                            raise PermissionError(
                                "Tier 3 BLOCK: Pipe-to-network "
                                f"(potential exfil): {rhs_bin.name}"
                            )
                except PermissionError as e:
                    if "Binary not found" not in str(e):
                        raise


def _check_execution_location(binary_path: Path) -> None:
    for tdir in TEMP_DIRS:
        if str(binary_path).startswith(tdir):
            raise PermissionError(
                f"Tier 3 BLOCK: Execution from temp directory prohibited: {binary_path}"
            )


def _apply_path_security(
    command: str, tokens: list[str], cwd: Path, project_root: Path
) -> int:
    security_tier = 1
    for token in tokens[1:]:
        check_value = token
        if token.startswith("-"):
            if "=" in token:
                check_value = token.split("=", 1)[1]
            elif "@" in token:
                check_value = token.split("@", 1)[1]
            elif len(token) > 2 and not token.startswith("--"):
                check_value = token[2:]
            else:
                continue

            if not check_value:
                continue

        try:
            p = Path(check_value).expanduser()
            p = (cwd / p).resolve() if not p.is_absolute() else p.resolve()

            basename = p.name
            if any(
                fnmatch.fnmatch(basename, pattern)
                for pattern in SENSITIVE_FILE_PATTERNS
            ):
                raise PermissionError(
                    "Tier 3 BLOCK: Access to sensitive file pattern prohibited: "
                    f"{token} (resolved: {basename})"
                )
        except PermissionError:
            raise
        except Exception as e:
            logger.debug(f"Path security check failure: {e}")
            continue

    for token in tokens:
        if token.startswith(("/", "~", ".")) or "/" in token:
            try:
                p = Path(token).expanduser()
                abs_p = (cwd / p).resolve() if not p.is_absolute() else p.resolve()
                resolved_token = str(abs_p)
            except Exception as e:
                logger.debug(f"Path expansion failure: {e}")
                resolved_token = token
        else:
            resolved_token = token

        for sys_dir in SYSTEM_DIRECTORIES:
            abs_sys_dir = str(Path(sys_dir).expanduser().resolve())
            if resolved_token == abs_sys_dir or resolved_token.startswith(
                abs_sys_dir + "/"
            ):
                if any(m in command for m in SHELL_METACHARACTERS):
                    raise PermissionError(
                        "Tier 3 BLOCK: Metacharacter targeting system directory: "
                        f"{token}"
                    )
                raise PermissionError(
                    f"Tier 3 BLOCK: Access to sensitive directory prohibited: {token}"
                )

        if resolved_token.startswith(("/", "..")):
            try:
                p = Path(resolved_token).resolve()
                if not (project_root in p.parents or p == project_root):
                    security_tier = 2
            except Exception as e:
                logger.debug(f"Project boundary check failure: {e}")

    return security_tier


def _check_open_command(tokens: list[str]) -> int:
    for token in tokens[1:]:
        if token in {"-a", "-e"}:
            return 3
    return 1


def _apply_policy_checks(
    command: str,
    tokens: list[str],
    binary_name: str,
    security_tier: int,
) -> int:
    if binary_name in NETWORK_BINARIES:
        tier = _check_network_egress(binary_name, tokens)
        if tier == 3:
            raise PermissionError(
                "Tier 3 BLOCK: Network egress violation "
                f"(upload/exfil flags): {command}"
            )
        security_tier = max(security_tier, tier)

    if binary_name in PACKAGE_MANAGERS:
        tier = _check_package_manager(binary_name, tokens)
        if tier == 3:
            raise PermissionError(
                f"Tier 3 BLOCK: Package manager supply chain violation: {command}"
            )
        security_tier = max(security_tier, tier)

    if binary_name in INTERPRETERS:
        tier = _inspect_interpreter_payload(binary_name, tokens)
        if tier == 3:
            raise PermissionError(
                f"Tier 3 BLOCK: Risky or obfuscated payload detected: {command}"
            )
        security_tier = max(security_tier, tier)

    if binary_name == "open":
        tier = _check_open_command(tokens)
        if tier == 3:
            raise PermissionError(
                f"Tier 3 BLOCK: 'open' with -a/-e flag is prohibited: {command}"
            )
        security_tier = max(security_tier, tier)

    if binary_name in HIGH_RISK_TIER2_BINARIES:
        security_tier = max(security_tier, 2)

    if binary_name == "find" and _check_find_command(tokens):
        security_tier = max(security_tier, 2)

    return security_tier


def _evaluate_final_tier(
    command: str,
    tokens: list[str],
    binary_name: str,
    security_tier: int,
    cwd: Path,
) -> None:
    if security_tier == 1 and binary_name == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        if sub not in TIER1_GIT_SUBCOMMANDS:
            security_tier = 2

    if security_tier >= 2:
        if _BASH_CONFIRMATION_CALLBACK:
            if not _BASH_CONFIRMATION_CALLBACK(tokens, str(cwd)):
                raise PermissionError("Execution cancelled by user.")
        else:
            raise PermissionError(f"Tier 2 Confirmation Required: {command}")


def _get_fs_state(target_dir: Path) -> dict[str, float]:
    from ..config import INTERNAL_BLACKLIST as FILE_INTERNAL_BLACKLIST

    state = {}
    try:
        for entry in os.scandir(target_dir):
            if entry.is_file():
                if entry.name in FILE_INTERNAL_BLACKLIST:
                    continue
                try:
                    state[str(Path(entry.path))] = entry.stat().st_mtime
                except OSError:
                    continue
    except Exception as e:
        logger.debug(f"FS state scan failure: {e}")
    return state


def _execute_subprocess(
    binary_path: Path,
    tokens: list[str],
    cwd: Path,
    project_root: Path,
    pre_state: dict[str, float],
    is_mutation: bool = True,
) -> str:
    validated_tokens = [str(binary_path), *tokens[1:]]
    final_tokens = _wrap_in_sandbox(validated_tokens, cwd, project_root)

    def _set_limits() -> None:
        os.setsid()
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            resource.setrlimit(
                resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            )
        except Exception as e:
            logger.debug(f"setrlimit failure: {e}")

    try:
        result = subprocess.run(  # nosec B603 # noqa: S603
            final_tokens,
            shell=False,
            cwd=cwd,
            env={
                "PATH": ":".join(TRUSTED_PREFIXES),
                "TERM": "dumb",
                "LANG": "en_US.UTF-8",
                "HOME": os.environ.get("HOME", ""),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            close_fds=True,
            preexec_fn=_set_limits if sys.platform != "win32" else None,
        )

        stdout = result.stdout[: 100 * 1024].decode("utf-8", errors="replace")
        stderr = result.stderr[: 100 * 1024].decode("utf-8", errors="replace")

        logger.info(
            f"Subprocess exit: binary={binary_path.name}, rc={result.returncode}, "
            f"stdout_bytes={len(result.stdout)}, stderr_bytes={len(result.stderr)}, "
            f"sandbox={_SB_TYPE!r}"
        )

        stdout = _ANSI_ESCAPE.sub("", stdout)
        stderr = _ANSI_ESCAPE.sub("", stderr)

        for pattern in SECRET_PATTERNS:
            stdout = re.sub(pattern, "[REDACTED]", stdout)
            stderr = re.sub(pattern, "[REDACTED]", stderr)

        output = stdout + stderr

        if is_mutation:
            post_state = _get_fs_state(cwd)
            all_paths = set(pre_state.keys()) | set(post_state.keys())
            changed = [f for f in all_paths if pre_state.get(f) != post_state.get(f)]
            if changed:
                basenames = [Path(f).name for f in changed]
                logger.info(f"FS mutation detected by bash in {cwd}: {basenames}")
                output += f"\n\n[WARNING] Files mutated in cwd: {', '.join(basenames)}"

        logger.debug(f"bash full output: {output!r}")
        changed_count = len(changed) if is_mutation else 0
        logger.info(
            f"bash: rc={result.returncode}, out_len={len(output)}, "
            f"mutated_files={changed_count}"
        )

        return output

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (30s limit)"
    except Exception as e:
        error_msg = str(e)
        if "Security Block" in error_msg and _BASH_NOTIFICATION_CALLBACK:
            _BASH_NOTIFICATION_CALLBACK(f"BASH TOOL FAILED: {error_msg}")
        logger.error(f"Execution failure: {e}")
        return f"Error executing command: {e!s}"


def _verify_binary(cmd: str) -> Path:
    binary = shutil.which(cmd)
    if not binary:
        raise PermissionError(f"Binary not found: {cmd}")
    return Path(binary).resolve()
