import ast
import fnmatch
import re
import shlex
import shutil
from functools import lru_cache
from pathlib import Path

from .. import resolved_project_root
from . import sandbox
from .defs import (
    AUTO_APPROVED_BINARIES,
    AUTO_APPROVED_GIT_SUBCOMMANDS,
    BLOCKED_BINARIES,
    BLOCKED_PYTHON_MODULES,
    CONFIRMATION_REQUIRED_BINARIES,
    CONFIRMATION_REQUIRED_PYTHON_MODULES,
    FILE_READING_BINARIES,
    INLINE_PAYLOAD_INTERPRETER_BINARIES,
    NETWORK_DATA_PREFIXES,
    NETWORK_EGRESS_BINARIES,
    NETWORK_UPLOAD_FLAGS,
    PACKAGE_INSTALL_SUBCOMMANDS,
    PACKAGE_MANAGER_BINARIES,
    RISKY_PYTHON_ATTRIBUTE_NAMES,
    RISKY_PYTHON_SYMBOL_NAMES,
    RM_BLOCKED_TARGET_PATHS,
    SENSITIVE_FILE_PATTERNS,
    SENSITIVE_SYSTEM_PATHS,
    SHELL_INTERPRETER_BINARIES,
    TIER_BLOCK,
    TIER_CONFIRM,
    TIER_SAFE,
    UNTRUSTED_EXECUTION_TEMP_DIRS,
)

_RESOLVED_SENSITIVE_SYSTEM_PATHS: tuple[str, ...] = tuple(
    str(Path(d).expanduser().resolve()) for d in SENSITIVE_SYSTEM_PATHS
)

_GLOBAL_FLAGS: frozenset[str] = frozenset({"-g", "--global", "--user"})

_RISKY_NODE_RE = re.compile(
    r"(require\((?!['\"][\w./\-@]+['\"])|import\(|child_process|"
    r"fs\.(?:write|unlink|rm|rename|truncate)|process\.|eval|Function\(|"
    r"Buffer\.from\(.*'base64'\)|atob\(|btoa\(|"
    r"(?:global|globalThis|process)\s*\[)"
)


@lru_cache(maxsize=128)
def verify_binary(command_name: str) -> Path:
    binary_path = shutil.which(command_name)
    if not binary_path:
        raise PermissionError(f"Binary not found: {command_name}")
    return Path(binary_path).resolve()


def _check_find_command(tokens: list[str]) -> bool:
    risky_flags = {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
    return any(token in risky_flags for token in tokens)


def _find_exec_payload_tier(tokens: list[str]) -> int:
    for i, token in enumerate(tokens):
        if token in {"-exec", "-execdir", "-ok", "-okdir"} and i + 1 < len(tokens):
            payload_token = tokens[i + 1]
            payload_tokens = shlex.split(payload_token)
            if payload_tokens:
                binary_name = payload_tokens[0]
                try:
                    binary_path = verify_binary(binary_name)
                    if binary_path.name in BLOCKED_BINARIES:
                        return TIER_BLOCK
                    inner_command = shlex.join(payload_tokens)
                    tier = apply_policy_checks(
                        inner_command, payload_tokens, binary_name, TIER_SAFE
                    )
                    if tier >= TIER_BLOCK:
                        return TIER_BLOCK
                    if tier >= TIER_CONFIRM:
                        return TIER_CONFIRM
                except PermissionError:
                    return TIER_BLOCK
    return TIER_CONFIRM


def _check_network_egress(binary_name: str, tokens: list[str]) -> int:
    command_string = " ".join(tokens)
    if "|" in command_string or "<" in command_string:
        return TIER_BLOCK

    for token in tokens:
        if token in NETWORK_UPLOAD_FLAGS:
            return TIER_BLOCK
        if any(token.startswith(prefix) for prefix in NETWORK_DATA_PREFIXES):
            return TIER_BLOCK
        if "@" in token and binary_name in ["http", "httpie"]:
            return TIER_BLOCK

    for i in range(len(tokens) - 1):
        if tokens[i] in {
            "-d",
            "--data",
            "--data-binary",
            "--data-urlencode",
        } and tokens[i + 1].startswith("@"):
            return TIER_BLOCK

    return TIER_SAFE


def _check_package_manager(binary_name: str, tokens: list[str]) -> int:
    command_string = " ".join(tokens)
    if any(
        x in command_string
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
        return TIER_BLOCK

    if any(t in _GLOBAL_FLAGS for t in tokens):
        return TIER_BLOCK

    is_install = any(token in PACKAGE_INSTALL_SUBCOMMANDS for token in tokens)
    if is_install:
        if (
            "--only-binary=:all:" in tokens
            or "--ignore-scripts" in tokens
            or "--no-sync-scripts" in tokens
            or "--no-deps" in tokens
        ):
            return TIER_CONFIRM
        if binary_name in ["yarn", "bun", "go", "cargo", "poetry", "gem"]:
            return TIER_CONFIRM
        return TIER_CONFIRM

    return TIER_SAFE


def check_complex_syntax(command: str) -> None:
    if "$(" in command or "`" in command:
        raise PermissionError("Tier 3 BLOCK: Command substitution prohibited")

    lexer = shlex.shlex(command, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    all_tokens = list(lexer)
    if "|" not in all_tokens:
        return

    stages: list[list[str]] = [[]]
    for tok in all_tokens:
        if tok == "|":
            stages.append([])
        else:
            stages[-1].append(tok)

    for stage in stages[1:]:
        if not stage:
            continue
        try:
            rhs_binary = verify_binary(stage[0])
        except PermissionError:
            raise
        except Exception:  # noqa: S112 # nosec B112
            continue
        if rhs_binary.name in SHELL_INTERPRETER_BINARIES:
            raise PermissionError(
                f"Tier 3 BLOCK: Pipe-to-shell detected: {rhs_binary.name}"
            )
        if rhs_binary.name in NETWORK_EGRESS_BINARIES:
            lhs_tokens = stages[0]
            if any(
                t in FILE_READING_BINARIES or t.startswith(("/", ".", "~"))
                for t in lhs_tokens
            ):
                raise PermissionError(
                    "Tier 3 BLOCK: Pipe-to-network (potential exfil): "
                    f"{rhs_binary.name}"
                )


def check_execution_location(binary_path: Path) -> None:
    path_str = str(binary_path)
    for temp_dir in UNTRUSTED_EXECUTION_TEMP_DIRS:
        if path_str.startswith(temp_dir):
            raise PermissionError(
                f"Tier 3 BLOCK: Execution from temp directory prohibited: {path_str}"
            )


def check_rm_nuclear(binary_name: str, tokens: list[str], cwd: Path) -> None:
    if binary_name != "rm":
        return
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        try:
            path = Path(token).expanduser()
            target = (
                (cwd / path).resolve() if not path.is_absolute() else path.resolve()
            )
            if str(target) in RM_BLOCKED_TARGET_PATHS:
                raise PermissionError(
                    f"Tier 3 BLOCK: rm targeting system-critical path: {target}"
                )
        except PermissionError:
            raise
        except Exception:  # noqa: S112 # nosec B112
            continue


def _gate_inner_shell_payload(inner_tokens: list[str], depth: int) -> int:
    if not inner_tokens:
        return TIER_CONFIRM
    base_cmd = inner_tokens[0]
    if "/" in base_cmd:
        return TIER_BLOCK
    try:
        binary_path = verify_binary(base_cmd)
        binary_name = binary_path.name
        if binary_name in BLOCKED_BINARIES:
            return TIER_BLOCK
        check_execution_location(binary_path)
        inner_command = shlex.join(inner_tokens)
        check_complex_syntax(inner_command)
        project_root = resolved_project_root()
        apply_path_security(inner_command, inner_tokens, project_root, project_root)
        tier = apply_policy_checks(inner_command, inner_tokens, binary_name, TIER_SAFE)
        if tier == TIER_BLOCK:
            return TIER_BLOCK
        if binary_name in INLINE_PAYLOAD_INTERPRETER_BINARIES:
            return _inspect_interpreter_payload(binary_name, inner_tokens, depth + 1)
        return tier
    except Exception:
        return TIER_BLOCK


def _check_open_command(tokens: list[str]) -> int:
    if any(token in tokens[1:] for token in {"-a", "-e"}):
        return TIER_BLOCK
    return TIER_SAFE


def _inspect_python(payload: str) -> int:
    try:
        tree = ast.parse(payload)
        worst_tier = TIER_SAFE
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_names = (
                    [n.name for n in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                )
                if any(name in BLOCKED_PYTHON_MODULES for name in module_names):
                    return TIER_BLOCK
                if any(
                    name in CONFIRMATION_REQUIRED_PYTHON_MODULES
                    for name in module_names
                ):
                    worst_tier = max(worst_tier, TIER_CONFIRM)

            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in RISKY_PYTHON_SYMBOL_NAMES:
                    return TIER_BLOCK

                if func_name in {
                    "__import__",
                    "getattr",
                    "import_module",
                    "exec",
                    "eval",
                }:
                    for arg in node.args:
                        if isinstance(arg, (ast.BinOp, ast.JoinedStr, ast.Call)):
                            return TIER_BLOCK

                if func_name == "open":
                    write_modes = {"w", "a", "x", "wb", "ab", "xb"}
                    mode_arg = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode_arg = node.args[1].value
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode_arg = kw.value.value
                    if mode_arg and any(m in str(mode_arg) for m in write_modes):
                        worst_tier = max(worst_tier, TIER_CONFIRM)

            if isinstance(node, ast.Name) and node.id in RISKY_PYTHON_SYMBOL_NAMES:
                return TIER_BLOCK
            if (
                isinstance(node, ast.Attribute)
                and node.attr in RISKY_PYTHON_ATTRIBUTE_NAMES
            ):
                return TIER_BLOCK
            if isinstance(node, ast.Subscript) and (
                not isinstance(node.slice, ast.Constant)
                or str(node.slice.value) in RISKY_PYTHON_SYMBOL_NAMES
            ):
                return TIER_BLOCK

        return worst_tier
    except Exception:
        return TIER_BLOCK


def _inspect_node(payload: str) -> int:
    if _RISKY_NODE_RE.search(payload) or ("`" in payload and "${" in payload):
        return TIER_BLOCK
    return TIER_SAFE


def _inspect_shell(payload: str, depth: int) -> int:
    if any(m in payload for m in [";", "&&", "||", "|", ">", "<", "`", "$("]):
        return TIER_BLOCK
    try:
        return _gate_inner_shell_payload(shlex.split(payload), depth)
    except Exception:
        return TIER_BLOCK


def _inspect_interpreter_payload(
    binary_name: str, tokens: list[str], depth: int = 0
) -> int:
    if depth > 1:
        return TIER_BLOCK

    payload = ""
    for i, token in enumerate(tokens):
        if token in ["-c", "-e"] and i + 1 < len(tokens):
            payload = tokens[i + 1]
            break

    if not payload:
        return TIER_SAFE

    if len(payload) > 500:
        return TIER_BLOCK

    if any(fnmatch.fnmatch(payload, f"*{p}*") for p in SENSITIVE_FILE_PATTERNS):
        return TIER_BLOCK

    if binary_name in {"python", "python3"}:
        return _inspect_python(payload)

    if binary_name in ["node", "js"]:
        return _inspect_node(payload)

    if binary_name in ["bash", "sh"]:
        return _inspect_shell(payload, depth)

    return TIER_SAFE


def apply_path_security(
    command: str, tokens: list[str], cwd: Path, project_root: Path
) -> int:
    security_tier = TIER_SAFE
    for token in tokens:
        check_value = _extract_path_candidate(token)
        target: Path | None = None
        if check_value:
            try:
                path = Path(check_value).expanduser()
                target = (
                    path.resolve() if path.is_absolute() else (cwd / path).resolve()
                )
                if any(
                    fnmatch.fnmatch(target.name, p) for p in SENSITIVE_FILE_PATTERNS
                ):
                    raise PermissionError(
                        f"Tier 3 BLOCK: sensitive file pattern: {target.name}"
                    )
            except PermissionError:
                raise
            except Exception:  # nosec B110
                target = None

        path_string = (
            str(target) if target is not None else _resolve_path_string(token, cwd)
        )
        for abs_system_dir in _RESOLVED_SENSITIVE_SYSTEM_PATHS:
            if path_string == abs_system_dir or path_string.startswith(
                abs_system_dir + "/"
            ):
                raise PermissionError(
                    f"Tier 3 BLOCK: sensitive directory / system directory: {token}"
                )

        if path_string.startswith(("/", "..")):
            resolved = target
            if resolved is None:
                try:
                    resolved = Path(path_string).resolve()
                except Exception:  # nosec B110
                    resolved = None
            if resolved is not None and not (
                project_root in resolved.parents or resolved == project_root
            ):
                security_tier = TIER_CONFIRM

    return security_tier


def _extract_path_candidate(token: str) -> str | None:
    if any(token.startswith(flag) for flag in ["-d=", "--file=", "@"]):
        return token.split("=", 1)[1] if "=" in token else token.split("@", 1)[1]
    if len(token) > 2 and token.startswith("-") and not token.startswith("--"):
        return token[2:]
    if token.startswith("-"):
        return None
    return token


def _resolve_path_string(token: str, cwd: Path) -> str:
    try:
        path = Path(token).expanduser()
        abs_path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        return str(abs_path)
    except Exception as exc:
        from ...logger import logger

        logger.warning(
            f"_resolve_path_string failed for token={token!r}: "
            f"{type(exc).__name__}: {exc} — failing closed"
        )
        return "\x00"


def apply_policy_checks(
    command: str, tokens: list[str], binary_name: str, security_tier: int
) -> int:
    if binary_name in NETWORK_EGRESS_BINARIES:
        result = _check_network_egress(binary_name, tokens)
        if result == TIER_BLOCK:
            raise PermissionError(f"Tier 3 BLOCK: Network egress violation: {command}")
        security_tier = max(security_tier, result)

    if binary_name in PACKAGE_MANAGER_BINARIES:
        result = _check_package_manager(binary_name, tokens)
        if result == TIER_BLOCK:
            raise PermissionError(
                f"Tier 3 BLOCK: Package manager supply chain violation: {command}"
            )
        security_tier = max(security_tier, result)

    if binary_name in INLINE_PAYLOAD_INTERPRETER_BINARIES:
        result = _inspect_interpreter_payload(binary_name, tokens)
        if result == TIER_BLOCK:
            raise PermissionError(
                f"Tier 3 BLOCK: Risky or obfuscated payload: {command}"
            )
        security_tier = max(security_tier, result)

    if binary_name == "open":
        result = _check_open_command(tokens)
        if result == TIER_BLOCK:
            raise PermissionError(f"Tier 3 BLOCK: 'open' with -a/-e flag: {command}")
        security_tier = max(security_tier, result)

    if binary_name in CONFIRMATION_REQUIRED_BINARIES:
        security_tier = max(security_tier, TIER_CONFIRM)

    if binary_name == "find" and _check_find_command(tokens):
        exec_tier = _find_exec_payload_tier(tokens)
        security_tier = max(security_tier, exec_tier)

    return security_tier


def apply_tier1_allowlist(
    tokens: list[str], binary_name: str, security_tier: int
) -> int:
    if security_tier >= TIER_CONFIRM:
        return security_tier
    if binary_name in AUTO_APPROVED_BINARIES:
        return TIER_SAFE
    if binary_name == "git":
        subcommand = tokens[1] if len(tokens) > 1 else ""
        if subcommand in AUTO_APPROVED_GIT_SUBCOMMANDS:
            return TIER_SAFE
    return TIER_CONFIRM


def evaluate_final_tier(
    command: str, tokens: list[str], binary_name: str, security_tier: int, cwd: Path
) -> None:
    if security_tier == TIER_SAFE and binary_name == "git":
        subcommand = tokens[1] if len(tokens) > 1 else ""
        if subcommand not in AUTO_APPROVED_GIT_SUBCOMMANDS:
            security_tier = TIER_CONFIRM

    if security_tier >= TIER_CONFIRM:
        if sandbox.bash_confirmation_callback:
            if not sandbox.bash_confirmation_callback(tokens, str(cwd)):
                raise PermissionError("Execution cancelled by user.")
        else:
            raise PermissionError(f"Tier 2 Confirmation Required: {command}")
