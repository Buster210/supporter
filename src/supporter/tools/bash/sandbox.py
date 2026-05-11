import os
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

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
_PROFILE_CACHE: tuple[float, str] | None = None


def _load_profile_template(profile_path: Path) -> str:
    global _PROFILE_CACHE
    if not profile_path.exists():
        raise RuntimeError(
            f"Security Block: macOS sandbox profile missing: {profile_path}"
        )
    mtime = profile_path.stat().st_mtime
    if _PROFILE_CACHE is not None and _PROFILE_CACHE[0] == mtime:
        return _PROFILE_CACHE[1]
    with open(profile_path) as f:
        content = f.read()
    _PROFILE_CACHE = (mtime, content)
    return content


def _wrap_in_sandbox(tokens: list[str], cwd: Path, root: Path) -> list[str]:
    if not _SB_BIN:
        raise RuntimeError("Security Block: Sandbox tool not found")

    if _SB_TYPE == "macos":
        profile_path = Path(__file__).parent / "supporter.sb"
        content = _load_profile_template(profile_path)
        content = content.replace("{{PROJECT_ROOT}}", str(root))
        content = content.replace("{{HOME}}", os.environ.get("HOME", str(Path.home())))

        return [_SB_BIN, "-p", content, *tokens]

    if _SB_TYPE == "linux":
        return [
            _SB_BIN,
            "-Mo",
            "--chroot",
            "/",
            "--cwd",
            str(cwd),
            "--bindmount",
            f"{root}:{root}",
            "--",
            *tokens,
        ]

    raise RuntimeError(
        f"Security Block: Unsupported sandbox configuration ({_SB_TYPE})"
    )


def set_bash_notification_callback(callback: Callable[[str], None] | None) -> None:
    """Sets the callback function for security-related notifications."""
    global _BASH_NOTIFICATION_CALLBACK
    _BASH_NOTIFICATION_CALLBACK = callback


def check_bash_availability() -> bool:
    """Checks if a supported sandbox tool is available on the system."""
    return _SB_BIN is not None


def notify_bash_unavailable() -> None:
    """Triggers a notification if the bash tool is disabled due to missing sandbox."""
    if _BASH_NOTIFICATION_CALLBACK:
        _BASH_NOTIFICATION_CALLBACK("BASH TOOL DISABLED: Sandbox tool not found")


def set_bash_confirmation_callback(
    callback: Callable[[list[str], str], bool] | None,
) -> None:
    """Sets the callback function for user confirmation of risky commands."""
    global _BASH_CONFIRMATION_CALLBACK
    _BASH_CONFIRMATION_CALLBACK = callback
