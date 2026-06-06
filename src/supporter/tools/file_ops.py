from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathspec

from ..config import INTERNAL_BLACKLIST, config
from ..logger import logger
from . import resolved_project_root
from .base import ToolError

_CONFIRMATION_CALLBACK: Callable[[Path, str], bool] | None = None
_GITIGNORE_CACHE: dict[str, Any] = {"spec": None, "mtime": 0}


def set_confirmation_callback(callback: Callable[[Path, str], bool] | None) -> None:
    global _CONFIRMATION_CALLBACK
    _CONFIRMATION_CALLBACK = callback


def register_confirmation_callback(
    callback: Callable[[Path, str], bool] | None,
) -> None:
    set_confirmation_callback(callback)


def emit_confirmation_line(message: str = "") -> None:
    try:
        from textual._context import active_app

        app = active_app.get()
    except LookupError, RuntimeError:
        print(message)
        return

    app.log(message)


def _get_gitignore_spec(project_root: Path) -> pathspec.PathSpec[Any] | None:
    import pathspec as _pathspec

    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return None

    try:
        mtime = gitignore_path.stat().st_mtime
        if _GITIGNORE_CACHE["spec"] is not None and _GITIGNORE_CACHE["mtime"] == mtime:
            from typing import cast

            return cast("pathspec.PathSpec[Any]", _GITIGNORE_CACHE["spec"])

        with gitignore_path.open("r") as f:
            spec = _pathspec.PathSpec.from_lines("gitignore", f)
            _GITIGNORE_CACHE["spec"] = spec
            _GITIGNORE_CACHE["mtime"] = mtime
            return spec
    except Exception as e:
        logger.warning(f"Could not process .gitignore: {e}")
        return None


def _is_blacklisted(relative_path: str) -> bool:
    return any(
        relative_path == pattern or relative_path.startswith(f"{pattern}/")
        for pattern in INTERNAL_BLACKLIST
    )


def validate_path(path: str) -> Path:
    try:
        project_root = resolved_project_root()
    except PermissionError as e:
        raise ToolError(str(e)) from e
    target_path = Path(path).expanduser()

    target_path = (
        (project_root / target_path).resolve()
        if not target_path.is_absolute()
        else target_path.resolve()
    )

    if not (target_path == project_root or project_root in target_path.parents):
        raise PermissionError(
            f"Path '{target_path}' is outside project root '{project_root}'. "
            "Only files within the project directory are allowed."
        )

    relative_path = str(target_path.relative_to(project_root))
    if relative_path == config.log_file:
        return target_path

    if _is_blacklisted(relative_path):
        raise PermissionError(
            f"File '{relative_path}' is protected and cannot be accessed."
        )

    spec = _get_gitignore_spec(project_root)
    if spec and spec.match_file(relative_path):
        raise PermissionError(
            f"File '{relative_path}' is ignored by .gitignore and cannot be accessed."
        )

    return target_path


async def read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """Reads file content safely within project boundaries.
    Args:
        path: File path (abs/rel).
        offset: Start character offset.
        limit: Max characters to read.
        encoding: Text encoding (default: 'utf-8').
    Returns:
        File content as string, or error message.
    """
    logger.info(f"Tool: read_file — path='{path}', offset={offset}, limit={limit}")

    def _sync_read() -> str:
        p = validate_path(path)
        if not p.exists():
            return f"Error: File not found: {p}"

        try:
            with p.open("r", encoding=encoding) as f:
                if offset and offset > 0:
                    f.read(offset)  # text-mode: advance by characters, not bytes
                content = f.read(limit) if limit is not None else f.read()
                logger.debug(f"read_file content: {content!r}")
                return content
        except PermissionError:
            raise
        except Exception as e:
            raise ToolError(f"Could not read '{path}': {e}") from e

    try:
        return await asyncio.to_thread(_sync_read)
    except PermissionError as e:
        raise ToolError(str(e)) from e
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Could not read '{path}': {e}") from e


async def write_file(
    path: str,
    content: str,
    offset: int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """Writes/updates file content securely. Creates parents if missing.
    Args:
        path: Target file path.
        content: Data to write.
        offset: Byte offset for partial writes.
        limit: Bytes to replace (retains tail).
        encoding: Text encoding (default: 'utf-8').
    Returns:
        Success/error message. Triggers UI confirmation if required.
    """
    logger.info(
        f"Tool: write_file — path='{path}', content_len={len(content)}, "
        f"offset={offset}, limit={limit}"
    )

    def _confirm_write(p: Path, content: str, encoding: str) -> str | None:
        if not config.require_write_confirmation:
            return None

        import difflib

        if p.exists():
            try:
                with p.open("r", encoding=encoding) as f:
                    old_content = f.read()
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{p.name}",
                    tofile=f"b/{p.name}",
                    n=3,
                )
                display_diff = "".join(diff) or "(No changes detected)"
            except Exception as e:
                display_diff = (
                    f"Error generating diff: {e}\n\nProposed Content:\n{content}"
                )
        else:
            lines = content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                [], lines, fromfile="/dev/null", tofile=f"b/{p.name}", n=len(lines)
            )
            display_diff = "".join(diff)

        if _CONFIRMATION_CALLBACK:
            if not _CONFIRMATION_CALLBACK(p, display_diff):
                return "Write operation cancelled by user security preference."
        elif sys.stdin.isatty():
            emit_confirmation_line()
            emit_confirmation_line("=" * 60)
            emit_confirmation_line(" SECURITY CONFIRMATION REQUIRED ".center(60, "="))
            emit_confirmation_line(f" TARGET FILE: {p}")
            emit_confirmation_line("-" * 60)
            emit_confirmation_line(" PROPOSED DIFF ".center(60, "-"))
            for line in display_diff.splitlines():
                emit_confirmation_line(line)
            emit_confirmation_line("-" * 60)
            confirm = input(" Proceed with write? (y/n): ").lower().strip()
            emit_confirmation_line("=" * 60)
            emit_confirmation_line()
            if confirm not in ("y", "yes"):
                return "Write operation cancelled by user security preference."
        else:
            logger.warning(
                f"Write confirmation required for {p} but no TTY or callback. Skipping."
            )
            return "Error: Interactive confirmation required but unavailable."
        return None

    def _sync_write() -> str:
        p = validate_path(path)
        logger.debug(f"write_file input: {content!r}")

        if error_msg := _confirm_write(p, content, encoding):
            return error_msg

        p.parent.mkdir(parents=True, exist_ok=True)

        if offset is None and limit is None:
            with p.open("w", encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {p}"

        actual_offset = offset or 0
        if not p.exists():
            with p.open("wb") as f:
                f.write(b"\0" * actual_offset)
                f.write(content.encode(encoding))
            return f"Created {p} and wrote {len(content)} characters at {actual_offset}"

        with p.open("r+b") as f:
            f.seek(actual_offset)
            tail = b""
            if limit is not None:
                f.seek(actual_offset + limit)
                tail = f.read()
                f.seek(actual_offset)
                f.write(content.encode(encoding))
                f.write(tail)
                f.truncate()
            else:
                f.write(content.encode(encoding))
                f.truncate()

        return f"Successfully updated {p} at offset {actual_offset}"

    try:
        return await asyncio.to_thread(_sync_write)
    except PermissionError as e:
        raise ToolError(str(e)) from e
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Could not write '{path}': {e}") from e
