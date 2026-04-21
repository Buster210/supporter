import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pathspec

from .config import config
from .logger import logger

_CONFIRMATION_CALLBACK: Callable[[Path, str], bool] | None = None


def set_confirmation_callback(callback: Callable[[Path, str], bool] | None) -> None:
    global _CONFIRMATION_CALLBACK
    _CONFIRMATION_CALLBACK = callback


async def google_search(query: str) -> str:
    logger.info(f"Tool Execute: google_search(query='{query}')")

    from .index import get_provider

    provider = get_provider(live=True, model_name=config.gemini_live_fallback_model)

    try:
        result = await provider.generate(
            prompt=query,
            options={
                "use_search": True,
                "system_instruction": (
                    "You are a search expert. Provide a detailed, highly accurate "
                    "answer based on the search results. Include all relevant "
                    "facts, figures, and technical details. Format the output "
                    "to be consumed by another LLM."
                ),
            },
        )

        candidates = getattr(result.raw, "candidates", None)
        if not candidates:
            return result.text

        meta = getattr(candidates[0], "grounding_metadata", None)
        if not meta:
            return result.text

        sources = []
        grounding_chunks = getattr(meta, "grounding_chunks", []) or []
        for chunk in grounding_chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue

            url = getattr(web, "uri", "")
            if url:
                title = getattr(web, "title", "Search Result")
                sources.append(f"- {title}: {url}")

        if not sources:
            return result.text

        full_response = f"{result.text}\n\n\nSOURCES FOUND:\n" + "\n".join(sources)
        logger.debug(f"Tool Success: google_search returned {len(full_response)} chars")
        return full_response

    except Exception as e:
        logger.error(f"Tool Failure: google_search failed: {e}")
        return f"Error performing search: {e!s}"


_INTERNAL_BLACKLIST = [
    ".env",
    ".gitignore",
    ".git",
    ".venv",
    "src/supporter",
    "uv.lock",
    "pyproject.toml",
]

_GITIGNORE_CACHE: dict[str, Any] = {"spec": None, "mtime": 0}


def _get_gitignore_spec(project_root: Path) -> pathspec.PathSpec | None:
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return None

    try:
        mtime = gitignore_path.stat().st_mtime
        if _GITIGNORE_CACHE["spec"] is not None and _GITIGNORE_CACHE["mtime"] == mtime:
            from typing import cast

            return cast(pathspec.PathSpec, _GITIGNORE_CACHE["spec"])

        with gitignore_path.open("r") as f:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
            _GITIGNORE_CACHE["spec"] = spec
            _GITIGNORE_CACHE["mtime"] = mtime
            return spec
    except Exception as e:
        logger.warning(f"Could not process .gitignore: {e}")
        return None


def _validate_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    project_root = Path(config.allowed_directories[0]).expanduser().resolve()

    if not (p == project_root or project_root in p.parents):
        raise PermissionError(
            f"Access denied: Path {p} is outside project root: {project_root}"
        )

    rel_p_str = str(p.relative_to(project_root))
    if any(rel_p_str.startswith(pattern) for pattern in _INTERNAL_BLACKLIST):
        raise PermissionError(
            f"Access denied: {rel_p_str} is a protected internal file."
        )

    spec = _get_gitignore_spec(project_root)
    if spec and spec.match_file(rel_p_str):
        raise PermissionError(f"Access denied: {rel_p_str} is ignored by .gitignore")

    return p


async def read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> str:
    logger.info(
        f"Tool Execute: read_file(path='{path}', offset={offset}, limit={limit})"
    )

    def _sync_read() -> str:
        p = _validate_path(path)
        if not p.exists():
            return f"Error: File not found: {p}"

        with p.open("r", encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            if limit is not None:
                return f.read(limit)
            return f.read()

    try:
        return await asyncio.to_thread(_sync_read)
    except Exception as e:
        logger.error(f"Tool Failure: read_file failed: {e}")
        return f"Error reading file: {e!s}"


async def write_file(
    path: str,
    content: str,
    offset: int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> str:
    logger.info(
        f"Tool Execute: write_file(path='{path}', content_len={len(content)}, "
        f"offset={offset}, limit={limit})"
    )

    def _sync_write() -> str:
        p = _validate_path(path)

        if config.require_write_confirmation:
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
                    [],
                    lines,
                    fromfile="/dev/null",
                    tofile=f"b/{p.name}",
                    n=len(lines),
                )
                display_diff = "".join(diff)

            if _CONFIRMATION_CALLBACK:
                if not _CONFIRMATION_CALLBACK(p, display_diff):
                    return "Write operation cancelled by user security preference."
            elif sys.stdin.isatty():
                print("\n" + "=" * 60)
                print(" SECURITY CONFIRMATION REQUIRED ".center(60, "="))
                print(f" TARGET FILE: {p}")
                print("-" * 60)
                print(" PROPOSED DIFF ".center(60, "-"))
                print(display_diff)
                print("-" * 60)
                confirm = input(" Proceed with write? (y/n): ").lower().strip()
                print("=" * 60 + "\n")
                if confirm not in ("y", "yes"):
                    return "Write operation cancelled by user security preference."
            else:
                logger.warning(
                    f"Write confirmation required for {p} but no interactive TTY "
                    "or callback available. Skipping write for safety."
                )
                return "Error: Interactive confirmation required but unavailable."

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
    except Exception as e:
        logger.error(f"Tool Failure: write_file failed: {e}")
        return f"Error writing file: {e!s}"


async def list_dir(path: str) -> str:
    logger.info(f"Tool Execute: list_dir(path='{path}')")

    def _sync_list() -> str:
        p = _validate_path(path)
        if not p.is_dir():
            return f"Error: Path is not a directory: {p}"

        project_root = Path(config.allowed_directories[0]).expanduser().resolve()
        rel_parent = p.relative_to(project_root)
        spec = _get_gitignore_spec(project_root)

        items = []
        try:
            with os.scandir(p) as it:
                for entry in it:
                    rel_path = str(rel_parent / entry.name)

                    if any(rel_path.startswith(p) for p in _INTERNAL_BLACKLIST):
                        continue
                    if spec and spec.match_file(rel_path):
                        continue

                    try:
                        tag = "[DIR]" if entry.is_dir() else "[FILE]"
                        items.append(f"{tag} {entry.name}")
                    except OSError:
                        continue
        except OSError as e:
            return f"Error accessing directory: {e!s}"

        if not items:
            return "Directory is empty or all items are restricted."

        items.sort()
        return "\n".join(items)

    try:
        return await asyncio.to_thread(_sync_list)
    except Exception as e:
        logger.error(f"Tool Failure: list_dir failed: {e}")
        return f"Error listing directory: {e!s}"
