import asyncio
from pathlib import Path
from typing import Any

import pathspec

from .config import config
from .logger import logger


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

        response_parts = [result.text]

        raw_response = result.raw
        if hasattr(raw_response, "candidates") and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if (
                hasattr(candidate, "grounding_metadata")
                and candidate.grounding_metadata
            ):
                meta = candidate.grounding_metadata

                sources = []
                if hasattr(meta, "grounding_chunks") and meta.grounding_chunks:
                    for chunk in meta.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            title = getattr(chunk.web, "title", "Search Result")
                            url = getattr(chunk.web, "uri", "")
                            if url:
                                sources.append(f"- {title}: {url}")

                if sources:
                    response_parts.append("\n\nSOURCES FOUND:\n" + "\n".join(sources))

        full_response = "\n".join(response_parts)
        logger.debug(f"Tool Success: google_search returned {len(full_response)} chars")
        return full_response

    except Exception as e:
        logger.error(f"Tool Failure: google_search failed: {e}")
        return f"Error performing search: {e!s}"


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
    internal_blacklist = [
        ".env",
        ".gitignore",
        ".git",
        ".venv",
        "src/supporter",
        "uv.lock",
        "pyproject.toml",
    ]
    if any(rel_p_str.startswith(pattern) for pattern in internal_blacklist):
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
