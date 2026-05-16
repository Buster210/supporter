import functools
from pathlib import Path

from ..config import config


@functools.cache
def _resolve_path(key: str) -> Path:
    return Path(key).expanduser().resolve()


def resolved_project_root() -> Path:
    if not config.allowed_directories:
        raise PermissionError("No allowed directories set. Check your configuration.")
    return _resolve_path(config.allowed_directories[0])
