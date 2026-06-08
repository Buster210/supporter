from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import config
from .logger import logger

__all__ = ["DecisionEntry", "log_decision", "recent_decisions"]

_RING_CAPACITY = 256
_DECISIONS_LOGGER_NAME = "supporter.decisions"

# Google API keys are ``AIza`` + 35 url-safe chars; redact those plus any
# configured key so a rationale/context field can never leak a live secret.
_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z\-_]{35}")
_REDACTED = "***"


@dataclass(frozen=True)
class DecisionEntry:
    timestamp: str
    site: str
    chosen: str
    reason: str = ""
    options: tuple[str, ...] = ()
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_RING: deque[DecisionEntry] = deque(maxlen=_RING_CAPACITY)
_decisions_logger: logging.Logger | None = None


def _decision_log_path() -> Path:
    return Path(config.log_file).expanduser().resolve().with_name("decisions.log")


def _get_decisions_logger() -> logging.Logger | None:
    global _decisions_logger
    if _decisions_logger is not None:
        return _decisions_logger
    try:
        path = _decision_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            mode="a",
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger(_DECISIONS_LOGGER_NAME)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for existing in lg.handlers[:]:
            lg.removeHandler(existing)
        lg.addHandler(handler)
        _decisions_logger = lg
    except Exception as exc:
        logger.debug(f"decisions.log init failed [{type(exc).__name__}]: {exc}")
        return None
    return _decisions_logger


def _redact(text: str) -> str:
    if not text:
        return text
    cleaned = _GOOGLE_KEY.sub(_REDACTED, text)
    for key in config.gemini_api_keys:
        if key:
            cleaned = cleaned.replace(key, _REDACTED)
    return cleaned


def log_decision(
    site: str,
    chosen: str,
    *,
    options: Sequence[str] | None = None,
    reason: str = "",
    correlation_id: str | None = None,
) -> None:
    """Record one autonomously-made decision (SPEC §0 rationale, §9 audit trail).

    Append-only to ``decisions.log`` beside the main log file, plus an in-memory
    ring for live inspection. Never raises and never alters caller control flow —
    a logging failure is swallowed at debug level. Secrets in free-text fields are
    redacted before persistence.
    """
    entry = DecisionEntry(
        timestamp=datetime.now().isoformat(timespec="milliseconds"),
        site=site,
        chosen=_redact(chosen),
        reason=_redact(reason),
        options=tuple(_redact(o) for o in options) if options else (),
        correlation_id=correlation_id,
    )
    _RING.append(entry)
    lg = _get_decisions_logger()
    if lg is None:
        return
    try:
        lg.info(json.dumps(entry.to_dict(), ensure_ascii=False))
    except Exception as exc:
        logger.debug(f"decisions.log write failed [{type(exc).__name__}]: {exc}")


def recent_decisions() -> list[DecisionEntry]:
    """Snapshot of the in-memory decision ring, oldest first."""
    return list(_RING)
