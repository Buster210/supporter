from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue

from .config import config
from .logger import logger

__all__ = ["DecisionEntry", "log_decision", "recent_decisions", "reset_decision_log"]

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



_RING: deque[DecisionEntry] = deque(maxlen=_RING_CAPACITY)
_decisions_logger: logging.Logger | None = None
_decisions_file_handler: RotatingFileHandler | None = None
_decisions_queue_listener: QueueListener | None = None


def _decision_log_path() -> Path:
    return Path(config.log_file).expanduser().resolve().with_name("decisions.log")


def _get_decisions_logger() -> logging.Logger | None:
    global _decisions_logger, _decisions_file_handler, _decisions_queue_listener
    if _decisions_logger is not None:
        return _decisions_logger
    try:
        path = _decision_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            path,
            mode="a",
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger(_DECISIONS_LOGGER_NAME)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        for existing in lg.handlers[:]:
            lg.removeHandler(existing)
        log_queue: Queue[logging.LogRecord] = Queue(-1)
        qh = QueueHandler(log_queue)
        lg.addHandler(qh)
        _decisions_file_handler = fh
        _decisions_queue_listener = QueueListener(log_queue, fh)
        _decisions_queue_listener.start()
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
        lg.info(json.dumps(asdict(entry), ensure_ascii=False))
    except Exception as exc:
        logger.debug(f"decisions.log write failed [{type(exc).__name__}]: {exc}")


def recent_decisions() -> list[DecisionEntry]:
    """Snapshot of the in-memory decision ring, oldest first."""
    return list(_RING)


def reset_decision_log() -> None:
    """Clear the in-memory decision ring (test isolation / reconfiguration).

    Does NOT touch the logger handle or the on-disk log file — call
    ``shutdown_decision_logger()`` separately for that.
    """
    _RING.clear()


def shutdown_decision_logger() -> None:
    """Stop the decisions queue listener and close the file handler."""
    global _decisions_logger, _decisions_file_handler, _decisions_queue_listener

    listener = _decisions_queue_listener
    _decisions_queue_listener = None
    if listener is not None:
        try:
            listener.stop()
        except Exception as exc:
            logger.debug(
                f"Failed to stop decisions log listener [{type(exc).__name__}]: {exc}"
            )

    handler = _decisions_file_handler
    _decisions_file_handler = None
    if handler is not None:
        try:
            handler.flush()
        except Exception as exc:
            logger.debug(
                f"Failed to flush decisions log handler [{type(exc).__name__}]: {exc}"
            )
        try:
            handler.close()
        except Exception as exc:
            logger.debug(
                f"Failed to close decisions log handler [{type(exc).__name__}]: {exc}"
            )

    if _decisions_logger is not None:
        for h in _decisions_logger.handlers[:]:
            _decisions_logger.removeHandler(h)
