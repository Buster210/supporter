import atexit
import logging
import os
import platform
import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Queue
from typing import Any

_LEVEL_MAP: dict[str, int] = {
    "off": logging.CRITICAL + 10,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


_FLIGHT_RECORDER: deque[tuple[str, str, str]] = deque(maxlen=20)


def _record(level_name: str, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _FLIGHT_RECORDER.append((ts, level_name, message))


def _dump_flight_recorder(file_handler: RotatingFileHandler | None) -> None:
    if file_handler is None:
        return
    sep = "=" * 72
    lines = [
        "",
        sep,
        "  FLIGHT RECORDER — last context before error",
        sep,
    ]
    for entry in _FLIGHT_RECORDER:
        ts, lvl, msg = entry
        lines.append(f"  {ts}  [{lvl:5s}]  {msg}")
    lines += [sep, ""]
    raw = "\n".join(lines) + "\n"
    try:
        if file_handler.stream:
            file_handler.stream.write(raw)
            file_handler.stream.flush()
    except Exception as exc:
        sys.stderr.write(f"Failed to dump supporter flight recorder: {exc}\n")


_capture_active: bool = False
_pre_capture_level: int | None = None
_capture_lock = threading.Lock()


def _enter_capture(inner: logging.Logger) -> bool:
    global _capture_active, _pre_capture_level
    with _capture_lock:
        if _capture_active:
            return False
        _pre_capture_level = inner.level
        inner.setLevel(logging.DEBUG)
        _capture_active = True
        return True


def _exit_capture(inner: logging.Logger) -> None:
    global _capture_active, _pre_capture_level
    with _capture_lock:
        if not _capture_active:
            return
        if _pre_capture_level is not None:
            inner.setLevel(_pre_capture_level)
        _pre_capture_level = None
        _capture_active = False


class SupporterFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%m/%d/%Y, %I:%M:%S %p"
        )
        level = record.levelname.upper().ljust(5)
        category = record.name
        cat_str = f" [{category}]" if category and category != "supporter" else ""
        return f"{timestamp} [{level}]{cat_str} {record.getMessage()}"


logging.getLogger().setLevel(logging.CRITICAL)

_file_handler: RotatingFileHandler | None = None
_queue_listener: QueueListener | None = None


class _FlightRecorderLogger:
    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("DEBUG", msg)
        self._inner.debug(msg, *args, **kwargs)
        _exit_capture(self._inner)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("INFO", msg)
        self._inner.info(msg, *args, **kwargs)
        _exit_capture(self._inner)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("WARN", msg)
        self._inner.warning(msg, *args, **kwargs)
        _exit_capture(self._inner)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.get("exc_info")
        if exc_info is True:
            msg += f"\n{traceback.format_exc()}"
        _record("ERROR", msg)
        self._inner.error(msg, *args, **kwargs)
        if _enter_capture(self._inner):
            _dump_flight_recorder(_file_handler)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        msg += f"\n{traceback.format_exc()}"
        _record("EXCPT", msg)
        self._inner.exception(msg, *args, **kwargs)
        if _enter_capture(self._inner):
            _dump_flight_recorder(_file_handler)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


logger = _FlightRecorderLogger(logging.getLogger("supporter"))


def init_logger() -> None:
    from .config import config

    global _file_handler, _queue_listener

    try:
        log_path = Path(config.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
        )
        fh.setFormatter(SupporterFormatter())

        supporter_logger = logging.getLogger("supporter")
        _stop_queue_listener()
        _close_file_handler()

        for h in supporter_logger.handlers[:]:
            supporter_logger.removeHandler(h)

        log_queue: Queue[logging.LogRecord] = Queue(-1)
        qh = QueueHandler(log_queue)
        supporter_logger.addHandler(qh)

        _queue_listener = QueueListener(log_queue, fh)
        _queue_listener.start()
        _file_handler = fh
    except Exception as e:
        logger.warning(f"Failed to initialize file logging: {e}")

    raw_level = config.log_level.lower()
    numeric_level = _LEVEL_MAP.get(raw_level, logging.INFO)

    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger("supporter").setLevel(numeric_level)

    logger.info(f"Logging initialized at level: {raw_level!r}")
    logger.debug(
        f"Config — model={config.gemini_model}, provider={config.provider}, "
        f"log_file={config.log_file}, api_keys={len(config.gemini_api_keys)}"
    )
    logger.debug(
        f"Self-Diagnosis — Python={sys.version.split()[0]}, "
        f"OS={platform.system()} {platform.release()}, "
        f"CWD={os.getcwd()}"
    )


def shutdown_logger() -> None:
    from .decision_log import shutdown_decision_logger

    _stop_queue_listener()
    _close_file_handler()
    shutdown_decision_logger()


def _stop_queue_listener() -> None:
    global _queue_listener

    listener = _queue_listener
    _queue_listener = None
    if listener is None:
        return
    try:
        listener.stop()
    except Exception as e:
        logger.warning(f"Failed to stop async log listener [{type(e).__name__}]: {e}")


def _close_file_handler() -> None:
    global _file_handler

    handler = _file_handler
    _file_handler = None
    if handler is None:
        return
    try:
        handler.flush()
    except Exception as e:
        sys.stderr.write(f"Failed to flush supporter log handler: {e}\n")
    try:
        handler.close()
    except Exception as e:
        sys.stderr.write(f"Failed to close supporter log handler: {e}\n")


atexit.register(shutdown_logger)


def main() -> None:
    init_logger()
    logger.info("Test message")


if __name__ == "__main__":
    main()
