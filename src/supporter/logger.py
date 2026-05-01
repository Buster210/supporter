import logging
import os
import platform
import sys
import traceback
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from .config import config

_LEVEL_MAP: dict[str, int] = {
    "off": logging.CRITICAL + 10,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


_FLIGHT_RECORDER: deque[tuple[str, str, str]] = deque(maxlen=100)


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
        _ = exc


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


class _FlightRecorderLogger:
    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("DEBUG", msg)
        self._inner.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("INFO", msg)
        self._inner.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        _record("WARN", msg)
        self._inner.warning(msg, *args, **kwargs)
        _dump_flight_recorder(_file_handler)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.get("exc_info")
        if exc_info is True:
            msg += f"\n{traceback.format_exc()}"
        _record("ERROR", msg)
        self._inner.error(msg, *args, **kwargs)
        _dump_flight_recorder(_file_handler)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        msg += f"\n{traceback.format_exc()}"
        _record("EXCPT", msg)
        self._inner.exception(msg, *args, **kwargs)
        _dump_flight_recorder(_file_handler)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


logger = _FlightRecorderLogger(logging.getLogger("supporter"))


def init_logger() -> None:
    global _file_handler

    try:
        fh = RotatingFileHandler(
            config.log_file,
            mode="a",
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
        )
        fh.setFormatter(SupporterFormatter())

        supporter_logger = logging.getLogger("supporter")
        for h in supporter_logger.handlers[:]:
            supporter_logger.removeHandler(h)

        supporter_logger.addHandler(fh)
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


def main() -> None:
    init_logger()
    logger.info("Test message")


if __name__ == "__main__":
    main()
