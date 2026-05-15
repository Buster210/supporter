import logging
import sys
from collections.abc import Iterator
from datetime import datetime
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.logger import (
    SupporterFormatter,
    _close_file_handler,
    _dump_flight_recorder,
    _record,
    _stop_queue_listener,
    init_logger,
    logger,
    main,
)


@pytest.fixture(autouse=True)
def _reset_flight_recorder_state() -> Iterator[None]:
    yield
    from supporter import logger as logger_mod

    with logger_mod._capture_lock:
        logger_mod._capture_active = False
        logger_mod._pre_capture_level = None
    logger_mod._FLIGHT_RECORDER.clear()


def test_supporter_formatter() -> None:
    formatter = SupporterFormatter()
    record = logging.LogRecord(
        name="supporter.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2026, 1, 1, 12, 0, 0).timestamp()
    formatted = formatter.format(record)
    assert "[INFO ]" in formatted
    assert "[supporter.test]" in formatted
    assert "Test message" in formatted


def test_supporter_formatter_no_category() -> None:
    formatter = SupporterFormatter()
    record = logging.LogRecord(
        name="supporter",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=1,
        msg="Debug message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "[DEBUG]" in formatted
    assert "[supporter]" not in formatted


def test_init_logger(tmp_path: Any) -> None:
    log_file = tmp_path / "supporter.log"
    with (
        patch("supporter.config.config.log_file", str(log_file)),
        patch("supporter.config.config.log_level", "DEBUG"),
    ):
        init_logger()
        assert log_file.exists()
        assert logger.level == logging.DEBUG
        import supporter.logger

        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in logger.handlers
        ) or any(
            isinstance(h, logging.FileHandler)
            for h in getattr(supporter.logger._queue_listener, "handlers", [])
        )
        assert has_file_handler


def test_init_logger_failure(tmp_path: Any) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with patch("supporter.config.config.log_file", str(log_dir)):
        init_logger()


def test_logger_main_runs_init_and_logs_message(tmp_path: Any) -> None:
    with (
        patch("supporter.logger.init_logger") as mock_init,
        patch("supporter.logger.logger") as mock_supporter_logger,
    ):
        main()
        mock_init.assert_called_once()
        mock_supporter_logger.info.assert_any_call("Test message")


def test_dump_flight_recorder_stream_write_error() -> None:
    _record("INFO", "test entry")
    handler = MagicMock()
    handler.stream.write.side_effect = OSError("disk full")
    captured = StringIO()
    with patch.object(sys, "stderr", captured):
        _dump_flight_recorder(handler)
    assert "Failed to dump supporter flight recorder" in captured.getvalue()


def test_error_with_exc_info_appends_traceback() -> None:
    with patch.object(logger, "_inner") as mock_inner:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("something failed", exc_info=True)
    call_args = mock_inner.error.call_args
    assert "boom" in call_args[0][0]


def test_exception_method_appends_traceback() -> None:
    with patch.object(logger, "_inner") as mock_inner:
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            logger.exception("caught it")
    call_args = mock_inner.exception.call_args
    assert "kaboom" in call_args[0][0]


def test_stop_queue_listener_exception() -> None:
    import supporter.logger

    fake_listener = MagicMock()
    fake_listener.stop.side_effect = RuntimeError("stop failed")
    original = supporter.logger._queue_listener
    try:
        supporter.logger._queue_listener = fake_listener
        _stop_queue_listener()
    finally:
        supporter.logger._queue_listener = original


def test_close_file_handler_flush_exception() -> None:
    import supporter.logger

    fake_handler = MagicMock()
    fake_handler.flush.side_effect = OSError("flush failed")
    original = supporter.logger._file_handler
    captured = StringIO()
    try:
        supporter.logger._file_handler = fake_handler
        with patch.object(sys, "stderr", captured):
            _close_file_handler()
    finally:
        supporter.logger._file_handler = original
    assert "Failed to flush supporter log handler" in captured.getvalue()
    fake_handler.close.assert_called_once()


def test_close_file_handler_close_exception() -> None:
    import supporter.logger

    fake_handler = MagicMock()
    fake_handler.close.side_effect = OSError("close failed")
    original = supporter.logger._file_handler
    captured = StringIO()
    try:
        supporter.logger._file_handler = fake_handler
        with patch.object(sys, "stderr", captured):
            _close_file_handler()
    finally:
        supporter.logger._file_handler = original
    assert "Failed to close supporter log handler" in captured.getvalue()
