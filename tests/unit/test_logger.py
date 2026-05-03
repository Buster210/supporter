import logging
from datetime import datetime
from typing import Any
from unittest.mock import patch

from supporter.logger import SupporterFormatter, init_logger, logger, main


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
        has_queue_handler = any(
            isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers
        )
        assert has_queue_handler


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
