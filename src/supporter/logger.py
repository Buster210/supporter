import logging
from datetime import datetime

from .config import config


class SupporterFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%m/%d/%Y, %I:%M:%S %p"
        )
        level = record.levelname.upper().ljust(5)
        category = record.name
        cat_str = f" [{category}]" if category and category != "supporter" else ""
        return f"{timestamp} [{level}]{cat_str} {record.getMessage()}"


logger = logging.getLogger("supporter")


def init_logger():

    try:
        with open(config.log_file, "w") as f:
            f.write("")
    except Exception:
        pass

    log_level_str = config.log_level
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(SupporterFormatter())
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized at level: {log_level_str}")


if __name__ == "__main__":
    init_logger()
    logger.info("Test message")
