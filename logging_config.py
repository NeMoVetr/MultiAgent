import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = os.getenv("LOG_FILE", "./logs/agents.log")
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "3")),
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    xmpp_log_level = os.getenv("XMPP_LOG_LEVEL", "WARNING").upper()
    xmpp_level = getattr(logging, xmpp_log_level, logging.WARNING)

    logging.getLogger("spade").setLevel(xmpp_level)
    logging.getLogger("slixmpp").setLevel(xmpp_level)
    logging.getLogger("asyncio").setLevel(logging.WARNING)