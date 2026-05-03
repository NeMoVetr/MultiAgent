import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_file = Path(os.getenv("LOG_FILE", "./logs/agents.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = UTCFormatter(
        fmt="%(asctime)sZ | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=int(os.getenv("LOG_MAX_BYTES", "5242880")),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
        force=True,
    )

    logging.getLogger("spade").setLevel(
        getattr(logging, os.getenv("SPADE_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    )
    logging.getLogger("slixmpp").setLevel(
        getattr(logging, os.getenv("SLIXMPP_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    )
    logging.getLogger("aiomqtt").setLevel(
        getattr(logging, os.getenv("AIOMQTT_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    )