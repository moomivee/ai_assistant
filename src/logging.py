"""Operational JSON logs, deliberately without request bodies and credentials."""

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone

request_id = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": request_id.get(),
                **getattr(record, "fields", {}),
            },
            ensure_ascii=False,
            default=str,
        )


def configure_logging():
    logger = logging.getLogger("analyst")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log(event, level=logging.INFO, **fields):
    configure_logging()
    logging.getLogger("analyst").log(level, event, extra={"fields": fields})
