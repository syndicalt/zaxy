"""Structured logging setup for Zaxy.

Provides JSON-formatted logs in production and human-readable logs in dev.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from zaxy.config import get_settings


def setup_logging() -> None:
    """Configure root logger based on Settings.

    - console: human-readable, colored (default for dev)
    - json: machine-readable, one line per record (default for prod)
    """
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("zaxy")
    root.setLevel(level)
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the zaxy namespace."""
    return logging.getLogger(f"zaxy.{name}")


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
