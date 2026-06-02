# logs/logging_config.py
"""
Centralised logging configuration for RealProp MVP.
Usage:
    from logs.logging_config import setup_logging
    setup_logging()
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent
LOG_DIR.mkdir(parents=True, exist_ok=True)

APP_LOG_FILE = LOG_DIR / "app.log"
PIPELINE_LOG_FILE = LOG_DIR / "pipeline.log"
VALIDATION_LOG_FILE = LOG_DIR / "validation.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

# ---------------------------------------------------------------------------
# Log level from environment (default: INFO)
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("REALPROP_LOG_LEVEL", "INFO").upper()


# ---------------------------------------------------------------------------
# Config dict
# ---------------------------------------------------------------------------
LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "detailed": {
            "format": (
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "%(filename)s:%(lineno)d | %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(asctime)s | %(levelname)-8s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json_like": {
            "format": (
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","msg":"%(message)s"}'
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },

    "handlers": {
        # Console – always on
        "console": {
            "class": "logging.StreamHandler",
            "level": _LOG_LEVEL,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },

        # General app log – all levels
        "app_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(APP_LOG_FILE),
            "maxBytes": 10 * 1024 * 1024,   # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },

        # Pipeline-specific log
        "pipeline_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(PIPELINE_LOG_FILE),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },

        # Validation-specific log
        "validation_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(VALIDATION_LOG_FILE),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },

        # Error-only log – easy to grep in production
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json_like",
            "filename": str(ERROR_LOG_FILE),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 10,
            "encoding": "utf-8",
        },
    },

    "loggers": {
        # Root realprop namespace → app_file + console
        "realprop": {
            "level": "DEBUG",
            "handlers": ["console", "app_file", "error_file"],
            "propagate": False,
        },

        # Pipeline namespace
        "realprop.validation_pipeline": {
            "level": "DEBUG",
            "handlers": ["validation_file", "console", "error_file"],
            "propagate": False,
        },
        "realprop.rules_pipeline": {
            "level": "DEBUG",
            "handlers": ["pipeline_file", "console", "error_file"],
            "propagate": False,
        },

        # Suppress noisy third-party libs
        "uvicorn": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        "uvicorn.access": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        "httpx": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        "mlflow": {"level": "WARNING", "handlers": ["console"], "propagate": False},
    },

    "root": {
        "level": _LOG_LEVEL,
        "handlers": ["console", "app_file"],
    },
}


def setup_logging() -> None:
    """Apply the logging configuration. Call once at application startup."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.getLogger("realprop").info(
        "Logging initialised | level=%s | logs_dir=%s", _LOG_LEVEL, LOG_DIR
    )