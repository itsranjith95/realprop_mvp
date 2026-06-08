# ============================================================
# src/core/logger.py
# Structured Python logging for RealProp MVP
# Phase 8.3 — Observability (simple → CloudWatch-ready)
#
# Usage:
#   from src.core.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Pipeline started", extra={"doc_id": "abc123"})
# ============================================================

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENV = os.getenv("ENV", "development")

# Log file paths  (mirroring the logs/ folder already in the repo)
APP_LOG_FILE     = LOG_DIR / "app.log"
PIPELINE_LOG_FILE = LOG_DIR / "pipeline.log"
ERROR_LOG_FILE   = LOG_DIR / "errors.log"


# ── JSON Formatter ────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """
    Emits one JSON object per log line — compatible with:
    - CloudWatch Logs Insights  (TODO: plug in awslogs driver on ECS)
    - OpenTelemetry log exporter (TODO: add OTLP handler below)
    - Datadog / Grafana Loki (TODO: add structured log ingestion)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "env":       ENV,
        }

        # Include extra context fields (doc_id, pipeline_step, etc.)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__ and not k.startswith("_")
               and k not in ("msg", "args", "levelname", "name", "pathname",
                             "filename", "module", "exc_info", "exc_text",
                             "stack_info", "lineno", "funcName", "created",
                             "msecs", "relativeCreated", "thread", "threadName",
                             "processName", "process", "message", "taskName")
        }
        if extras:
            log_obj["context"] = extras

        # Attach exception traceback if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


# ── Plain Text Formatter (for local dev) ─────────────────────
class DevFormatter(logging.Formatter):
    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


# ── Logger Factory ────────────────────────────────────────────
def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    In production (ENV=production): emits JSON to file + stderr.
    In development: emits human-readable text to stdout + file.

    Args:
        name:     Module name, typically __name__
        log_file: Optional specific log file path. Defaults to APP_LOG_FILE.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    target_file = log_file or APP_LOG_FILE

    if ENV == "production":
        # ── Production: JSON to rotating file + stderr ──────────
        file_handler = logging.handlers.RotatingFileHandler(
            filename=target_file,
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(JSONFormatter())
        logger.addHandler(stderr_handler)

        # TODO: Add CloudWatch handler (post-MVP)
        # from watchtower import CloudWatchLogHandler
        # cw_handler = CloudWatchLogHandler(
        #     log_group="/realprop/app",
        #     log_stream_name=f"{ENV}/{name}",
        # )
        # logger.addHandler(cw_handler)

        # TODO: Add OpenTelemetry log handler (post-MVP)
        # from opentelemetry.sdk._logs import LoggerProvider
        # from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        # ... configure and attach OTLP handler here

    else:
        # ── Development: plain text to stdout + file ─────────────
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(DevFormatter())
        logger.addHandler(stdout_handler)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=target_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(DevFormatter())
        logger.addHandler(file_handler)

    # ── Always write ERRORs to errors.log ────────────────────
    error_handler = logging.handlers.RotatingFileHandler(
        filename=ERROR_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)

    return logger


# ── Pipeline-specific logger ──────────────────────────────────
def get_pipeline_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to pipeline.log.
    Use this in src/pipelines/*.py modules.
    """
    return get_logger(name, log_file=PIPELINE_LOG_FILE)


# ── Module-level default logger ──────────────────────────────
logger = get_logger("realprop")