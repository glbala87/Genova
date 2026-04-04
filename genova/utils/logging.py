"""Structured logging setup for Genova using loguru.

Provides console and file sinks with configurable levels and structured
JSON output for production use.

Example::

    from genova.utils.logging import setup_logging, get_logger

    setup_logging(level="DEBUG", log_dir="outputs/logs")
    logger = get_logger(__name__)
    logger.info("Training started", epoch=1, lr=1e-4)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# Remove the default stderr sink so we control formatting.
logger.remove()

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - "
    "{message}"
)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    structured: bool = False,
    rotation: str = "100 MB",
    retention: str = "30 days",
    enqueue: bool = True,
) -> None:
    """Configure Genova's logging system.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If *None*, only console output
            is enabled.
        structured: If *True*, emit newline-delimited JSON to the log file
            for ingestion by observability systems.
        rotation: Size or time-based rotation policy for log files.
        retention: How long to keep rotated log files.
        enqueue: If *True*, all log messages are pushed through a
            thread-safe queue (recommended for multiprocess training).
    """
    # Always reset existing sinks to make the function idempotent.
    logger.remove()

    # Console sink -----------------------------------------------------------
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level.upper(),
        colorize=True,
        enqueue=enqueue,
    )

    # File sink(s) -----------------------------------------------------------
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if structured:
            logger.add(
                str(log_path / "genova_{time}.jsonl"),
                serialize=True,
                level=level.upper(),
                rotation=rotation,
                retention=retention,
                enqueue=enqueue,
            )
        else:
            logger.add(
                str(log_path / "genova_{time}.log"),
                format=_FILE_FORMAT,
                level=level.upper(),
                rotation=rotation,
                retention=retention,
                enqueue=enqueue,
            )

    logger.debug("Logging initialised (level={})", level.upper())


def get_logger(name: str = "genova") -> "logger":
    """Return a contextualised logger bound to *name*.

    This is a thin wrapper around ``loguru.logger.bind`` so that the
    ``{name}`` field reflects the calling module.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru logger instance bound with the given name.
    """
    return logger.bind(name=name)
