from __future__ import annotations

import logging


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a stream logger with a consistent format."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level.upper())

    return logger

