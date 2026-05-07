"""
Logging Configuration
======================

Sets up structured logging for experiment tracking.
Logs to both console and file for reproducibility.
"""

import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "rsid",
    log_dir: str = "./results",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure logger with console + file output.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler  --  shows INFO and above
    console = logging.StreamHandler()
    console.setLevel(level)
    console_fmt = logging.Formatter("%(message)s")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler  --  saves everything with timestamps (UTC, to avoid local-TZ leaks)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"rsid_{timestamp}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
