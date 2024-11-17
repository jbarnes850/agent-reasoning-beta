"""Logging utilities for the Agent Reasoning Beta platform."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from utils.config import ConfigManager


class LogFormatter(logging.Formatter):
    """Custom formatter with color support and structured output."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        """Initialize formatter with color support.

        Args:
            use_color: Whether to use colored output.
        """
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional color."""
        # Save original values
        orig_levelname = record.levelname
        orig_msg = record.msg

        if self.use_color:
            color = self.COLORS.get(record.levelno, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            if isinstance(record.msg, str):
                record.msg = f"{color}{record.msg}{self.RESET}"

        result = super().format(record)

        # Restore original values
        record.levelname = orig_levelname
        record.msg = orig_msg

        return result


class Logger:
    """Centralized logging management for the application."""

    _instance: Optional[Logger] = None

    def __new__(cls) -> Logger:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger if not already initialized."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._config = ConfigManager()
            self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(
            logging.DEBUG if self._config.config.env.debug else logging.INFO
        )

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(LogFormatter(use_color=True))
        root_logger.addHandler(console_handler)

        # File handler
        log_dir = self._config.get_log_dir()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"agent_reasoning_{current_time}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(LogFormatter(use_color=False))
        root_logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name. If None, returns the root logger.

        Returns:
            Configured logger instance.
        """
        if cls._instance is None:
            cls()
        return logging.getLogger(name)

    @classmethod
    def set_level(cls, level: Union[int, str]):
        """Set the root logger level.

        Args:
            level: Logging level (e.g., "DEBUG", "INFO", etc.)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logging.getLogger().setLevel(level)


# Convenience functions
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    return Logger.get_logger(name)


def set_log_level(level: Union[int, str]):
    """Set the root logger level."""
    Logger.set_level(level)
