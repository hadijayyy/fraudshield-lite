"""
Shared utility functions for FraudShield.

Provides configuration loading, logging setup, and other cross-cutting
concerns used by multiple modules.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load the project configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file (relative to project root or
        absolute).

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logging.warning("Config file not found at %s; returning empty config.", config_path)
        return {}

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config if config else {}


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "logs/fraudshield.log",
) -> None:
    """Configure logging for the FraudShield application.

    Sets up both console (stdout) and rotating file handlers with a standard
    format including timestamps and log levels.

    Parameters
    ----------
    level : int
        Logging level (e.g., ``logging.INFO``, ``logging.DEBUG``).
    log_file : str
        Path to the log file.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # Already configured
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    logging.info("Logging configured — writing to %s", log_path)
