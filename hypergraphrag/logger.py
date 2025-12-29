"""Logging configuration for hypergraphrag"""
import logging
import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def setup_logger(
    name: str = "hypergraphrag",
    level: Optional[int] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure logger
    
    Args:
        name: Logger name
        level: Logging level (default: INFO, or from LOG_LEVEL env var)
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Get log level from environment or use default
    if level is None:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, logging.INFO)
    
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to parent loggers to avoid duplicate logs
    logger.propagate = False
    
    if level != logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("neo4j").setLevel(logging.WARNING)
    
    return logger


# Default logger instance
logger = setup_logger()
