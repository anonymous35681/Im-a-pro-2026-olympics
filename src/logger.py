import sys

from loguru import logger

# Configure logging with custom format
logger.remove()
LOG_FORMAT = (
    "<green>{time:DD.MM.YYYY HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<white>{message}</white>"
)
logger.add(sys.stderr, format=LOG_FORMAT, level="INFO")


def setup_logger() -> None:
    """Initialize logger configurations."""
    logger.info("Logger initialized.")
