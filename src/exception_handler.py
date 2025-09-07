import logging
from .logging_config import get_logger

logger = get_logger(__name__)

class ExceptionLogging:
    """Context manager for logging exceptions automatically."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error('Exception occurred', exc_info=(exc_type, exc_val, exc_tb))
        return False  # Do not suppress exceptions
