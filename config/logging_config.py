"""Logging Configuration"""
import logging
import sys
from config.settings import settings


def setup_logging():
    """Configure logging for the application"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    
    logger = logging.getLogger()
    logger.setLevel(settings.log_level)
    logger.addHandler(handler)
    
    # Set third-party loggers to WARNING
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    
    return logger


logger = setup_logging()
