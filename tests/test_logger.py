import logging
import pytest
from llama_github.logger import configure_logging, logger

def test_configure_logging_defaults():
    """Test default logging configuration."""
    # Reset handlers
    logger.handlers = []
    
    configure_logging()
    
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_configure_logging_custom_level():
    """Test logging with custom level."""
    logger.handlers = []
    configure_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG

def test_configure_logging_custom_handler():
    """Test logging with a custom handler."""
    logger.handlers = []
    custom_handler = logging.NullHandler()
    configure_logging(handler=custom_handler)
    
    assert len(logger.handlers) == 1
    assert logger.handlers[0] == custom_handler