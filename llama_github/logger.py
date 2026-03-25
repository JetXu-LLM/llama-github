import logging

logger = logging.getLogger("llama_github")

def configure_logging(level=logging.INFO, handler=None):
    logger.setLevel(level)
    logger.propagate = False
    if handler:
        logger.handlers = []
        logger.addHandler(handler)
        return

    for existing_handler in logger.handlers:
        if isinstance(existing_handler, logging.StreamHandler):
            existing_handler.setLevel(level)
            return

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
