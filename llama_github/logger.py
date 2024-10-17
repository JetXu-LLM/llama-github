import logging

logger = logging.getLogger('llama_github')

def configure_logging(level=logging.INFO, handler=None):
    logger.setLevel(level)
    if handler:
        logger.addHandler(handler)
    else:
        # default handler output to console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)