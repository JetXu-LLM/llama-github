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

## Logging Configuration
# By default, `llama_github` does not configure logging to avoid interfering with your application's logging settings. If you want to see log outputs from `llama_github`, you can configure logging as follows:

# ```python
# import logging
# import llama_github

# # Configure the library's logger
# llama_github.configure_logging(level=logging.INFO)

# # Now you can see log outputs
# logger = logging.getLogger('llama_github')