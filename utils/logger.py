import logging


LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)
formatter = logging.Formatter(LOGGER_FORMAT)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
