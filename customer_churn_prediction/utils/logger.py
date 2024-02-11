"""Logger used throughout the project."""

import logging

from customer_churn_prediction.utils.config import app_config


def get_logger() -> logging.Logger:
    """Set up a python logger for stdout logging.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger()
    logger.setLevel(app_config.LOG_LEVEL.value)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    # Log to stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Log to file
    file_handler = logging.FileHandler("log.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
