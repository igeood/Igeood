import logging
from functools import wraps
from time import time


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt="[%(levelname)s] %(module)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers = []
    logger.addHandler(handler)
    logger.DEBUG = logging.DEBUG
    logger.INFO = logging.INFO
    logger.WARNING = logging.WARNING
    logger.ERROR = logging.ERROR
    logger.propagate = False
    return logger


def timing(fn):
    @wraps(fn)
    def wrap(*args, **kw):
        ts = time()
        result = fn(*args, **kw)
        te = time()
        print("[TIME] %r took: %2.1f sec" % (fn.__name__, te - ts))
        return result

    return wrap


logger = setup_custom_logger("root")
