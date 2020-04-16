"""
Define logger
"""
import os
import time
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def creat_out_dir():
    """
    Create output directory for each runs,
    folder name is timestamp
    """
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, RUN_DIR, timestamp))
    try:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except:
        out_dir = None
    return out_dir

RUN_DIR = "runs"
out_dir = creat_out_dir()

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = "{}/logs.txt".format(out_dir)

def get_console_handler():
    """ Get console handler """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    """ Get file handler """
    try:
        file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding="utf-8")
        file_handler.setFormatter(FORMATTER)
    except:
        file_handler = get_console_handler()
    return file_handler

def get_logger(logger_name):
    """ Get logger """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = True
    return logger