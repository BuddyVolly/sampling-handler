import logging
import time
import sys


def setup_logger(
        logger,
        console_out=True,
        console_level=logging.INFO,
        logfile_out=True,
        logfile_level=logging.DEBUG,
        logfile=""
):

    if not logfile:
        gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())
        logfile = f"/tmp/esbae_log_{gmt}"

    # Configure the logger
    logger.setLevel(logging.DEBUG)  # set the logging level

    log_file_format = logging.Formatter(
        "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    )
    log_console_format = logging.Formatter("%(levelname)s: %(message)s")

    if logfile_out:
        # Add a handler to write logs to a file
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logfile_level)
        file_handler.setFormatter(log_file_format)
        logger.addHandler(file_handler)

    if console_out:
        # Add a handler to write logs to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(log_console_format)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logfile
