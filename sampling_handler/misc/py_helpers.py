import os
import subprocess
import shlex
import time
import logging
from datetime import timedelta
import concurrent.futures

from .settings import setup_logger


# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


def run_command(command, logfile=None, elapsed=True, stdout=True, stderr=True):
    """bla    """

    currtime = time.time()

    # define output behaviour
    stdout = subprocess.STDOUT if stdout else subprocess.DEVNULL
    stderr = subprocess.STDOUT if stderr else subprocess.DEVNULL

    if os.name == "nt":
        process = subprocess.run(command, stderr=stderr, stdout=stdout)
    else:
        process = subprocess.run(
            shlex.split(command), stdout=stdout, stderr=stderr
        )

    return_code = process.returncode

    if return_code != 0 and logfile is not None:
        with open(str(logfile), "w") as file:
            for line in process.stderr.decode().splitlines():
                file.write(f"{line}\n")

    if elapsed:
        timer(currtime)

    return process.returncode


def timer(start, custom_msg=None):
    """A helper function to print a time elapsed statement

    :param start:
    :type start:
    :return:
    :rtype: str
    """

    elapsed = time.time() - start
    if custom_msg:
        logger.info(f"{custom_msg}: {timedelta(seconds=elapsed)}")
    else:
        logger.info(f"Time elapsed: {timedelta(seconds=elapsed)}")


def _run_in_threads(func, arg_list, config_dict):

    max_workers = config_dict["workers"]
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:

        # submit tasks
        futures = [executor.submit(func, *args) for args in arg_list]

        # gather results
        try:
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

            if None not in results:
                return_code = 0
            else:
                return_code = 1
        except Exception as e:
            return_code = 1

    return return_code


