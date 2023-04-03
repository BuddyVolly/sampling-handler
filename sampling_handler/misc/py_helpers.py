import os
import subprocess
import shlex
import time
import logging
from datetime import timedelta
import concurrent.futures
from pathlib import Path

import ee
import geemap

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


def save_gdf_locally(gdf, outdir=None, ceo_csv=True, gpkg=True):

    # if it is already a feature collection
    if isinstance(gdf, ee.FeatureCollection):
        gdf = geemap.ee_to_geopandas(gdf)

    if not outdir:
        outdir = Path.home().joinpath("module_results/e_sbae")

    if not isinstance(outdir, Path):
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f" Saving outputs to {outdir}")
    gdf["LON"] = gdf["geometry"].x
    gdf["LAT"] = gdf["geometry"].y

    # sort columns for CEO output
    gdf["PLOTID"] = gdf["point_id"]
    cols = gdf.columns.tolist()
    cols = [e for e in cols if e not in ("LON", "LAT", "PLOTID")]
    new_cols = ["PLOTID", "LAT", "LON"] + cols
    gdf = gdf[new_cols]

    if ceo_csv:
        gdf[["PLOTID", "LAT", "LON"]].to_csv(
            outdir.joinpath("01_sbae_points.csv"), index=False
        )

    if gpkg:
        gdf.to_file(outdir.joinpath("01_sbae_points.gpkg"), driver="GPKG")


def split_dataframe(df, chunk_size=25000):
    """Split a pandas DataFrame into different chunks

    """

    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks
