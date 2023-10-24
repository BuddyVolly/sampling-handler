import os
import json
import subprocess
import shlex
import time
import logging
from datetime import timedelta
import concurrent.futures
from pathlib import Path

import ee
import geemap
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .settings import setup_logger


# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


from threading import Thread
class ThreadWithReturnValue(Thread):

    def __init__(
            self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


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


def run_in_parallel(func, arg_list, workers, parallelization='threads'):

    # get max workers and
    if parallelization == 'threads':
        executor = concurrent.futures.ThreadPoolExecutor(workers)
    elif parallelization == 'processes':
        executor = concurrent.futures.ProcessPoolExecutor(workers)
    else:
        raise ValueError('Parallelization type not supported or unknown.')

    # submit tasks
    futures = [executor.submit(func, *args) for args in arg_list]

    # gather results
    results = [
        future.result()
        for future in concurrent.futures.as_completed(futures)
    ]
    executor.shutdown()
    return results


def save_gdf_locally(gdf, outdir=None, ceo_csv=None, gpkg=None, pid='point_id'):

    # if it is already a feature collection
    if isinstance(gdf, ee.FeatureCollection):
        gdf = geemap.ee_to_geopandas(gdf)

    if not outdir:
        outdir = Path.home().joinpath("module_results/esbae")

    if not isinstance(outdir, Path):
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    logger.debug(f'Saving outputs to {outdir}')

    if gpkg:
        gdf.to_file(outdir.joinpath(gpkg), driver="GPKG")

    if ceo_csv:

        gdf["LON"] = gdf["geometry"].x
        gdf["LAT"] = gdf["geometry"].y
        # sort columns for CEO output
        gdf["PLOTID"] = gdf[pid]
        cols = gdf.columns.tolist()
        cols = [e for e in cols if e not in ("LON", "LAT", "PLOTID")]
        new_cols = ["PLOTID", "LAT", "LON"] + cols
        gdf = gdf[new_cols]

        gdf[["PLOTID", "LAT", "LON"]].to_csv(
            outdir.joinpath(ceo_csv), index=False
        )


def split_dataframe(df, chunk_size=25000):
    """Split a pandas DataFrame into different chunks

    """

    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def read_any_aoi_to_single_row_gdf(aoi, incrs=None, outcrs='epsg:4326'):
    """

    :param aoi:
    :return:
    """

    if isinstance(aoi, ee.FeatureCollection):
        logger.debug("Turning ee FC into a GeoDataFrame")
        aoi = geemap.ee_to_geopandas(aoi).set_crs("epsg:4326", inplace=True)

    if isinstance(aoi, (str, Path)):
        aoi = gpd.read_file(aoi)

    if isinstance(aoi, dict):   # Feature Collection json type object
        try:
            aoi = gpd.GeoDataFrame.from_features(aoi)
        except:
            pass

    if not isinstance(aoi, gpd.geodataframe.GeoDataFrame):

        raise ValueError(
            'Area of Interest does not have the right format. '
            'Shall be either a path to a file, a geopandas GeoDataFrame '
            'or a Earth Engine Feature Collection'
        )

    if not aoi.crs:

        if incrs:
            aoi.set_crs(incrs, inplace=True)
        else:
            crs_original = input(
                "Your AOI does not have a coordinate reference system (CRS). "
                "Please provide the CRS of the AOI (e.g. epsg:4326): "
            )
            aoi.set_crs(crs_original, inplace=True)

    # ensure single geometry
    logger.debug("Dissolve geometry and reproject to crs.")
    geom = gpd.GeoDataFrame(aoi.geometry.explode('index_parts=True'))
    geom.drop(geom[geom['geometry'].type != 'Polygon'].index, inplace=True)

    return gpd.GeoDataFrame(
        index=[0],
        crs=outcrs,
        geometry=[geom.to_crs(outcrs).unary_union]
    )


def gdf_to_geojson(gdf, outfile, convert_dates=False):

    if convert_dates:
        gdf['dates'] = gdf.dates.apply(
            lambda row: [pd.to_datetime(ts).strftime('%Y%m%d') for ts in
                         list(row.values)])

    # this is how we dump
    with open(outfile, 'w') as outfile:
        json.dump(gdf.to_json(), outfile)


def convert_to_datetime(date):
    return pd.to_datetime(date, format='%Y%m%d')


def geojson_to_gdf(infile, convert_dates=False, cols=False, crs='epsg:4326'):

    # this is how we load
    with open(infile, 'r') as outfile:
        gdf = gpd.GeoDataFrame.from_features(
            json.loads(json.load(outfile))
        )

    # convert plain list of dates into a pandas datetime index
    if convert_dates:
        #results = py_helpers.run_in_parallel(convert_date, gdf['dates'].tolist())

        gdf['dates'] = gdf.dates.apply(
            lambda dates: pd.DatetimeIndex(
                [pd.to_datetime(date, format='%Y%m%d') for date in dates]
            )
        )

    if cols:
        return gdf[cols]
    else:
        return gdf.set_crs(crs)


def aggregate_outfiles(directory, convert_dates=False):  # glob all files in the data augmentation output folder

    files = Path(directory).glob('*geojson')
    # prepare for parallel execution
    files = [[str(file), convert_dates] for file in files]

    # read files in parallel nad put the in a list
    result = run_in_parallel(
        geojson_to_gdf,
        files,
        workers=os.cpu_count() * 2,
        parallelization='processes'
    )

    # concatenate dataframes from result's list
    return pd.concat(result)


def get_scalebar_distance(samples):

    bounds = samples.geometry.unary_union.bounds
    p1 = Point(bounds[0], bounds[1])
    p2 = Point(bounds[0], bounds[1]+1)

    points = gpd.GeoSeries([p1, p2], crs='epsg:4326')
    points = points.to_crs('EPSG:32662') # Plate Carree
    distance_meters = points[0].distance(points[1])
    return distance_meters
