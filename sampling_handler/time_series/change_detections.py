import asyncio
import logging
from pathlib import Path
import time
import pandas as pd

from ..misc import py_helpers, ts_helpers
from .py_change import py_change
from .ccdc import get_ccdc
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


async def async_py_change(gdf, config_dict):
    return py_change(gdf, config_dict)
    #return gdf


async def async_ccdc(gdf, samples, config_dict):
    return get_ccdc(gdf, samples, config_dict)
    #return gdf


async def merge_data(gdf, config_dict, samples):
    py_gdf = await async_py_change(gdf, config_dict)
    ee_gdf = await async_ccdc(gdf, samples, config_dict)
    results = await asyncio.gather(py_gdf, ee_gdf)
    return results


def change_routine(gdf, config_dict, samples=None):

    # get algorithms from config file

    ccdc = config_dict["ccdc_params"]["run"]
    landtrendr = config_dict["landtrendr_params"]["run"]
    glb_prd = config_dict["global_products"]["run"]
    bfast = config_dict["bfast_params"]["run"]
    cusum = config_dict["cusum_params"]["run"]
    bs_slope = config_dict["bs_slope_params"]["run"]
    ts_metrics = config_dict["ts_metrics_params"]["run"]
    jrc_nrt = config_dict["jrc_nrt_params"]["run"]

    # get parameters from configuration file
    ts_params = config_dict["ts_params"]
    bands = ts_params["bands"]
    ts_band = ts_params["ts_band"]
    pid = ts_params["point_id"]

    start = time.time()
    py_helpers.timer(start, "Removing outliers")
    gdf = ts_helpers.remove_outliers(gdf, bands, ts_band) if ts_params["outlier_removal"] else gdf
    py_helpers.timer(start, "Smooth TS")
    gdf = ts_helpers.smooth_ts(gdf, bands) if ts_params["smooth_ts"] else gdf
    # we cut ts data to actual change period only
    py_helpers.timer(start, "Subsetting TS")
    gdf[['dates', 'ts', 'images']] = gdf.apply(
        lambda row: ts_helpers.subset_ts(
            row,
            config_dict['ts_params']['start_calibration'],
            config_dict['ts_params']['end_monitor'],
            bands
        ),
        axis=1,
        result_type="expand",
    )

    # -----------
    # ASYNC
    #if any([bfast, cusum, bs_slope, ts_metrics, jrc_nrt])\
    #        and any([ccdc, glb_prd, landtrendr]):

    #    loop = asyncio.get_event_loop()
    #    cdfs = loop.run_until_complete(merge_data(gdf, config_dict, samples))
    #    loop.close()
    #    print(cdfs)
    #    gdf = pd.merge(cdfs[0], cdfs[1], on=pid)

    if any([bfast, cusum, bs_slope, ts_metrics, jrc_nrt]):
        gdf = py_change(gdf, config_dict)

    if ccdc and samples:

        # check that we have all bands
        check_bpb = all(
            item in bands for item in
            config_dict["ccdc_params"]["breakpointBands"]
        )

        if not check_bpb:
            logger.warning(
                "Selected breakpoint bands for CCDC are not available. "
                "Using the time-series band as breakpoint band."
            )
            config_dict["ccdc_params"]["breakpointBands"] =[ts_band]

        if "tmaskBands" in config_dict["ccdc_params"]:
            check_tmask = all(
                item in bands for item in
                config_dict["ccdc_params"]["tmaskBands"]
            )
            if not check_tmask:
                logger.warning(
                    "Selected tMask bands for CCDC are not available. "
                    "Not using tMask bands."
                )
                config_dict["ccdc_params"]["tmaskBands"] = False

        # run ccdc and add to dataframe
        gdf = get_ccdc(gdf, samples, config_dict)

    return gdf


def run_change(ts_dir, config_dict, samples):

    start = time.time()
    if isinstance(ts_dir, str):
        ts_dir = Path(ts_dir)

    outdir = config_dict["work_dir"]
    if outdir is None:
        outdir = Path.home().joinpath("module_results/sbae_point_analysis")
    else:
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # get all input files
    files = list(ts_dir.glob('*geojson'))

    p, j, to_concat = 0, 0, []
    for i, file in enumerate(files):

        gdf = py_helpers.geojson_to_gdf(file, convert_dates=True)
        p += len(gdf)
        to_concat.append(gdf)

        if p > config_dict['file_accumulation']:
            # accumulate files to dataframe until length is reached
            logger.info(f'Accumulated batch {j+1} of files')
            # run change routine on accumulated files
            py_helpers.timer(start)
            cdf = pd.concat(to_concat)
            logger.info(f'Starting the change detection on {len(cdf)}')
            change_df = change_routine(cdf, config_dict, samples)
            py_helpers.timer(start)
            logger.info("Writing to output.")
            py_helpers.gdf_to_geojson(
                change_df,
                outdir.joinpath(f'{j}_change.geojson'),
                convert_dates=True
            )
            py_helpers.timer(start)
            # reset points and list
            p, to_concat = 0, []
            j += 1

    # if we do not arrive at the file accumulation number,
    # or we have a last file to process.
    if to_concat:
        py_helpers.timer(start)
        logger.info("Starting the change detection")
        change_df = change_routine(pd.concat(to_concat), config_dict)
        py_helpers.timer(start)
        logger.info("Writing to output.")
        py_helpers.gdf_to_geojson(
            change_df,
            outdir.joinpath(f'{j}_change.geojson'),
            convert_dates=True
        )
        py_helpers.timer(start)
