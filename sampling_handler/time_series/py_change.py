import time
from datetime import datetime as dt
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from bfast import BFASTMonitor

from ..misc.py_helpers import run_in_parallel, timer
from ..misc.ts_helpers import subset_ts
from ..misc.settings import setup_logger
from .cusum import cusum_deforest
from .jrc_nrt import run_jrc_nrt

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)
# default bFast parameters
defaults = {
    "start_monitor": dt.strptime("2000-01-01", "%Y-%m-%d"),
    "freq": 365,
    "k": 3,
    "hfrac": 0.25,
    "trend": False,
    "level": 0.05,
    "backend": "python",
}


def bfast_monitor(data, dates, point_id, bfast_params):
    """
    Wrapper for BFAST's python implementation

    Parameters
    ----------

    dates : int
        list of dates for the time-series
    data : float
        list/array of time-series data
    start_monitor : datetime object
        start of the monitoring period
    bfast_params : dict
        dictionary of bfast parameters

    Returns
    -----------

    bfast_date : float32
        Change Date in fractional year date format
    bfast_magnitude : float32
        Change magnitude of detected break
    bfast_means : float32
        Change confidence of detected break
    """

    warnings.filterwarnings(
        'ignore', 'invalid value', RuntimeWarning
    )
    warnings.filterwarnings(
        'ignore', 'divide by zero', RuntimeWarning
    )

    # initialize model
    params = bfast_params.copy()
    params.update(
        start_monitor=dt.strptime(bfast_params["start_monitor"], "%Y-%m-%d")
    )

    params.pop("run", None)
    model = BFASTMonitor(**params)

    # check if we have dates in the monitoring period
    mon_dates = [date for date in dates if date > params["start_monitor"]]
    if mon_dates:
        # fit gistorical period
        model.fit(data, dates)

        # get breaks in the monitoring period
        if model.breaks < 0:
            # in case not enough images or no breaks
            bfast_date = model.breaks
            # get magnitude and means
            bfast_magnitude = 0
            bfast_means = 0
        else:
            # get index of break
            bfast_date = mon_dates[model.breaks - 1]
            # transform dates to fractional years
            bfast_date = bfast_date.year + np.round(bfast_date.dayofyear / 365, 3)

            # get magnitude and means
            bfast_magnitude = model.magnitudes
            bfast_means = model.means
    else:
        # no image in historical period
        bfast_date = -2
        # get magnitude and means
        bfast_magnitude = 0
        bfast_means = 0

    return bfast_date, bfast_magnitude, bfast_means, point_id


def timescan(ts, point_id, outlier_removal, z_threshhold):

    if ts:
        if outlier_removal:
            z_score = np.abs(stats.zscore(np.array(ts)))
            ts_out = np.ma.MaskedArray(ts, mask=z_score > z_threshhold)
        else:
            ts_out = ts

        return np.nanmean(ts_out), np.nanstd(ts_out), np.nanmin(ts_out), np.nanmax(ts_out), point_id

    else:
        return -1, -1, -1, -1, point_id

def _slope(x, y):

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


def bootstrap_slope(y, x, nr_bootstraps, point_id):
    # This function takes x and y and calculates the bootstrap on the
    # slope of the linear regression between both,
    # whereas values are sorted

    if x:
        x, y = np.array(x), np.array(y)

        boot_means = []
        for _ in range(nr_bootstraps):

            # the fraction of sample we want to include (randon)
            # size = np.abs(np.random.normal(0.5, 0.1))
            size = 0.66

            # select the random samples
            rand_idx = sorted(
                np.random.choice(
                    np.arange(y.size), int(y.size * size), replace=False
                )
            )

            # calculate the slope on the randomly selected samples
            s = _slope(x[rand_idx], y[rand_idx])

            # add to list of bootstrap samples
            boot_means.append(s)

        # calculate stats adn return
        boot_means_np = np.array(boot_means)
        return (
            np.mean(boot_means_np),
            np.std(boot_means_np),
            np.max(boot_means_np),
            np.min(boot_means_np),
            point_id,
        )
    else:
        return -1, -1, -1, -1, point_id


def py_change(df, config_dict):
    """
    Parallel implementation of the bfast_monitor function
    """

    pid = config_dict["ts_params"]["point_id"]
    bands = config_dict['ts_params']['bands']
    ts_band = config_dict["ts_params"]["ts_band"]
    bfast_params = config_dict["bfast_params"]

    # run criteria
    bfast = config_dict["bfast_params"]["run"]
    cusum = config_dict["cusum_params"]["run"]
    ts_metrics = config_dict["ts_metrics_params"]["run"]
    bs_slope = config_dict["bs_slope_params"]["run"]
    jrc_nrt = config_dict["jrc_nrt_params"]["run"]

    # algorithmic specific params
    bs_bootstraps = config_dict['bs_slope_params']['nr_of_bootstraps']
    cusum_bootstraps = config_dict["cusum_params"]["nr_of_bootstraps"]
    ts_metrics_params = config_dict["ts_metrics_params"]
    outlier_removal, z_threshhold = (
        ts_metrics_params["outlier_removal"],
        ts_metrics_params["z_threshhold"],
    )

    # get monitor only
    if any([bs_slope, cusum, ts_metrics]):
        df[['dates_mon', 'ts_mon', 'mon_images']] = df.apply(
            lambda row: subset_ts(
                row,
                config_dict['ts_params']['start_monitor'],
                config_dict['ts_params']['end_monitor'],
                bands
            ),
            axis=1,
            result_type='expand',
        )

    bfast_args, bs_args, cusum_args, ts_args, nrt_args, d = [], [], [], [], [], {}
    for i, row in df.iterrows():

        if bfast:
            bfast_args.append(
                [row.ts[ts_band], row.dates, row[pid], bfast_params]
            )

        if cusum or bs_slope:
            dates_float = [
                (date.year + np.round(date.dayofyear / 365, 3))
                for date in row.dates_mon
            ]

        if cusum:
            cusum_args.append(
                [row.ts_mon[ts_band], dates_float, row[pid], cusum_bootstraps]
            )

        if ts_metrics:
            ts_args.append(
                [row.ts_mon[ts_band], row[pid], outlier_removal, z_threshhold]
            )

        if bs_slope:
            bs_args.append(
                [row.ts_mon[ts_band], dates_float, bs_bootstraps, row[pid]]
            )

        if jrc_nrt:
            nrt_args.append([row.ts[ts_band], row.dates, row[pid], config_dict["ts_params"]])

    # parallel execution
    start = time.time()
    workers = config_dict['workers']
    if bfast:
        logger.info("Running BFAST")
        results = run_in_parallel(bfast_monitor, bfast_args, workers, 'processes')
        d = {i: result for i, result in enumerate(results)}
        bfast_df = pd.DataFrame.from_dict(d, orient="index")
        bfast_df.columns = ["bfast_change_date", "bfast_magnitude", "bfast_means", pid]
        df = pd.merge(df, bfast_df, on=pid)
        timer(start, "BFAST done")

    if cusum:
        logger.info("Running Cusum")
        results = run_in_parallel(cusum_deforest, cusum_args, workers, 'processes')
        d = {i: result for i, result in enumerate(results)}
        cusum_df = pd.DataFrame.from_dict(d, orient="index")
        cusum_df.columns = ["cusum_change_date", "cusum_confidence", "cusum_magnitude", pid]
        df = pd.merge(df, cusum_df, on=pid)
        timer(start, "Cusum done")

    if ts_metrics:
        logger.info("Running timescan")
        results = run_in_parallel(timescan, ts_args, workers)
        d = {i: result for i, result in enumerate(results)}
        tscan_df = pd.DataFrame.from_dict(d, orient="index")
        tscan_df.columns = ["ts_mean", "ts_sd", "ts_min", "ts_max", pid]
        df = pd.merge(df, tscan_df, on=pid)
        timer(start, "TS metrics done")

    if bs_slope:
        logger.info("Running slope")
        results = run_in_parallel(bootstrap_slope, bs_args, workers, 'processes')
        d = {i: result for i, result in enumerate(results)}
        bs_df = pd.DataFrame.from_dict(d, orient="index")
        bs_df.columns = ['bs_slope_mean', 'bs_slope_sd', 'bs_slope_max', 'bs_slope_min', pid]
        df = pd.merge(df, bs_df, on=pid)
        timer(start, "slope done")

    if jrc_nrt:
        logger.info("Running JRC")
        results = run_in_parallel(run_jrc_nrt, nrt_args, workers, 'processes')
        d = {i: result for i, result in enumerate(results)}
        nrt_df = pd.DataFrame.from_dict(d, orient="index")
        nrt_df.columns = [
            pid,
            "ewma_jrc_date", "ewma_jrc_change", "ewma_jrc_magnitude",
            "mosum_jrc_date", "mosum_jrc_change", "mosum_jrc_magnitude",
            "cusum_jrc_date", "cusum_jrc_change", "cusum_jrc_magnitude"
        ]
        df = pd.merge(df, nrt_df, on=pid)
        timer(start, "jrc done")
    # drop unnecessary columns
    cols_to_drop = ['dates_mon', 'ts_mon']
    df.drop([col for col in cols_to_drop if col in df], axis=1, inplace=True)
    return df
