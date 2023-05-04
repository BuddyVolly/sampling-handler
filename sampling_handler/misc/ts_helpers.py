from datetime import datetime as dt
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


def subset_ts(row, start, end, bands):
    """Helper function to extract only monitoring period"""

    if isinstance(start, str):
        start = dt.strptime(start, "%Y-%m-%d")

    if isinstance(end, str):
        end = dt.strptime(end, "%Y-%m-%d")

    # create index for monitoring period
    idx = (row.dates > start) & (row.dates < end)

    # subset dates
    dates = row.dates[idx]

    # subset ts data
    ts = {}
    for band in bands:
        ts[band] = np.array(row.ts[band])[idx].tolist()

    # get new image length
    images = len(dates)

    return dates, ts, images


def rolling_mean(dates, ts, bands, interval="60d"):

    d = {}
    for band in bands:
        tmp_df = pd.DataFrame(
            data=ts[band], index=pd.DatetimeIndex(dates), columns=["ts"]
        )
        d[band] = tmp_df.rolling(interval).mean().ts.tolist()

    return d


def smooth_ts(df, bands):

    df["ts"] = df.apply(lambda x: rolling_mean(x.dates, x.ts, bands), axis=1)
    return df


def outlier_removal(dates, ts, bands, ts_band):

    # get time-series band to remove outliers
    out_ts = np.array(ts[ts_band]).astype(float)
    z_score = np.abs(stats.zscore(out_ts, axis=0))
    out_ts[z_score > 3] = np.nan

    # replace in the ts dict
    ts[ts_band] = out_ts

    # create dataframe
    tmp_df = pd.DataFrame(
        data=ts, index=pd.DatetimeIndex(dates), columns=bands
    )

    # drop nans, applied to all columns
    tmp_df = tmp_df.dropna()

    # aggreagte band values to dict to send back to main df
    d = {}
    for band in bands:
        d[band] = tmp_df[band].tolist()

    return tmp_df.index, d


def remove_outliers(df, bands, ts_band):

    df[['dates', 'ts']] = df.apply(
        lambda x: outlier_removal(x.dates, x.ts, bands, ts_band),
        axis=1,
        result_type="expand"
    )
    return df


def plot_timeseries(pickle_file, point_id, point_id_name="point_id"):

    df = pd.read_pickle(pickle_file)
    dates = df[df[point_id_name] == point_id].dates.values[0]
    ts = np.array(df[df[point_id_name] == point_id].ts.values[0])

    sns.scatterplot(x=dates, y=ts)
