import datetime
import pandas as pd
import xarray as xr
import numpy as np

from nrt.monitor.ewma import EWMA
from nrt.monitor.cusum import CuSum
from nrt.monitor.mosum import MoSum

# from nrt.monitor.ccdc import CCDC


def run_jrc_nrt(dates, ts, pid, ts_params):

    # extract point id column name
    start_hist = ts_params["start_calibration"]
    start_mon = ts_params["start_monitor"]
    end_mon = ts_params["end_monitor"]
    point_id_name = ts_params["point_id"]

    try:
        # aggregate to arrays for multiindexing
        arrays = [dates, [1 for i in range(len(ts))], [1 for i in range(len(ts))]]
        nrt_df = pd.DataFrame(
            {"data": ts, point_id_name: pid}, index=arrays
        ).rename_axis(["time", "x", "y"])

        da = xr.Dataset.from_dataframe(nrt_df)
        da["time"] = da["time"].astype("datetime64[ns]")
        da["data"] = da.data.astype("float32")

        # slice for calibration and monitoring
        history = da.data.sel(time=slice(start_hist, start_mon))
        monitoring = da.data.sel(time=slice(start_mon, end_mon))

        # Instantiate monitoring class and fit stable history
        EwmaMonitor = EWMA(trend=False)
        EwmaMonitor.fit(dataarray=history, method="OLS",
                        screen_outliers="Shewhart")

        # CcdcMonitor = CCDC(trend=True)
        # CcdcMonitor.fit(dataarray=history, screen_outliers='Shewhart')

        CuSumMonitor = CuSum(trend=False)
        CuSumMonitor.fit(
            dataarray=history, trend=False, method="ROC",
            screen_outliers="Shewhart"
        )

        MoSumMonitor = MoSum(trend=False)
        MoSumMonitor.fit(dataarray=history, screen_outliers="Shewhart")

        # Monitor new observations
        for array, date in zip(
                monitoring.values,
                monitoring.time.values.astype("M8[s]").astype(datetime.datetime),
        ):
            EwmaMonitor.monitor(array=array, date=date)
            # CcdcMonitor.monitor(array=array, date=date)
            CuSumMonitor.monitor(array=array, date=date)
            MoSumMonitor.monitor(array=array, date=date)

        return (
            pid,
            EwmaMonitor.detection_date.flatten(),
            np.where(EwmaMonitor.detection_date.flatten() > 0, 1, 0),
            EwmaMonitor.process.flatten(),
            MoSumMonitor.detection_date.flatten(),
            np.where(MoSumMonitor.detection_date.flatten() > 0, 1, 0),
            MoSumMonitor.process.flatten(),
            CuSumMonitor.detection_date.flatten(),
            np.where(CuSumMonitor.detection_date.flatten() > 0, 1, 0),
            CuSumMonitor.process.flatten()
        )

    except:
        return pid, 0, 0, 0, 0, 0, 0, 0, 0, 0
