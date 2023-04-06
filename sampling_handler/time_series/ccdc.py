from datetime import datetime as dt
import logging

import ee
import requests
import numpy as np
import pandas as pd
from retrying import retry

from ..misc import py_helpers
from ..misc.settings import setup_logger


# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


def get_segments(ccdcAst, mask_1d):
    """ """
    bands_2d = ccdcAst.select(".*_coefs").bandNames()
    bands_1d = ccdcAst.bandNames().removeAll(bands_2d)
    segment_1d = ccdcAst.select(bands_1d).arrayMask(mask_1d)
    mask_2d = mask_1d.arrayReshape(ee.Image(ee.Array([-1, 1])), 2)
    segment_2d = ccdcAst.select(bands_2d).arrayMask(mask_2d)
    return segment_1d.addBands(segment_2d)


def get_segment(ccdcAst, mask_1d):
    """ """
    bands_2d = ccdcAst.select(".*_coefs").bandNames()
    bands_1d = ccdcAst.bandNames().removeAll(bands_2d)
    segment_1d = ccdcAst.select(bands_1d).arrayMask(mask_1d).arrayGet([0])
    mask_2d = mask_1d.arrayReshape(ee.Image(ee.Array([-1, 1])), 2)
    segment_2d = ccdcAst.select(bands_2d).arrayMask(mask_2d).arrayProject([1])
    return segment_1d.addBands(segment_2d)


def transform_date(date):
    """ """
    date = pd.to_datetime(dt.fromtimestamp(date / 1000.0))
    dates_float = date.year + np.round(date.dayofyear / 365, 3)
    dates_float = 0 if dates_float == "1970.003" else dates_float
    return dates_float


@retry(stop_max_attempt_number=3, wait_random_min=5000, wait_random_max=10000)
def run_ccdc(df, samples, config_dict):

    ccdc_params = config_dict["ccdc_params"]
    ts_band = config_dict["ts_params"]["ts_band"]
    bands = config_dict["ts_params"]["bands"]
    point_id_name = config_dict["ts_params"]["point_id"]

    start_calibration = config_dict["ts_params"]["start_calibration"]
    start_monitor = config_dict["ts_params"]["start_monitor"]
    end_monitor = config_dict["ts_params"]["end_monitor"]

    args_list, iColl, points = [], None, []
    for i, row in df.iterrows():

        # get dates
        dates = ee.List([dt.strftime(date, "%Y-%m-%d") for date in row.dates])

        # transform ts dict into way to ingest into imagery
        ts = []
        for j in range(len(row.dates)):
            ts.append([v[j] for v in row.ts.values()])

        # gather points for feature collection to reduce on
        geom = ee.Feature(
            row.geometry.__geo_interface__
        ).set(point_id_name, row[point_id_name])
        squared = geom.geometry().buffer(100, 10).bounds()
        points.append(row[point_id_name])

        # merge dates with ts data
        ts = ee.List(ts).zip(dates)

        def zip_to_image(element):

            values = ee.List(element).get(0)
            date = ee.List(element).get(1)

            return ee.Image(
                ee.Image.constant(values)
                .rename(list(row.ts.keys()))
                .clip(squared)
                .set("system:time_start",
                     ee.Date.parse("YYYY-MM-dd", date).millis())
                .toFloat()
            )

        # create the image collection
        tsee = ee.ImageCollection(ts.map(zip_to_image))
        iColl = iColl.cat(tsee.toList(tsee.size())) if iColl else tsee.toList(
            tsee.size())

    points = samples.filter(ee.Filter.inList(point_id_name, points))
    # print(points.getInfo())
    iColl = ee.ImageCollection.fromImages(iColl)

    # add collection and remove run from parameter dict
    params = ccdc_params.copy()
    params['collection'] = iColl
    params.pop("run", None)

    # run ccdc
    ccdc = ee.Algorithms.TemporalSegmentation.Ccdc(**params)

    # extract info
    # create array of start of monitoring in shape of tEnd
    tEnd = ccdc.select("tEnd")
    mon_date_array_start = tEnd.multiply(0).add(
        ee.Date(start_monitor).millis())
    mon_date_array_end = tEnd.multiply(0).add(ee.Date(end_monitor).millis())

    # create the date mask
    date_mask = tEnd.gte(mon_date_array_start).And(
        tEnd.lte(mon_date_array_end))

    # use date mask to mask all of ccdc
    monitoring_ccdc = get_segments(ccdc, date_mask)

    # mask for highest magnitude in monitoring period
    magnitude = monitoring_ccdc.select(f'{ts_band}_magnitude')
    max_abs_magnitude = (
        magnitude.abs()
        .arrayReduce(ee.Reducer.max(), [0])
        .arrayGet([0])
        .rename("max_abs_magnitude")
    )

    mask = magnitude.abs().eq(max_abs_magnitude)
    segment = get_segment(monitoring_ccdc, mask)
    magnitude = ee.Image(
        segment.select([f'{ts_band}_magnitude', "tBreak", "tEnd"]))

    def pixel_value_nan(feature):
        pixel_value = ee.List(
            [feature.get(f'{ts_band}_magnitude'), -9999]).reduce(
            ee.Reducer.firstNonNull()
        )
        return feature.set({f'{ts_band}_magnitude': pixel_value})

    sampled_points = magnitude.reduceRegions(**{
        "reducer": ee.Reducer.first(),
        "collection": points,
        "scale": 100,
        "tileScale": 4,
    }).map(pixel_value_nan).select(
        propertySelectors=[
            point_id_name, f'{ts_band}_magnitude', "tBreak", "tEnd"
        ],
        retainGeometry=False
    )

    url = sampled_points.getDownloadUrl("geojson")

    # Handle downloading the actual pixels.
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()

    return pd.DataFrame(
        [feature['properties'] for feature in r.json()['features']]
    )


def get_ccdc(df, samples, config_dict):

    logger.info('Running CCDC')
    point_id_name = config_dict['ts_params']['point_id']
    ts_band = config_dict['ts_params']['ts_band']

    ccdc_args = []
    for i in range(0, len(df), 10):
        ccdc_args.append([df.iloc[i:i+10], samples, config_dict])

    result = py_helpers.run_in_parallel(run_ccdc, ccdc_args, workers=20)
    eedf = pd.concat(result)
    eedf["ccdc_change_date"] = eedf["tBreak"].apply(lambda x: transform_date(x))
    eedf["ccdc_magnitude"] = eedf[f"{ts_band}_magnitude"]
    return pd.merge(
            df,
            eedf[["ccdc_change_date", "ccdc_magnitude", point_id_name]],
            on=point_id_name,
        )
