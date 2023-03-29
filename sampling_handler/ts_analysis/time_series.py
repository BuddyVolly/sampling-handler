import json
import logging
import time
import concurrent.futures
from pathlib import Path

import ee
import geojson
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from retrying import retry

from ..sampling import grid
from ee_preproc import landsat_collection
from ..misc.ee_helpers import (
    export_to_ee, delete_sub_folder, processing_grid,
    _ee_export_table, cleanup_tmp_esbae
)
from ..misc.py_helpers import timer
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


def extract(input_grid, config_dict, start_res=8, end_res=13):

    # create output directory
    out_dir = Path(config_dict["work_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # read config file i existing, so we can compare to new one
    config_file = out_dir.joinpath("config.json")
    if config_file.exists():
        with open(config_file) as f:
            old_config_dict = json.load(f)

        # create a copy of the new config for comparison
        new_config_dict = config_dict.copy()
        # define keys that can be changed
        keys_list = [
            "work_dir", "workers", "max_points_per_chunk", "grid_size_levels"
        ]
        # remove those keys from both configs
        [new_config_dict.pop(key) for key in keys_list]
        [old_config_dict.pop(key) for key in keys_list]

        if not new_config_dict == old_config_dict:
            config_change = input(
                "Your processing parameters in your config file changed. "
                "If you continue, all of your already processed files will be "
                "deleted. Are you sure you want to continue? (yes/no)"
            )
            if config_change == "no":
                return
            elif config_change == "yes":
                logger.info("Cleaning up results folder.")
                sat = config_dict["ts_params"]["satellite"]
                final_dir = Path(config_dict["work_dir"]).joinpath(sat)
                [file.unlink() for file in final_dir.glob("*geojson")]
            else:
                raise ValueError(
                    "Answer is not recognized, should be \"yes\" or \"no\""
                )

    with open(config_file, "w") as f:
        json.dump(config_dict, f)

    if isinstance(input_grid, ee.FeatureCollection):
        cascaded_extraction_ee(input_grid, config_file)

    elif isinstance(input_grid, gpd.geodataframe.GeoDataFrame):
        cascaded_extraction(input_grid, config_file, start_res, end_res)


def cascaded_extraction_ee(input_grid, config_file):

    # read config file
    with open(config_file) as f:
        config_dict = json.load(f)

    chunk_sizes = config_dict["grid_size_levels"]
    # get number of points to process
    size = input_grid.size().getInfo()

    if size > 25000:
        subsets = int(np.ceil(size/25000))
        logger.info(
            f"The number of points exceeds 25000. "
            f"Processing will be split into {subsets} subsets."
        )
    # sort by point id
    input_grid = input_grid.sort(config_dict['ts_params']['point_id'])

    # if already processed files are in the tmp_folder
    logger.info("Checking for already processed files.")
    input_grid = _get_missing_points(input_grid, config_dict)

    # if collection is greater than that, we iterate over chunks
    subs = []
    for i in range(0, size, 25000):

        if size > 25000:
            logger.info(f"------------------------------------------------")
            logger.info(f"Processing subset {int(i/25000+1)}/{subsets}")
            logger.info(f"------------------------------------------------")

        sub = ee.FeatureCollection(input_grid.toList(25000, i))
        for chunk_size in chunk_sizes:

            _ = _parallel_extract_ee(
                sub, chunk_size, config_dict, subset=int(i/25000+1)
            )
            logger.info(
                "Checking for points not processed at the current "
                "aggregation level."
            )
            sub = _get_missing_points(sub, config_dict)
            if sub == 'completed':
                break

        # if the final run did not finished with all
        if sub != 'completed':
            subs.append(sub.size().getInfo())

    if subs:
        points_left = np.sum(subs)
        log = Path(LOGFILE)
        logger.error(
            f"Extraction of time-series has failed or is incomplete with "
            f"{points_left} not being processed. You can check "
            f"the logfile {log.name} within your results directory."
        )
        log.rename(Path(config_dict["work_dir"].joinpath(log.name)))
    else:
        logger.info(
            "Extraction of time-series has been finished for all points."
        )

    # cleanup all tmp_esbae folders
    logger.info(
        "Cleaning up temporary Earth Engine assets created "
        "during the processing."
    )
    cleanup_tmp_esbae()


def cascaded_extraction(input_grid, config_file, start_res, end_res=None):

    # read config file
    with open(config_file) as f:
        config_dict = json.load(f)

    # get number of points to process
    nr_all_points = len(input_grid)

    # check for points that might have been already processed
    input_grid = _get_missing_points(input_grid, config_dict)

    # set end resolution
    if not end_res:
        end_res = start_res + 5

    for resolution in range(start_res, end_res):

        if len(input_grid) == 0:
            logger.info(f"All of the points have been processed. ")
            return
        elif len(input_grid) != nr_all_points:
            logger.info(
                f"Some points have already been processed. "
                f"There are {len(input_grid)} points left to process."
            )
        logger.info(
            f"Aggregating points at dggrid resolution {resolution} "
            f"for parallel processing."
        )
        # run parallel extract
        _ = _parallel_extract(input_grid, resolution, config_file)

        # check if everything has been processed
        input_grid = _get_missing_points(input_grid, config_dict)


def _get_missing_points(input_grid, config_dict):

    # get point id name
    point_id = config_dict["ts_params"]["point_id"]
    sat = config_dict["ts_params"]["satellite"]
    final_dir = Path(config_dict["work_dir"]).joinpath(sat)

    if len(list(final_dir.glob('*geojson'))) > 0:

        tmp_dfs = []
        for file in final_dir.glob('*geojson'):
            with open(file, 'r') as outfile:
                d = gpd.GeoDataFrame.from_features(
                    geojson.loads(geojson.load(outfile))).drop(
                    ['dates', 'ts', 'images', 'geometry'], axis=1)
                tmp_dfs.append(d)

        tmp_df = pd.concat(tmp_dfs).drop_duplicates(point_id)

        if isinstance(input_grid, gpd.geodataframe.GeoDataFrame):

            logger.warning("Points within those file are discarded.")
            # update input_grid and keep only non processed points
            input_grid = input_grid[
                ~input_grid[point_id].isin(tmp_df[point_id].to_list())
            ]

        elif isinstance(input_grid, ee.FeatureCollection):

            # create a unique id
            gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())
            # get already processed points
            processed_points = ee.List(tmp_df[point_id].to_list())
            # filter the input grid to non-processed points
            input_grid = input_grid.filter(
                ee.Filter.inList('point_id', processed_points).Not()
            )

            if input_grid.size().getInfo() > 0:
                logger.warning(
                    "Points within those files are discarded and a temporary "
                    "FeatureCollection including the non-processed points "
                    "will be created on Earth Engine"
                )
                # export the filtered grid to a new feature collection
                _, input_grid = _ee_export_table(
                    ee_fc=input_grid,
                    description=f"tmp_esbae_table_export_{gmt}",
                    asset_id=f"tmp_esbae_table_{gmt}",
                    sub_folder=f"tmp_esbae_{gmt}"
                )

                input_grid = ee.FeatureCollection(input_grid)
            else:
                logger.info(
                    "This batch of points has successfully been processed."
                )
                input_grid = 'completed'

    return input_grid


def _parallel_extract_ee(points_fc, chunk_size, config_dict, subset):

    # create a unique id (gmt time)
    gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())

    # create an outfile and skip routine if it already exists
    sat = config_dict["ts_params"]["satellite"]
    final_dir = Path(config_dict["work_dir"]).joinpath(sat)
    final_dir.mkdir(parents=True, exist_ok=True)

    # extract ts params
    ts_params = config_dict["ts_params"]
    sat = ts_params["satellite"]
    start_hist = ts_params["start_calibration"]
    end_mon = ts_params["end_monitor"]

    # automatically get aoi from extent of points and upload it as an asset
    logger.info(
        f"Create AOI from points and upload as temporary EE asset "
        f"inside tmp_esbae_{gmt}."
    )
    aoi_fc = ee.FeatureCollection(points_fc.geometry().convexHull(100))
    _, aoi_fc = _ee_export_table(
        ee_fc=aoi_fc,
        description=f"tmp_esbae_aoi_{gmt}",
        asset_id=f"tmp_esbae_aoi_{gmt}",
        sub_folder=f"tmp_esbae_{gmt}"
    )
    aoi_fc = ee.FeatureCollection(aoi_fc)

    if sat == "Landsat":
        sat_coll = landsat_collection(
            start_hist, end_mon, aoi_fc, **config_dict["lsat_params"]
        )
    logger.info(
        f"Creating processing chunks of {chunk_size} degrees for "
        f"parallel extraction."
    )
    chunks_fc = processing_grid(aoi_fc, chunk_size)

    # filter for chunks with valid number of points
    to_proc = chunks_fc.map(
        lambda x: ee.Feature(x).set(
            'intersects', points_fc.filterBounds(x.geometry()).size()
        )
    )
    to_proc = to_proc.filter(ee.Filter.And(
        ee.Filter.gt('intersects', 0),
        ee.Filter.lt('intersects', config_dict['max_points_per_chunk'])
    ))
    
    processing_chunks = ee.FeatureCollection(to_proc).aggregate_array(
        '.geo').getInfo()

    # create args_list for each grid cell
    logger.info(
        f"Preparing the parallel extraction over a total of "
        f"{len(processing_chunks)} chunks. This may take a while..."
    )
    args_to_process = []
    for i, chunk in enumerate(processing_chunks):

        # create an identifier
        identifier = f"{subset}_{i}_{chunk_size}_{gmt}"
        # this could go to the extract time-series, so it runs in parallel
        # therefore we only put the chunk geom to the extract time -series
        # (might give problems though)
        # mazbe we could map? as all other arguments are the same
        cell_fc = points_fc.filterBounds(chunk)

        # return_code = extract_time_series(
        # sat_coll, cell_fc, config_dict, identifier
        # )
        args_to_process.append((
            sat_coll, cell_fc, config_dict, identifier
        ))

    logger.info(
        f"Starting the parallel extraction routine."
    )
    # run in parallel
    if args_to_process:
        return_code = _run_in_threads(
            extract_time_series, args_to_process, config_dict
        )
    else:
        return_code = 1

    try:
        delete_sub_folder(f"tmp_esbae_{gmt}")
    except ee.EEException:
        pass

    if return_code != 0:
        logger.warning(
            "Not fully processed. Will retry at a higher aggregation level."
        )

    return return_code


def _parallel_extract(input_grid, dggrid_res, config_file):

    # create a unique string
    gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())
    export_folder = f"tmp_esbae_{gmt}"

    # read config file
    with open(config_file) as f:
        config_dict = json.load(f)

    # create an outfile and skip routine if it already exists
    sat = config_dict["ts_params"]["satellite"]
    final_dir = Path(config_dict["work_dir"]).joinpath(sat)
    final_dir.mkdir(parents=True, exist_ok=True)

    # extract ts params
    ts_params = config_dict["ts_params"]
    sat = ts_params["satellite"]
    start_hist = ts_params["start_calibration"]
    end_mon = ts_params["end_monitor"]

    # create a polygon surrounding all points
    aoi = gpd.GeoDataFrame(
        index=[0],
        crs=input_grid.crs,
        geometry=[input_grid.geometry.unary_union.convex_hull]
    ).to_crs("epsg:4326")

    # run dggrid routine at given resolution scale
    processing_chunks = grid.hexagonal_grid(
        aoi, dggrid_res, outcrs="epsg:4326", grid_only=True
    )

    # create image collection (not being changed)
    aoi_fc = ee.FeatureCollection(json.loads(aoi.to_json()))
    if sat == "Landsat":
        sat_coll = landsat_collection(
            start_hist, end_mon, aoi_fc, **config_dict["lsat_params"]
        )

    logger.info("Preparing processing grid for parallel execution in chunks.")
    # create an empty list of args and fill list with
    args_to_process, processed_dfs = [], []
    for i, row in processing_chunks.iterrows():

        identifier = f"0_{i}_{dggrid_res}_{gmt}"
        # get intersecting points for processing chunk
        point_subset = input_grid[
            input_grid.to_crs("epsg:4326").intersects(row.geometry)
        ]

        # add to list is adheres to amx points per chunk
        if 0 < len(point_subset) < config_dict['max_points_per_chunk']:
            # debug line
            # return_code = extract_time_series(
            #   sat_coll, point_subset, config_dict, identifier, export_folder
            # )
            args_to_process.append((
                sat_coll, point_subset, config_dict, identifier, export_folder
            ))

    # run in parallel
    if args_to_process:
        return_code = _run_in_threads(
            extract_time_series, args_to_process, config_dict
        )
    else:
        return_code = 1

    try:
        delete_sub_folder(f"tmp_esbae_{gmt}")
    except ee.EEException:
        pass

    if return_code != 0:
        logger.warning(
            "Not fully processed. Will retry at a higher aggregation level."
        )

    return return_code


def _run_in_threads(func, arg_list, config_dict):

    max_workers = config_dict["workers"]
    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:

        # submit tasks
        futures = [executor.submit(func, *args) for args in arg_list]

        # gather results
        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]

    return results


@retry(stop_max_attempt_number=3, wait_random_min=2000, wait_random_max=5000)
def extract_time_series(image_collection, points, config_dict, identifier=None, export_folder=None):

    try:
        if isinstance(points, ee.FeatureCollection):
            nr_points = points.size().getInfo()
        elif isinstance(points, gpd.geodataframe.GeoDataFrame):
            nr_points = len(points)

        logger.info(
            f"Extracting {nr_points} points for chunk nr "
            f"{identifier.split('_')[1]} at resolution {identifier.split('_')[2]}"
        )

        # get start time
        start = time.time()

        # create an outfile and skip routine if it already exists
        sat = config_dict["ts_params"]["satellite"]
        final_dir = Path(config_dict["work_dir"]).joinpath(sat)
        final_dir.mkdir(parents=True, exist_ok=True)
        outfile = list(
            final_dir.glob(f"{'_'.join(identifier.split('_')[:2])}.geojson")
        )
        if outfile:
            return 0
        else:
            outfile = final_dir.joinpath(f"{identifier}.geojson")

        # get ts parameters from configuration file
        ts_params = config_dict["ts_params"]
        bands = ts_params["bands"]
        ee_bands = ee.List(ts_params["bands"])
        point_id_name = ts_params["point_id"]
        scale = ts_params["scale"]

        # create a fc from the points
        # geo_json = points[['point_id', 'geometry']].to_crs('epsg:4326').to_json()
        if not isinstance(points, ee.FeatureCollection):

            points_fc = f"{identifier}"
            asset_id = export_to_ee(
                gdf=points[['point_id', 'geometry']],
                asset_name=points_fc,
                ee_sub_folder=export_folder
            )

            if not asset_id:
                return None

            # update points fc name
            points_fc = ee.FeatureCollection(asset_id)
        else:
            points_fc = points

        #if bounds_reduce:
        #    points_fc = points_fc.map(
        #        lambda x: ee.Feature(
        #            x.geometry().buffer(scale/2).bounds()
        #        ).copyProperties(x, x.propertyNames())
        #    )

        # mask lsat collection for grid cell
        cell = points_fc.geometry().convexHull(100)
        masked_coll = image_collection.filterBounds(cell)
        reducer = (
            ee.Reducer.first().setOutputs(bands) if len(bands) == 1 else ee.Reducer.first()
        )

        # mapping function to extract time-series from each image
        def map_over_img_coll(image):

            def pixel_value_nan(feature):

                pixel_values = ee_bands.map(
                    lambda band: ee.List([feature.get(band), -9999]).reduce(
                        ee.Reducer.firstNonNull()
                    )
                )
                properties = ee.Dictionary.fromLists(ee_bands, pixel_values)
                return feature.set(properties.combine({"imageID": image.id()}))

            return image.reduceRegions(
                #collection=points_fc.filter(ee.Filter.isContained(".geo", image.geometry()))
                collection=points_fc.filterBounds(image.geometry()),
                reducer=reducer,
                scale=scale   # 30 for bounds_reduce
            ).map(pixel_value_nan)

        # apply mapping function over landsat collection
        cell_fc = masked_coll.map(map_over_img_coll).flatten().filter(
            ee.Filter.neq(bands[0], -9999)
            )

        # and get the url of the data
        url = cell_fc.getDownloadUrl("geojson")

        # Handle downloading the actual pixels.
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise r.raise_for_status()

        # write the FC to a geodataframe
        try:
            point_gdf = gpd.GeoDataFrame.from_features(r.json())
        except json.JSONDecodeError as e:
            logger.warning(
                f"Point extraction from Earth Engine for chunk nr "
                f"{identifier.split('_')[1]} failed at resolution "
                f"{identifier.split('_')[2]}"
                "Will try again."
            )
            return None

        if len(point_gdf) > 0:
            structured_df = structure_ts_data(point_gdf, point_id_name, bands)
            # dump to geojson (we cannot use geopandas, as it contains dicts)
            with open(outfile, "w") as f:
                geojson.dump(structured_df.to_json(), f)

        timer(
            start, (
                f"{nr_points} points for chunk nr "
                f"{identifier.split('_')[1]} at resolution "
                f"{identifier.split('_')[2]} extracted in:"
            )
        )
    except Exception as e:
        logger.debug(str(e))
        # add identifier here
        raise e

    return 0


def structure_ts_data(df, point_id_name, bands):

    df.index = pd.DatetimeIndex(pd.to_datetime(df.imageID.apply(
            lambda x: x.split("_")[-1]), format="%Y%m%d"
        )
    )
    df["dates"] = df.imageID.apply(lambda x: x.split("_")[-1])

    # loop over point_ids and run cusum
    d = {}
    for i, point in enumerate(df[point_id_name].unique()):

        # read only orws of points and sort by date
        sub = df[df[point_id_name] == point].sort_index()

        # LANDSAT ONLY ###########
        sub["pathrow"] = sub.imageID.apply(lambda x: x.split("_")[-2])

        # if more than one path row combination covers the point,
        # we select only the one with the most images
        if len(sub.pathrow.unique()) > 1:
            # set an initil length
            length = -1
            # loop through pathrw combinations
            for pathrow in sub.pathrow.unique():
                # check length
                l = len(sub[sub.pathrow == pathrow])
                # compare ot previous length, and if higher reset
                # pathrow and length variable
                if l > length:
                    pr = pathrow
                    length = l
            # finally filter sub df for pathrow with most images
            sub = sub[sub.pathrow == pr]
        # LANDSAT ONLY ###########

        # still duplicates may appear between l9 and l8 that
        # would make bfast crash, so we drop
        sub = sub[~sub.index.duplicated(keep="first")]

        # fill ts dictionary
        ts_dict = {}
        for band in bands:
            ts_dict.update({band: sub[band].tolist()})

        # write everything to a dict
        d[i] = {
            point_id_name: point,
            "dates": sub["dates"].tolist(),
            "ts": ts_dict,
            "images": len(sub),
            "geometry": sub.geometry.head(1).values[0],
        }

    # turn the dict into a geodataframe and return
    return gpd.GeoDataFrame(
        pd.DataFrame.from_dict(d, orient="index")
    ).set_geometry("geometry")
