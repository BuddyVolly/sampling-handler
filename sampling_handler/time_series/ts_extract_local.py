import json
import logging
import time
from pathlib import Path

import ee
import geopandas as gpd
from ee_preproc import landsat_collection

from ..sampling import sample_design
from ..misc.ee_helpers import delete_sub_folder
from ..misc import py_helpers
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)





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
    lsat_params = ts_params["lsat_params"]

    # create a polygon surrounding all points
    aoi = gpd.GeoDataFrame(
        index=[0],
        crs=input_grid.crs,
        geometry=[input_grid.geometry.unary_union.convex_hull]
    ).to_crs("epsg:4326")

    # run dggrid routine at given resolution scale
    processing_chunks = sample_design.hexagonal_grid(
        aoi, dggrid_res, outcrs="epsg:4326", grid_only=True
    )

    # create image collection (not being changed)
    aoi_fc = ee.FeatureCollection(json.loads(aoi.to_json()))
    if sat == "Landsat":
        sat_coll = landsat_collection(
            start_hist, end_mon, aoi_fc, **lsat_params
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
        if 0 < len(point_subset) < ts_params['max_points_per_chunk']:
            # debug line
            # return_code = extract_time_series(
            #   sat_coll, point_subset, config_dict, identifier, export_folder
            # )
            args_to_process.append((
                sat_coll, point_subset, config_dict, identifier, export_folder
            ))

    # run in parallel
    if args_to_process:
        return_code = py_helpers.run_in_parallel(
            extract_time_series, args_to_process, config_dict
        )
    else:
        return_code = 1

    try:
        delete_sub_folder(f"tmp_esbae_{gmt}")
    except ee.EEException:
        pass

    if any(return_code) != 0:
        logger.warning(
            "Not fully processed. Will retry at a higher aggregation level."
        )

    return return_code
