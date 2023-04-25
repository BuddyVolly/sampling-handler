import json
import logging
import time
from pathlib import Path
import shutil

import ee
import geojson
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from retrying import retry

from ee_preproc import landsat_collection
from ..misc.ee_helpers import (
    export_to_ee, delete_sub_folder, processing_grid,
    _ee_export_table, cleanup_tmp_esbae
)

from ..esbae import Esbae
from ..misc import py_helpers, config
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class TimeSeriesExtraction(Esbae):

    def __init__(
            self,
            project_name,
            ts_start,
            ts_end,
            satellite,
            scale,
            bounds_reduce,
            bands,
            aoi=None
    ):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('03_Timeseries_Extract'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.start = ts_start
        self.end = ts_end
        self.satellite = satellite
        self.scale = scale
        self.bounds_reduce = bounds_reduce
        self.bands = bands

        # get params from befre steps (or default values)
        conf = self.config_dict
        self.pid = conf['design_params']['pid']
        self.sample_asset = conf['design_params']['ee_samples_fc']

        # load default params
        self.lsat_params = conf['ts_params']['lsat_params']
        self.workers = conf['ts_params']['ee_workers']
        self.max_points_per_chunk = conf['ts_params']['max_points_per_chunk']
        self.grid_size_levels = conf['ts_params']['grid_size_levels']

    def get_time_series_data(self):

        # update config_dict
        # apply the basic configuration set in the cell above
        self.config_dict['ts_params']['outdir'] = self.out_dir
        self.config_dict['ts_params']['satellite'] = self.satellite
        self.config_dict['ts_params']['ts_start'] = self.start
        self.config_dict['ts_params']['ts_end'] = self.end
        self.config_dict['ts_params']['scale'] = self.scale
        self.config_dict['ts_params']['bounds_reduce'] = self.bounds_reduce
        self.config_dict['ts_params']['ee_workers'] = self.workers
        self.config_dict['ts_params']['max_points_per_chunk'] = self.max_points_per_chunk
        self.config_dict['ts_params']['grid_size_levels'] = self.grid_size_levels
        self.config_dict['design_params']['pid'] = self.pid
        self.config_dict['design_params']['ee_samples_fc'] = self.sample_asset

        # check if relevant value changed, and clean up out folder in case, to keep output data consistent
        check_config_changed(self.config_dict)

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        # run extract routine
        extract(ee.FeatureCollection(self.sample_asset), self.config_dict)

    def check_if_completed(self):

        logger.info('Verifying ')
        actual_size = ee.FeatureCollection(self.sample_asset).size().getInfo()
        dfs = []
        for file in Path(f'{self.out_dir}/{self.satellite}').glob('*geojson'):
            with open(file) as f:
                dfs.append(
                    gpd.GeoDataFrame.from_features(
                        geojson.loads(geojson.load(f))
                    ).drop(['ts', 'dates', 'geometry'], axis=1)
                )
            df = pd.concat(dfs)

        if not dfs:
            logger.info('No time-series data has been extracted yet.')
            return

        if actual_size > len(df[self.pid].unique()):
            missing = actual_size - len(df[self.pid].unique())
            logger.info(
                'Time-series data has been extracted partially. '
                f'{missing} points missing of a total of {actual_size}'
            )
            return

        if actual_size == len(df[self.pid].unique()):
            logger.info(
                'Time-series data has been extracted completely. '
                'Time to move on with the data augmentation notebook.'
            )
        return


def _structure_ts_data(df, point_id_name, bands):

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
        ts_params = config_dict["ts_params"]
        sat = ts_params["satellite"]
        final_dir = Path(ts_params["outdir"]).joinpath(sat)
        final_dir.mkdir(parents=True, exist_ok=True)
        outfile = list(
            final_dir.glob(f"{'_'.join(identifier.split('_')[:2])}.geojson")
        )
        if outfile:
            return 0
        else:
            outfile = final_dir.joinpath(f"{identifier}.geojson")

        # get ts parameters from configuration file
        bands = ts_params['lsat_params']['bands']
        ee_bands = ee.List(bands)
        scale = ts_params['scale']
        point_id_name = config_dict['design_params']['pid']
        bounds_reduce = ts_params['bounds_reduce']

        # features to extract
        features = bands.copy()
        features.extend(['imageID', point_id_name])

        # create a fc from the points
        # geo_json = points[['point_id', 'geometry']].to_crs('epsg:4326').to_json()
        if not isinstance(points, ee.FeatureCollection):

            points_fc = f"{identifier}"
            asset_id = export_to_ee(
                gdf=points[[point_id_name, 'geometry']],
                asset_name=points_fc,
                ee_sub_folder=export_folder
            )

            if not asset_id:
                return None

            # update points fc name
            points_fc = ee.FeatureCollection(asset_id)
        else:
            points_fc = points

        if bounds_reduce:
            # we create a bounding buffer around the point for later reduce regions
            points_fc = points_fc.map(
                lambda x: ee.Feature(
                    x.geometry().buffer(scale/2).bounds()
                ).copyProperties(x, x.propertyNames())
            )

        # mask lsat collection for grid cell
        cell = points_fc.geometry().convexHull(100)
        masked_coll = image_collection.filterBounds(cell)

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

            if bounds_reduce:
                reducer = (
                    ee.Reducer.mean().setOutputs(bands) if len(bands) == 1 else ee.Reducer.mean()
                )
                sampled_ts = ee.FeatureCollection(
                    image.reduceRegions(
                        collection=points_fc.filter(ee.Filter.isContained(".geo", image.geometry())),
                        reducer=reducer,
                        scale=30
                    ))
                # TODO do we want the centroid? (this doesn't work yet)
                # .map(lambda feature: ee.Feature(feature.geometry().centroid(), feature.toDictionary()))

            else:
                reducer = (
                    ee.Reducer.first().setOutputs(bands) if len(bands) == 1 else ee.Reducer.first()
                )
                sampled_ts = ee.FeatureCollection(
                    image.reduceRegions(
                        collection=points_fc.filterBounds(image.geometry()),
                        reducer=reducer,
                        scale=scale
                    ))

            return sampled_ts.map(pixel_value_nan).select(
                    propertySelectors=features,
                    retainGeometry=True
            )

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
        except json.JSONDecodeError:
            logger.warning(
                f"Point extraction from Earth Engine for chunk nr "
                f"{identifier.split('_')[1]} failed at resolution "
                f"{identifier.split('_')[2]}"
                "Will try again."
            )
            return None

        if len(point_gdf) > 0:
            structured_df = _structure_ts_data(point_gdf, point_id_name, bands)
            # dump to geojson (we cannot use geopandas, as it contains dicts)
            with open(outfile, "w") as f:
                geojson.dump(structured_df.to_json(), f)

        py_helpers.timer(
            start, (
                f"{nr_points} points for chunk nr "
                f"{identifier.split('_')[1]} at resolution "
                f"{identifier.split('_')[2]} extracted in:"
            )
        )
    except Exception as e:
        logger.debug(str(e))
        # add identifier here
        return 1

    return 0


def _get_missing_points(input_grid, config_dict, subset=None, upload_sub=False):

    # get point id name
    pid = config_dict['design_params']['pid']
    sat = config_dict['ts_params']["satellite"]
    out_dir = Path(config_dict['ts_params']['outdir']).joinpath(sat)

    # create a unique id
    gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())

    if subset:
        files = list(out_dir.glob(f'{subset}*geojson'))
    else:
        files = list(out_dir.glob(f'*geojson'))

    if len(files) > 0:

        tmp_dfs = []
        for file in files:
            with open(file, 'r') as outfile:
                d = gpd.GeoDataFrame.from_features(
                    geojson.loads(geojson.load(outfile))).drop(
                    ['dates', 'ts', 'images', 'geometry'], axis=1)
                tmp_dfs.append(d)

        tmp_df = pd.concat(tmp_dfs).drop_duplicates(pid)
        logger.info(f'Found {len(tmp_df)} already processed points')

        if isinstance(input_grid, gpd.geodataframe.GeoDataFrame):

            logger.warning("Points within those file are discarded.")
            # update input_grid and keep only non processed points
            input_grid = input_grid[
                ~input_grid[pid].isin(tmp_df[pid].to_list())
            ]

        elif isinstance(input_grid, ee.FeatureCollection):

            # get already processed points
            processed_points = ee.List(tmp_df[pid].to_list())
            # filter the input grid to non-processed points
            input_grid = input_grid.filter(
                ee.Filter.inList(pid, processed_points).Not()
            )

            if input_grid.size().getInfo() > 0:
                logger.warning(
                    "These points are discarded and a temporary "
                    "FeatureCollection including the non-processed points "
                    "will be uploaded on Earth Engine"
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

    elif upload_sub:

        _, input_grid = _ee_export_table(
            ee_fc=input_grid,
            description=f"tmp_esbae_table_export_{gmt}",
            asset_id=f"tmp_esbae_table_{gmt}",
            sub_folder=f"tmp_esbae_{gmt}"
        )
        input_grid = ee.FeatureCollection(input_grid)

    return input_grid


def _parallel_extract_ee(points_fc, chunk_size, config_dict, subset):

    # create a unique id (gmt time)
    gmt = time.strftime('%y%m%d_%H%M%S', time.gmtime())

    # create an outfile and skip routine if it already exists
    ts_params = config_dict['ts_params']
    sat = config_dict['ts_params']['satellite']
    start_hist = ts_params['ts_start']
    end_mon = ts_params['ts_end']
    lsat_params = ts_params['lsat_params']
    final_dir = Path(ts_params['outdir']).joinpath(sat)
    final_dir.mkdir(parents=True, exist_ok=True)

    # automatically get aoi from extent of points and upload it as an asset
    logger.info(
        f'Create AOI from points and upload as temporary EE asset '
        f'inside tmp_esbae_{gmt}.'
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
            start_hist, end_mon, aoi_fc, **lsat_params
        )
    logger.info(
        f"Creating processing chunks of {chunk_size} degrees for "
        f"parallel extraction."
    )

    chunks_fc = processing_grid(aoi_fc, chunk_size)
    #chunks_fc = aoi_fc.geometry().coveringGrid(aoi_fc.geometry(), chunk_size)

    # filter for chunks with valid number of points
    to_proc = chunks_fc.map(
        lambda x: ee.Feature(x).set(
            'intersects', points_fc.filterBounds(x.geometry()).size()
        )
    )
    to_proc = to_proc.filter(ee.Filter.And(
        ee.Filter.gt('intersects', 0),
        ee.Filter.lt('intersects', ts_params['max_points_per_chunk'])
    ))

    # check if there is anything to process,
    # and if it is too large, export
    try:
        proc_size = to_proc.size().getInfo()
        if proc_size > 250:
            _, to_proc = _ee_export_table(
                ee_fc=to_proc,
                description=f"tmp_esbae_grid_{gmt}",
                asset_id=f"tmp_esbae_grid_{gmt}",
                sub_folder=f"tmp_esbae_{gmt}"
            )
            to_proc = ee.FeatureCollection(to_proc)
    except ee.EEException:
        _, to_proc = _ee_export_table(
            ee_fc=to_proc,
            description=f"tmp_esbae_grid_{gmt}",
            asset_id=f"tmp_esbae_grid_{gmt}",
            sub_folder=f"tmp_esbae_{gmt}"
        )
        to_proc = ee.FeatureCollection(to_proc)

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
        # therefore we only put the chunk geom to the extract time-series
        # (might give problems though)
        # mazbe we could map? as all other arguments are the same
        cell_fc = points_fc.filterBounds(chunk)

        #return_code = extract_time_series(
        #    sat_coll, cell_fc, config_dict, identifier
        #)
        args_to_process.append((
            sat_coll, cell_fc, config_dict, identifier
        ))

    logger.info(
        f"Starting the parallel extraction routine."
    )
    # run in parallel
    if args_to_process:
        return_code = py_helpers.run_in_parallel(
            extract_time_series, args_to_process, ts_params['ee_workers']
        )
    else:
        return_code = 1

    try:
        delete_sub_folder(f"tmp_esbae_{gmt}")
    except ee.EEException:
        pass

    if any([return_code]) != 0:
        logger.warning(
            "Not fully processed. Will retry at a higher aggregation level."
        )

    return return_code


def cascaded_extraction_ee(input_grid, config_dict):

    # get chunk levels
    chunk_sizes = config_dict['ts_params']['grid_size_levels']

    # sort by point id
    input_grid = input_grid.sort(config_dict['design_params']['pid'])

    # get number of points to process
    size = input_grid.size().getInfo()
    upload_sub = False
    if size > 25000:
        subsets = int(np.ceil(size / 25000))
        logger.info(
            f"The number of points exceeds 25000. "
            f"Processing will be split into {subsets} subsets."
        )
        # if the input_grid has more than 25k points, we upload it to a separate FC
        # in the get_missing_routine, independent if files have been already processed
        upload_sub = True
    else:
        logger.info(
            f"Preparing the processing of {size} points."
        )

    # if collection is greater than that, we iterate over chunks
    subs = []
    for i in range(0, size, 25000):

        if size > 25000:
            logger.info(f"------------------------------------------------")
            logger.info(f"Processing subset {int(i/25000+1)}/{subsets}")
            logger.info(f"------------------------------------------------")

        sub = ee.FeatureCollection(input_grid.toList(25000, i))
        # if already processed files are in the tmp_folder
        logger.info("Checking for already processed files.")
        sub = _get_missing_points(sub, config_dict, subset=int(i/25000+1), upload_sub=upload_sub)
        if sub == 'completed':
            continue

        for chunk_size in chunk_sizes:

            _ = _parallel_extract_ee(
                sub, chunk_size, config_dict, subset=int(i/25000+1)
            )
            logger.info(
                "Checking for points not processed at the current "
                "aggregation level."
            )
            sub = _get_missing_points(
                sub, config_dict, subset=int(i/25000+1), upload_sub=upload_sub
            )
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
        shutil.copy(log, Path(config_dict['ts_params']["outdir"]).joinpath(log.name))
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


def _check_config_changed(config_dict):

    # read config file is existing, so we can compare to new one
    project_dir = Path(config_dict['project_params']['project_dir'])
    sat = config_dict['ts_params']['satellite']
    out_dir = Path(config_dict['ts_params']['outdir']).joinpath(sat)
    config_file = project_dir.joinpath('config.json')
    if config_file.exists():
        with open(config_file) as f:
            old_config_dict = json.load(f)

        # create a copy of the new config for comparison
        new_ts_params = config_dict['ts_params'].copy()
        old_ts_params = old_config_dict['ts_params'].copy()

        # define keys that can be changed
        keys_list = [
            'outdir', 'ee_workers', 'max_points_per_chunk', 'grid_size_levels'
        ]
        # remove those keys from both configs
        [new_ts_params.pop(key) for key in keys_list]
        [old_ts_params.pop(key) for key in keys_list]

        if not new_ts_params == old_ts_params:
            config_change = input(
                'Your processing parameters in your config file changed. '
                'If you continue, all of your already processed files will be '
                'deleted. Are you sure you want to continue? (yes/no)'
            )
            if config_change == 'no':
                return
            elif config_change == 'yes':
                logger.info('Cleaning up results folder.')
                [file.unlink() for file in out_dir.glob('*geojson')]
                return
            else:
                raise ValueError(
                    'Answer is not recognized, should be \'yes\' or \'no\''
                )

    return


def extract(input_grid, config_dict):

    _check_config_changed(config_dict)
    if isinstance(input_grid, ee.FeatureCollection):
        cascaded_extraction_ee(input_grid, config_dict)

    # elif isinstance(input_grid, gpd.geodataframe.GeoDataFrame):
    #    cascaded_extraction(input_grid, config_file, start_res, end_res)
