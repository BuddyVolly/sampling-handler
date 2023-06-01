import logging
from pathlib import Path
import time
import json

import ee
import pandas as pd
import numpy as np

from .py_change import py_change
from .ccdc import get_ccdc
from .global_products import get_global_products
from sampling_handler.esbae import Esbae
from sampling_handler.misc import py_helpers, ts_helpers, config
from sampling_handler.misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class DatasetAugmentation(Esbae):

    def __init__(
            self,
            project_name,
            calibration_start,
            monitor_start,
            monitor_end,
            ts_band,
            aoi=None,
            satellite='Landsat'
    ):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('04_Dataset_Augmentation'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.calibration_start = calibration_start
        self.monitor_start = monitor_start
        self.monitor_end = monitor_end
        self.ts_band = ts_band
        self.satellite = satellite

        # get params from prior steps (or default values)
        conf = self.config_dict
        self.pid = conf['design_params']['pid']
        self.sample_asset = conf['design_params']['ee_samples_fc']

        # get run
        self.run_cusum = True
        self.run_bfast = True
        self.run_ts_metrics = True
        self.run_bs_slope = True
        self.run_jrc_nrt = True
        self.run_ccdc = True
        self.run_land_trendr = True
        self.run_global_products = True

        # get change detection defaults
        self.outlier_removal = conf['da_params']['outlier_removal']
        self.smooth_ts = conf['da_params']['smooth_ts']
        self.cusum = conf['da_params']['cusum']
        self.bfast = conf['da_params']['bfast']
        self.ts_metrics = conf['da_params']['ts_metrics']
        self.bs_slope = conf['da_params']['bs_slope']
        self.jrc_nrt = conf['da_params']['jrc_nrt']
        self.ccdc = conf['da_params']['ccdc']
        self.land_trendr = conf['da_params']['land_trendr']
        self.global_products = conf['da_params']['global_products']

        # get parallelization options
        self.py_workers = conf['da_params']['py_workers']
        self.ee_workers = conf['da_params']['ee_workers']

    def augment(self, skip_batches=None):

        # update config_dict
        # apply the basic configuration set in the cell above
        self.config_dict['da_params']['outdir'] = self.out_dir
        self.config_dict['da_params']['start_calibration'] = self.calibration_start
        self.config_dict['da_params']['start_monitor'] = self.monitor_start
        self.config_dict['da_params']['end_monitor'] = self.monitor_end
        self.config_dict['da_params']['ts_band'] = self.ts_band
        self.config_dict['da_params']['outlier_removal'] = self.outlier_removal
        self.config_dict['da_params']['smooth_ts'] = self.smooth_ts
        self.config_dict['da_params']['cusum'] = self.cusum
        self.config_dict['da_params']['bfast'] = self.bfast
        self.config_dict['da_params']['ts_metrics'] = self.ts_metrics
        self.config_dict['da_params']['bs_slope'] = self.bs_slope
        self.config_dict['da_params']['jrc_nrt'] = self.jrc_nrt
        self.config_dict['da_params']['ccdc'] = self.ccdc
        self.config_dict['da_params']['land_trendr'] = self.land_trendr
        self.config_dict['da_params']['global_products'] = self.global_products

        self.config_dict['da_params']['py_workers'] = self.py_workers
        self.config_dict['da_params']['ee_workers'] = self.ee_workers

        # check for relevant chagen that might necessitate the restart from scratch
        # check if relevant value changed,
        # and clean up out folder in case, to keep output data consistent
        if list(Path(self.out_dir).joinpath(self.satellite).glob('*geojson')):
            if _check_config_changed(self.config_dict, self.satellite):
                return

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        run_change(self.config_dict, self.satellite, skip_batches)


def ee_change(gdf, samples, config_dict):

    bands = config_dict['ts_params']['lsat_params']['bands']
    da_params = config_dict['da_params']

    ccdc = da_params['ccdc']['run']
    # landtrendr = da_params['land_trendr']['run']
    glb_prd = da_params['global_products']['run']
    ts_band = da_params['ts_band']

    if ccdc and samples:

        # check that we have all bands
        check_bpb = all(
            item in bands for item in
            da_params['ccdc']['breakpointBands']
        )

        if not check_bpb:
            logger.warning(
                "Selected breakpoint bands for CCDC are not available. "
                "Using the time-series band as breakpoint band."
            )
            da_params['ccdc']['breakpointBands'] = [ts_band]

        if "tmaskBands" in da_params['ccdc']:
            check_tmask = all(
                item in bands for item in
                da_params['ccdc']['tmaskBands']
            )
            if not check_tmask:
                logger.warning(
                    'Selected tMask bands for CCDC are not available. '
                    'Not using tMask bands.'
                )
                da_params['ccdc']['tmaskBands'] = False

        # run ccdc and add to dataframe
        gdf = get_ccdc(gdf, samples, config_dict)

    # TODO gdf = land_trendr(gdf, samples, config_dict) # if landtrendr else gdf
    # extract global products in case it's selected
    gdf = get_global_products(gdf, samples, config_dict) if glb_prd else gdf
    return gdf


def change_routine(gdf, config_dict, samples=None):

    # get algorithms from config file
    bands = config_dict['ts_params']['lsat_params']['bands']
    pid = config_dict['design_params']['pid']
    da_params = config_dict['da_params']
    outlier_removal =  da_params['outlier_removal']
    smooth_ts = da_params['smooth_ts']
    ccdc = da_params['ccdc']['run']
    land_trendr = da_params['land_trendr']['run']
    glb_prd = da_params['global_products']['run']
    bfast = da_params['bfast']['run']
    cusum = da_params['cusum']['run']
    bs_slope = da_params['bs_slope']['run']
    ts_metrics = da_params['ts_metrics']['run']
    jrc_nrt = da_params['jrc_nrt']['run']
    ts_band = da_params['ts_band']

    start = time.time()
    if outlier_removal:
        logger.info('Cleaning the time-series from outliers.')
        gdf = (
            ts_helpers.remove_outliers(gdf, bands, ts_band)
            if da_params['outlier_removal'] else gdf
        )
        py_helpers.timer(start, "Outlier removal finished in")

    if smooth_ts:
        start = time.time()
        logger.info('Smoothing the time-series with a rolling mean.')
        gdf = ts_helpers.smooth_ts(gdf, bands) if da_params['smooth_ts'] else gdf
        py_helpers.timer(start, 'Time-series smoothing finished in')

    # we cut ts data to actual change period only
    start = time.time()
    logger.info(
        'Creating a subset of the time-series for the '
        'full analysis period (calibration & monitoring).'
    )

    gdf[['dates', 'ts', 'images']] = gdf.apply(
        lambda row: ts_helpers.subset_ts(
            row,
            config_dict['da_params']['start_calibration'],
            config_dict['da_params']['end_monitor'],
            bands
        ),
        axis=1,
        result_type="expand",
    )
    py_helpers.timer(start, 'Time-series subsetting finished in')

    py_thread, ee_thread = None, None
    ee_gdf = gdf.copy()
    if any([bfast, cusum, bs_slope, ts_metrics, jrc_nrt]):
        py_thread = py_helpers.ThreadWithReturnValue(
            target=py_change, args=(gdf, config_dict)
        )
        py_thread.start()
        # gdf = py_change(gdf, config_dict)

    if any([ccdc, land_trendr, glb_prd]) and samples:
        ee_thread = py_helpers.ThreadWithReturnValue(
            target=ee_change, args=(ee_gdf, samples, config_dict)
        )
        ee_thread.start()

    if py_thread:
        py_gdf = py_thread.join()
        if not ee_thread:
            return py_gdf
    if ee_thread:
        ee_gdf = ee_thread.join()
        if not py_thread:
            return ee_gdf

    gdf = pd.merge(
        py_gdf,
        ee_gdf.drop(['ts', 'dates', 'geometry', 'images'], axis=1),
        on=pid
    )
    return gdf


def _check_config_changed(config_dict, satellite):

    # read config file is existing, so we can compare to new one
    project_dir = Path(config_dict['project_params']['project_dir'])
    out_dir = Path(config_dict['da_params']['outdir']).joinpath(satellite)
    config_file = project_dir.joinpath('config.json')
    if config_file.exists():
        with open(config_file) as f:
            old_config_dict = json.load(f)

        # create a copy of the new config for comparison
        new_ts_params = config_dict['da_params'].copy()
        old_ts_params = old_config_dict['da_params'].copy()

        # define keys that can be changed
        keys_list = [
            'outdir', 'ee_workers', 'py_workers'
        ]
        # remove those keys from both configs
        [new_ts_params.pop(key) for key in keys_list]
        [old_ts_params.pop(key) for key in keys_list]

        if new_ts_params != old_ts_params:
            config_change = input(
                'Your processing parameters in your config file changed. '
                'If you continue, all of your already processed files will be '
                'deleted. Are you sure you want to continue? (yes/no)'
            )
            if config_change == 'no':
                return True
            elif config_change == 'yes':
                logger.info('Cleaning up results folder.')
                [file.unlink() for file in out_dir.glob('*geojson')]
                return False
            else:
                raise ValueError(
                    'Answer is not recognized, should be \'yes\' or \'no\''
                )


def run_change(config_dict, satellite, skip_batches=None):

    logger.info('Initializing dataset augmentation routine...')

    if not config_dict['design_params']['ee_samples_fc']:
        raise ValueError(
            'No point feature collection defined. '
            'You need to run notebook 2 or set one manually in '
            'your configuration dictionary using '
            'the key [\'design_params\'][\'ee_samples_fc\']'
        )

    samples = ee.FeatureCollection(
        config_dict['design_params']['ee_samples_fc']
    )

    # check if samples are there
    _ = samples.limit(1).size().getInfo()

    # consists of TimeSeries and satellite
    ts_dir = Path(config_dict['ts_params']['outdir']).joinpath(satellite)
    # get number of batches from TS extraction (i.e. every 25000)
    batches = np.unique([file.name.split('_')[0] for file in ts_dir.glob('*geojson')])
    # outdir
    outdir = Path(config_dict['da_params']['outdir']).joinpath(satellite)
    outdir.mkdir(parents=True, exist_ok=True)

    for batch in batches:

        if skip_batches and batch in skip_batches:
            continue

        outfile = outdir.joinpath(f'{batch}_change.geojson')
        if outfile.exists():
            logger.info(
                f'Batch {int(batch)}/{len(batches)} has been already processed. '
                f'Going on with next one...'
            )
            continue

        start = time.time()
        logger.info(f'Accumulating batch files of {int(batch)}/{len(batches)}...')
        files = [[str(file), True] for file in ts_dir.glob(f'{batch}_*geojson')]
        result = py_helpers.run_in_parallel(
            py_helpers.geojson_to_gdf,
            files,
            workers=config_dict['da_params']['py_workers'],
            parallelization='processes'
        )
        cdf = pd.concat(result).drop_duplicates(config_dict['design_params']['pid'])

        logger.info(f'Running the dataset augmentation routines on {len(cdf)} points...')
        change_df = change_routine(cdf, config_dict, samples)

        logger.info('Dump results table to output file...')
        py_helpers.gdf_to_geojson(change_df, outfile, convert_dates=True)
        py_helpers.timer(start, f'Batch {int(batch)}/{len(batches)} finished in: ')

    logger.info('Dataset augmentation finished...')
