import asyncio
import logging
from pathlib import Path
import time

import ee
import pandas as pd
import numpy as np

from .py_change import py_change
from .ccdc import get_ccdc
from .global_products import get_global_products
from ..esbae import Esbae
from ..misc import py_helpers, ts_helpers, config
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class DataAugmentation(Esbae):

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
        self.out_dir = str(Path(self.project_dir).joinpath('04_Data_Augmentation'))
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
        self.file_accumulation = conf['da_params']['file_accumulation']

    def augment(self):

        # update config_dict
        # apply the basic configuration set in the cell above
        self.config_dict['da_params']['outdir'] = self.out_dir
        self.config_dict['da_params']['start_calibration'] = self.calibration_start
        self.config_dict['da_params']['start_monitor'] = self.monitor_start
        self.config_dict['da_params']['end_monitor'] = self.monitor_end
        self.config_dict['da_params']['ts_band'] = self.ts_band
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
        self.config_dict['da_params']['file_accumulation']= self.file_accumulation

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        run_change(self.config_dict, self.satellite)


async def async_py_change(gdf, config_dict):
    return py_change(gdf, config_dict)


async def async_ccdc(gdf, samples, config_dict):
    return get_ccdc(gdf, samples, config_dict)


async def merge_data(gdf, config_dict, samples):
    py_gdf = await async_py_change(gdf, config_dict)
    ee_gdf = await async_ccdc(gdf, samples, config_dict)
    results = await asyncio.gather(py_gdf, ee_gdf)
    return results


def change_routine(gdf, config_dict, samples=None):

    # get algorithms from config file
    bands = config_dict['ts_params']['lsat_params']['bands']
    cd_params = config_dict['da_params']
    ccdc = cd_params['ccdc']['run']
    # landtrendr = cd_params['land_trendr']['run']
    glb_prd = cd_params['global_products']['run']
    bfast = cd_params['bfast']['run']
    cusum = cd_params['cusum']['run']
    bs_slope = cd_params['bs_slope']['run']
    ts_metrics = cd_params['ts_metrics']['run']
    jrc_nrt = cd_params['jrc_nrt']['run']
    ts_band = cd_params['ts_band']

    start = time.time()
    logger.info('Cleaning the time-series from outliers.')
    gdf = (
        ts_helpers.remove_outliers(gdf, bands, ts_band)
        if cd_params['outlier_removal'] else gdf
    )
    py_helpers.timer(start, "Outlier removal finished in")

    logger.info('Smoothing the time-series with a rolling mean.')
    gdf = ts_helpers.smooth_ts(gdf, bands) if cd_params['smooth_ts'] else gdf
    py_helpers.timer(start, 'Time-series smoothing finished in')
    # we cut ts data to actual change period only

    logger.info(
        'Creating a subset of the time-series according '
        'to the analysis period.'
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
    # -----------
    # ASYNC
    # if any([bfast, cusum, bs_slope, ts_metrics, jrc_nrt])\
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
            cd_params['ccdc']['breakpointBands']
        )

        if not check_bpb:
            logger.warning(
                "Selected breakpoint bands for CCDC are not available. "
                "Using the time-series band as breakpoint band."
            )
            cd_params['ccdc']['breakpointBands'] = [ts_band]

        if "tmaskBands" in cd_params['ccdc']:
            check_tmask = all(
                item in bands for item in
                cd_params['ccdc']['tmaskBands']
            )
            if not check_tmask:
                logger.warning(
                    'Selected tMask bands for CCDC are not available. '
                    'Not using tMask bands.'
                )
                cd_params['ccdc']['tmaskBands'] = False

        # run ccdc and add to dataframe
        gdf = get_ccdc(gdf, samples, config_dict)

    # gdf = land_trendr(gdf, samples, config_dict) # if landtrendr else gdf
    # extract global products in case it's selected
    gdf = get_global_products(gdf, samples, config_dict) if glb_prd else gdf

    return gdf


def run_change(config_dict, satellite):

    logger.info('Initializing data augmentation routine...')
    # TODO do some check
    samples = ee.FeatureCollection(
        config_dict['design_params']['ee_samples_fc']
    )

    # consists of TimeSeries and satellite
    ts_dir = Path(config_dict['ts_params']['outdir']).joinpath(satellite)
    # get number of batches from TS extraction (i.e. every 25000)
    batches = np.unique([file.name.split('_')[0] for file in ts_dir.glob('*geojson')])
    # outdir
    outdir = Path(config_dict['da_params']['outdir']).joinpath(satellite)
    outdir.mkdir(parents=True, exist_ok=True)

    for batch in batches:
        start = time.time()
        logging.info(f'Accumulating batch files of {int(batch) + 1}/{len(batches)}...')
        files = [[str(file), True] for file in ts_dir.glob(f'{batch}_*geojson')]
        result = py_helpers.run_in_parallel(
            py_helpers.geojson_to_gdf,
            files,
            workers=config_dict['da_params']['py_workers'],
            parallelization='processes'
        )
        cdf = pd.concat(result)

        logger.info(f'Running the data augmentation routines on {len(cdf)} points...')
        change_df = change_routine(cdf, config_dict, samples)

        logger.info('Dump results table to output file...')
        py_helpers.gdf_to_geojson(
            change_df,
            outdir.joinpath(f'{batch}_change.geojson'),
            convert_dates=True
        )
        py_helpers.timer(start, f'Batch {int(batch) + 1}/{len(batches)} finished in: ')
    logger.info('Data augmentation finished...')
