import logging
import os
from pathlib import Path

import contextily as cx
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from ..esbae import Esbae
from ..misc import py_helpers, config
from ..misc.settings import setup_logger
from .sfc import sfc_subsample

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class ThresholdSubSampling(Esbae):

    def __init__(self, project_name, percentile, tree_cover, tree_height, max_points, random_state=42, satellite='Landsat', aoi=None):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get self.tree_heighte AOI right with self.tree_heighte CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('05_Subsampling/Threshold'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.pid = self.config_dict['design_params']['pid']

        self.percentile = percentile
        self.tree_cover = tree_cover
        self.tree_height = tree_height
        self.max_points = max_points
        self.random_state = random_state
        self.satellite = satellite

        self.training_df = None

        # put this in if it's an old config file
        if 'subsampling_params' not in self.config_dict.keys():
            self.config_dict['subsampling_params'] = {
                "th": {
                    "percentile": 95,
                    "tree_cover": 0,
                    "tree_height": 0,
                    "max_points": False,
                    "random_state": 42
                },
                "outdir": None
            }


        self.config_dict['subsampling_params']['outdir'] = str(self.out_dir)

        self.config_dict['subsampling_params']['th']['percentile'] = self.percentile
        self.config_dict['subsampling_params']['th']['tree_cover'] = self.tree_cover
        self.config_dict['subsampling_params']['th']['tree_height'] = self.tree_height
        self.config_dict['subsampling_params']['th']['max_points'] = self.max_points
        self.config_dict['subsampling_params']['th']['random_state'] = self.random_state

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

    def extract_training_samples(self, save_as_ceo=True):

        da_dic = Path(self.config_dict['da_params']['outdir']).joinpath(self.satellite)

        # filter out monitor period
        start_year = int(self.config_dict['da_params']['start_monitor'][:4])
        end_year = int(self.config_dict['da_params']['end_monitor'][:4])
        logger.info('Loading dataset augmentation')

        # glob all files in self.tree_heighte data augmentation output folder
        files = Path(da_dic).glob('*geojson')

        # prepare for parallel execution
        files = [[str(file), False] for file in files]

        # read files in parallel nad put self.tree_heighte in a list
        result = py_helpers.run_in_parallel(
            py_helpers.geojson_to_gdf,
            files,
            workers=os.cpu_count(),
            parallelization='processes'
        )

        # concatenate dataframes from result's list
        df = pd.concat(result)

        logger.info('Filter data')

        # set up filters
        bfast_filter = (
            df.bfast_magnitude.abs() > np.nanpercentile(df.bfast_magnitude.abs(), self.percentile)
        ) & ((df.gfc_tc00 > self.tree_cover ) | (df.potapov_tree_height > self.tree_height))

        ccdc_filter = (
            df.ccdc_magnitude.abs() > np.nanpercentile(df.ccdc_magnitude.abs(), self.percentile)
        ) & ((df.gfc_tc00 > self.tree_cover ) | (df.potapov_tree_height > self.tree_height))

        cusum_filter = (
            df.cusum_magnitude.abs() > np.nanpercentile(df.cusum_magnitude.abs(), self.percentile)
        ) & ((df.gfc_tc00 > self.tree_cover ) | (df.potapov_tree_height > self.tree_height))

        ewma_filter = (
            df.ewma_jrc_magnitude.abs() > np.nanpercentile(df.ewma_jrc_magnitude.abs(), self.percentile)
        ) & ((df.gfc_tc00 > self.tree_cover ) | (df.potapov_tree_height > self.tree_height))

        slope_filter = (
            df.bs_slope_mean > np.nanpercentile(df.bs_slope_mean, self.percentile)
        ) & (df.potapov_tree_height > self.tree_height)

        glb_products =(
            ((df.gfc_lossyear >= (start_year-2000)) & (df.gfc_lossyear <= (end_year-2000))) |
            ((df.tmf_defyear >= start_year) & (df.tmf_defyear <= end_year)) |
            ((df.tmf_degyear >= start_year) & (df.tmf_degyear <= end_year))
        )

        # filter data
        self.training_df = df[bfast_filter | ccdc_filter | cusum_filter | ewma_filter | slope_filter | glb_products]
        logger.info(f'Number of initial training samples: {len(self.training_df)}')
        # subsample by subselection
        if self.max_points:

            # subsample selection with SFC
            self.training_df = sfc_subsample(
                gdf=self.training_df,
                target_point_size=self.max_points,
                seed=self.random_state
            )
        logger.info(f'Number of final training samples: {len(self.training_df)}')

        if save_as_ceo:
            logger.info(f'Saving final training samples to {self.out_dir}')
            py_helpers.save_gdf_locally(
                self.training_df.drop(['dates', 'ts'], axis=1), self.out_dir, gpkg=f'training_samples.gpkg', ceo_csv='training_samples.csv'
            )

        return self.training_df

    def plot_training_samples(self, save_figure=True, markersize=5, basemap=cx.providers.Esri.WorldImagery):

        if self.training_df is None:
            raise ValueError(
                'No cluster dataframe produced yet. Run class function "cluster"'
            )

        if isinstance(
                self.training_df.head(1).geometry.values[0],
                shapely.geometry.polygon.Polygon
        ):
            self.training_df['geom'] = self.df.geometry
            self.training_df['geometry'] = self.df.geometry.centroid
            self.training_df.set_crs('epsg:4326', inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax = self.aoi.plot(color='lightgrey', ax=ax, alpha=0.25)
        self.training_df.plot(ax=ax, markersize=markersize, color='red', alpha=0.5)

        if basemap:
            cx.add_basemap(ax, crs=self.training_df.crs.to_string(), source=basemap)
            ax.add_artist(ScaleBar(py_helpers.get_scalebar_distance(self.training_df)))

        if save_figure:
            fig.savefig(f'{self.out_dir}/samples.png')

        plt.tight_layout()
