import logging
import warnings
from pathlib import Path
from zipfile import ZipFile

import contextily as cx
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..esbae import Esbae
from ..misc import py_helpers
from ..misc.settings import setup_logger
from .sfc import sfc_subsample

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class KMeansSubSampling(Esbae):

    def __init__(self, project_name, clusters, points_per_cluster, satellite='Landsat', random_state=42, aoi=None):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('05_Subsampling/KMeans_Unsupervised'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.pid = self.config_dict['design_params']['pid']

        self.clusters = clusters
        self.points_per_cluster = points_per_cluster
        self.random_state = random_state
        self.cols_to_cluster = None
        self.satellite = satellite
        self.df = None
        self.sample_df = None
        self.sampling_type = 'space_filling_curve'

    def cluster(self, standardize_inputs=False):

        da_dic = Path(self.config_dict['da_params']['outdir']).joinpath(self.satellite)
        logger.info('Aggregating files from dataset augmentation step')
        self.df = py_helpers.aggregate_outfiles(da_dic)

        # select columsn that are used by Kmeans
        esbae_cols = [
            # change algorithms
            'mon_images',
            'bfast_magnitude', 'bfast_means',
            'cusum_confidence', 'cusum_magnitude',
            'bs_slope_mean', 'bs_slope_sd', 'bs_slope_max', 'bs_slope_min',
            'ewma_jrc_change', 'ewma_jrc_magnitude',
            'mosum_jrc_change', 'mosum_jrc_magnitude',
            'cusum_jrc_change', 'cusum_jrc_magnitude',
            'ccdc_magnitude',

            # spectral indices
            'ndfi_mean', 'ndfi_sd', 'ndfi_min', 'ndfi_max',
            'swir2_mean', 'swir2_sd', 'swir2_min', 'swir2_max',
            'swir1_mean', 'swir1_sd', 'swir1_min', 'swir1_max',
            'nir_mean', 'nir_sd', 'nir_min', 'nir_max',
            'red_mean', 'red_sd', 'red_min', 'red_max',
            'green_mean', 'green_sd', 'green_min', 'green_max',
            'blue_mean', 'blue_sd', 'blue_min', 'blue_max',
            'brightness_mean', 'brightness_sd', 'brightness_min', 'brightness_max',
            'wetness_mean', 'wetness_sd', 'wetness_min', 'wetness_max',
            'greenness_mean', 'greenness_sd', 'greenness_min', 'greenness_max',

            # global products
            'gfc_tc00',
            'dw_class_mode', 'dw_tree_prob__max', 'dw_tree_prob__min',
            'dw_tree_prob__stdDev', 'dw_tree_prob_mean',
            'elevation', 'slope', 'aspect'
        ]

        self.cols_to_cluster = [col for col in esbae_cols if col in self.df.columns]
        logger.info('Using the following attributes as input for KMeans:')
        logger.info(f'{self.cols_to_cluster}')
        x_values = self.df[self.cols_to_cluster]

        # prepare predictive variables
        logger.info('Imputing missing data')
        x_values.replace([np.inf, -np.inf], np.nan, inplace=True)
        imp = SimpleImputer(strategy="mean")
        x_values = imp.fit_transform(x_values)

        if standardize_inputs:
            logger.info('Standardizing input data...')
            x_values = StandardScaler().fit_transform(x_values)

        # run kmeans
        logger.info('Running KMeans...')
        self.df['KMeans'] = KMeans(
            n_clusters=self.clusters,
            random_state=self.random_state,
            n_init='auto'
        ).fit_predict(x_values)

        # plot data
        logger.info('KMeans resulted in the following number of samples within each cluster:')
        cluster_df = pd.DataFrame(np.unique(self.df['KMeans'], return_counts=True)).T
        cluster_df.columns = ['Clusters', 'Nr. of samples per cluster']

        sns.set(style="white")
        fig, ax = plt.subplots(figsize=(12.5, 7.5))

        # barplot all years
        ax = sns.barplot(cluster_df, x='Clusters', y='Nr. of samples per cluster', color='darkgrey', ax=ax)

        # annotate all years
        for g in ax.patches:
            ax.annotate(
                format(int(g.get_height())),
                (g.get_x() + g.get_width() / 2, g.get_y() + g.get_height()),
                ha='center',
                va='center',
                xytext=(0, 16),
                textcoords='offset points'
            )

        # make the plot nice
        sns.despine(offset=10, trim=True, ax=ax)

    def plot_clusters(
            self,
            cluster_to_highlight=None,
            save_figure=True,
            markersize=5,
            basemap=cx.providers.Esri.WorldImagery
    ):

        if self.df is None:
            raise ValueError(
                'No cluster dataframe produced yet. Run class function "cluster"'
            )

        if isinstance(
                self.df.head(1).geometry.values[0],
                shapely.geometry.polygon.Polygon
        ):
            self.df['geom'] = self.df.geometry
            self.df['geometry'] = self.df.geometry.centroid
            self.df.set_crs('epsg:4326', inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax = self.aoi.plot(color='lightgrey', ax=ax, alpha=0.25)
        self.df.plot(
            ax=ax,
            column='KMeans', legend=True, markersize=markersize, cmap='tab20'
        )

        if cluster_to_highlight:
            self.df[self.df['KMeans'] == cluster_to_highlight].plot(
                ax=ax, markersize=markersize*2, facecolor='red'
            )

        if basemap:
            cx.add_basemap(ax, crs=self.df.crs.to_string(), source=basemap)
            ax.add_artist(ScaleBar(py_helpers.get_scalebar_distance(self.df)))

        if save_figure:
            fig.savefig(f'{self.out_dir}/clusters.png')

        plt.tight_layout()

    def plot_stats(self, class_column, cols_to_plot, save_figures=True):

        if self.df is None:
            raise ValueError(
                'No cluster dataframe produced yet. Run class function "cluster"'
            )

        figs, axs = {}, {}
        if isinstance(cols_to_plot, str):
            cols_to_plot = [cols_to_plot]

        for col in cols_to_plot:
            figs[col] = plt.figure(figsize=(15, 5))
            axs[col] = figs[col].add_subplot(111)
            axs[col] = sns.boxplot(x=class_column, y=col, data=self.df, ax=axs[col])

            if col.startswith("bfast_mag"):
                axs[col].set(ylim=(-3000, 3000))

            if col.startswith("bfast_mea"):
                axs[col].set(ylim=(-10, 10))

            if col == "dw_class_mode":
                axs[col].set_ylabel("Dynamic World Class (Mode)")
                axs[col].set_yticks(range(9))
                axs[col].set_yticklabels(
                    [
                        "Water (0)",
                        "Trees (1)",
                        "Grass (2)",
                        "Flooded Vegetation (3)",
                        "Crops (4)",
                        "Shrubs and Scrub (5)",
                        "Built (6)",
                        "Bare (7)",
                        "Snow and Ice (8)",
                    ]
                )

            if col == "esri_lc20":
                axs[col].set_ylabel("ESRI Land Cover 2020")
                axs[col].set_yticks(range(11))
                axs[col].set_yticklabels(
                    [
                        "No data (1)",
                        "Water (2)",
                        "Trees (3)",
                        "Grass (4)",
                        "Flooded Vegetation (5)",
                        "Crops (6)",
                        "Shrubs and Scrub (7)",
                        "Built (8)",
                        "Bare (9)",
                        "Snow and Ice (10)",
                        "Clouds (11)",
                    ]
                )

            if col == "esa_lc20":
                axs[col].set_ylabel("ESA World Cover 2020")
                axs[col].set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
                axs[col].set_yticklabels(
                    [
                        "Trees (10)",
                        "Shrubland (20)",
                        "Grassland (30)",
                        "Cropland (40)",
                        "Built (50)",
                        "Barren/Sparse veg. (60)",
                        "Snow and Ice (70)",
                        "Open Water (80)",
                        "Herbaceous wetland (90)",
                        "Mangroves (95)",
                        "Moss and lichen (100)",
                    ]
                )

            plt.grid(axis="x")

            if save_figures:
                figs[col].savefig(f'{self.out_dir}/{col}.png')

    def select_samples(self, save_as_ceo=True):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        if self.df is None:
            raise ValueError(
                'No cluster dataframe produced yet. Run class function "cluster"'
            )

        df = self.df
        subsets = []
        for cluster in df['KMeans'].unique():

            # if nr. of points in cluster is less, then take all
            if len(df[df['KMeans'] == cluster]) < self.points_per_cluster:
                subsets.append(
                    df[df['KMeans'] == cluster].sample(len(df[df['KMeans'] == cluster]))
                )
            else:
                if self.sampling_type == 'random':
                    subsets.append(df[df['KMeans'] == cluster].sample(self.points_per_cluster))
                elif self.sampling_type == 'space_filling_curve':
                    subsets.append(
                        sfc_subsample(df[df['KMeans'] == cluster], self.points_per_cluster)
                    )

        self.sample_df = pd.concat(subsets).set_crs('epsg:4326')
        logger.info(f'{len(self.sample_df)} samples have been selected in total')

        if save_as_ceo:
            logger.info(
                f'Saving CEO compatible zipped shapefile samples to {self.out_dir}/samples.zip'
            )

            self.sample_df['PLOTID'] = gpd.GeoDataFrame(self.sample_df)[self.pid]
            self.sample_df['LON'] = gpd.GeoDataFrame(self.sample_df).geometry.x
            self.sample_df['LAT'] = gpd.GeoDataFrame(self.sample_df).geometry.y

            if 'geom' in self.df.columns:
                sample_df = self.sample_df[['LON', 'LAT', 'PLOTID', 'KMeans', 'geom']].copy()
                sample_df['geometry'] = self.sample_df['geom']
                sample_df.set_geometry('geometry', inplace=True)
                sample_df.set_crs('epsg:4326', inplace=True)
                sample_df.drop('geom', axis=1, inplace=True)
            else:
                sample_df = self.sample_df[['LON', 'LAT', 'PLOTID', 'KMeans', 'geometry']].copy()

            sample_df.to_file(f'{self.out_dir}/samples.shp')
            zip_file_name = Path(f'{self.out_dir}/samples.zip')
            filenames = [
                zip_file_name.with_suffix(suffix) for suffix in ['.shp', '.prj', '.shx', '.dbf']
            ]
            with ZipFile(zip_file_name, 'w') as zip_object:
                for filename in filenames:
                    zip_object.write(filename, filename.name)
                    filename.unlink()

            logger.info(
                f'Saving CEO compatible file of sample centroids to {self.out_dir}/samples.csv'
            )
            sample_df.drop('geometry', axis=1).to_csv(
                f'{self.out_dir}/samples.csv', index=False
            )

    def plot_samples(
            self,
            save_figure=True,
            figsize=(12, 12),
            markersize=5,
            basemap=cx.providers.Esri.WorldImagery
    ):

        if isinstance(
                self.sample_df.head(1).geometry.values[0],
                shapely.geometry.polygon.Polygon
        ):

            self.sample_df['geom'] = self.df['geometry']
            self.sample_df['geometry'] = self.df.geometry.centroid
            self.sample_df.set_crs('EPSG:4326', inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = self.aoi.plot(color='lightgrey', ax=ax, alpha=0.25)
        self.sample_df.plot(
            ax=ax,
            column='KMeans', legend=True, markersize=markersize, cmap='tab20'
        )
        # add basemap
        cx.add_basemap(ax, crs=self.sample_df.crs.to_string(), source=basemap)
        ax.add_artist(ScaleBar(py_helpers.get_scalebar_distance(self.sample_df)))

        if save_figure:
            fig.savefig(f'{self.out_dir}/samples.png')
        plt.tight_layout()
