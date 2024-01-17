import logging
import pandas as pd
import geopandas as gpd
import ee
from datetime import datetime as dt, timedelta as td

from sampling_handler.time_series.ts_extract import TimeSeriesExtraction
from sampling_handler.misc.settings import setup_logger
from sampling_handler.misc.ee_helpers import export_to_ee

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)

class QAQC():

    def __init__(
            self,
            project_name,
            samples,
            point_id,
            change_column,
            change_categories,
            forest_categories,
            monitor_start,
            monitor_end,
            ts_band,
            scale,
            satellite='Landsat'
    ):

        self.project_name = project_name
        self.samples = samples
        self.change_column = change_column
        self.change_categories = change_categories
        self.forest_categories = forest_categories
        self.pid = point_id
        self.satellite = satellite
        self.scale = scale
        self.ts_band = ts_band
        self.gdf = None

        # read samples into pandas
        df = pd.read_csv(samples)
        # use lat and lon columns to create a geodataframe
        lon_col = [col for col in df.columns if col.lower() in ['lon', 'longitude', 'x']][0]
        lat_col = [col for col in df.columns if col.lower() in ['lat', 'latitude', 'y']][0]
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs='epsg:4326')

        # create a list of change categories
        self.gdf['CNC'] = self.gdf[change_column].apply(lambda x: 1 if x in change_categories else 0)
        self.gdf['FNF'] = self.gdf[change_column].apply(lambda x: 1 if x in forest_categories else 0)

        # upload points to earth engine
        point_fc = export_to_ee(self.gdf[[self.pid, 'geometry']], self.project_name)
        aoi = ee.FeatureCollection(point_fc).geometry().convexHull(100).buffer(1000, 100)
        bands =  [
            'green', 'red', 'nir', 'swir1', 'swir2',   # reflectance bands
            'ndfi', #'ndmi', 'ndvi',                    # indices
            'brightness', 'greenness', 'wetness'       # Tasseled Cap
        ]

        # run time-series
        ts = TimeSeriesExtraction(
            project_name=self.project_name,
            ts_start=dt.strptime(monitor_start, '%Y-%m-%d') + td(days=365*3),
            ts_end=monitor_end,
            satellite=self.satellite ,
            scale=self.scale,
            bounds_reduce=True,
            bands=bands,
            aoi=aoi
        )

        # landsat related parameters
        lsat_params = {
            'l9': True,
            'l8': True,
            'l7': True,
            'l5': True,
            'l4': True,
            'brdf': True,
            'bands': ts.bands,
            'max_cc': 75    # percent
        }

        # apply the basic configuration set in the cell above
        ts.lsat_params = lsat_params
        ts.workers = 10                   # this defines how many parallel requests will be send to EarthEngine at a time
        ts.max_points_per_chunk = 100     # this defines the maximum amount of points as send per request to Earth Engine at a time
        ts.grid_size_levels = [0.2, 0.15, 0.1, 0.075, 0.05]
        ts.sample_asset = point_fc
        ts.pid = self.pid

        finished = False
        while finished == False:
            ts.get_time_series_data()
            t = ts.check_if_completed()
            if 'Time to move on with the dataset augmentation notebook.' in t:
                finished = True

        # augment dataset

        # classify for change

        # classify for forest