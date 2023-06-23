import os
import json
import logging
from pathlib import Path

from sampling_handler.misc.config import get_default_config
from sampling_handler.misc.settings import setup_logger
from sampling_handler.misc import py_helpers

# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)

class Esbae:

    def __init__(self, project_name, aoi=None):

        self.project_name = project_name
        # TODO
        # Check if there are spaces

        if os.environ['USER'] == 'sepal-user':
            self.project_dir = str(Path.home().joinpath(f'module_results/esbae/{project_name}'))
        else:
            self.project_dir = str(Path.home().joinpath(f'{project_name}'))

        # create project directory if not existent
        if Path(self.project_dir).exists():
            logger.info(f'Using existing project directory at {self.project_dir}')
        else:
            logger.info(f'Creating project directory at {self.project_dir}')
            Path(self.project_dir).mkdir(parents=True, exist_ok=True)

        self.config_file = str(Path(self.project_dir).joinpath('config.json'))
        if Path(self.config_file).exists():
            # read config file
            logger.info(f'Using existent config file from project directory {self.project_dir}')
            with open(self.config_file) as f:
                self.config_dict = json.load(f)
        else:
            logger.info('Creating a new configuration file with default parameter settings.')
            self.config_dict = get_default_config()

        # here we transform the given aoi into a string for putting it into the configuration file
        self.aoi = self.config_dict['project_params']['aoi']
        self.aoi_crs = self.config_dict['project_params']['aoi_crs']

        if aoi:
            aoi_gdf = py_helpers.read_any_aoi_to_single_row_gdf(aoi)
            aoi_crs = f'EPSG:{aoi_gdf.crs.to_epsg()}'
            aoi_geom = aoi_gdf.geometry.__geo_interface__
        else:
            aoi_geom = None

        if self.aoi and aoi_geom:
            if json.dumps(self.aoi) != json.dumps(aoi_geom):
                overwrite_aoi = input(
                    'It seems a different AOI is already defined within your configuration. '
                    'Do you want to overwrite it (yes/no)'
                )
                if overwrite_aoi == 'yes':
                    self.aoi = aoi_geom
                    self.aoi_crs = aoi_crs

        elif aoi:
            self.aoi = aoi_geom
            self.aoi_crs = aoi_crs

        # update config dict and overwrite configuration file
        self.config_dict['project_params']['project_dir'] = self.project_dir
        self.config_dict['project_params']['aoi'] = self.aoi
        self.config_dict['project_params']['aoi_crs'] = self.aoi_crs

        # TODO
        with open(self.config_file, 'w') as f:
            json.dump(self.config_dict, f)
