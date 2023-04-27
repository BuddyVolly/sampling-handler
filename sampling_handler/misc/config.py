import json
import logging
import time
from pathlib import Path
from datetime import datetime as dt

import ee
from pyproj import Proj

from .reference_dict import REFERENCE_DICT
from ..misc.py_helpers import read_any_aoi_to_single_row_gdf, timer
from .settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


def _check_value(key, value, reference_dict):

    # if a value is set to None that's fine
    if not value:
        return

    if key == 'project_params':
        return

    # if it is a sub dictionary
    if type(value) == dict:
        for subkey in value.keys():
            _check_value(subkey, value[subkey], reference_dict[subkey])
        return

    expected_type = reference_dict['type']
    if expected_type == 'date':
        try:
            dt.strptime(value, '%Y-%m-%d')
            return
        except ValueError:
            raise ValueError(
                f'Wrong date format for {key}. Should be YYYY-MM-DD, e.g. 2018-12-31'
            )

    if expected_type == 'aoi':
        read_any_aoi_to_single_row_gdf(value, incrs='epsg:4326', outcrs='epsg:4326')
        return

    if expected_type == 'projection':
        Proj(value)
        return

    if expected_type == 'project_name':

        if not isinstance(value, str):
            raise TypeError(
                f"Configuration parameter {key} does not have the right type {value}. "
                f"It should be {str(expected_type)}."
            )

        if ' ' in 'project_name':
            raise ValueError(
                'A project\'s name cannot have spaces. Please select another name.'
            )

    if expected_type == 'project_dir':
        return

    if expected_type == 'FeatColl':
        try:
            feat = ee.FeatureCollection(value).limit(1).size().getInfo()
            return
        except ee.EEException as e:
            raise e

    # standard types
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Configuration parameter {key} does not have the right type {type(value)}. "
            f"It should be {str(expected_type)}."
        )

    # return if any value is fine
    try:
        choices = reference_dict['choices']
    except KeyError:
        pass

    if type(value) == float:
        if choices['min'] < value < choices['max']:
            return
        else:
            raise ValueError(
                f'Invalid value encountered in {key}. '
                f'Value needs to be in the range of {choices["min"]} to {choices["max"]}'
            )

    if type(value) == int:
        if type(choices) == dict:

            if choices['min'] < value < choices['max']:
                return
            else:
                raise ValueError(
                    f'Invalid value encountered in {key}. '
                    f'Value needs to be in the range of {choices["min"]} to {choices["max"]}'
                )

        if type(choices) == range:
            if value in choices:
                return
            else:
                raise ValueError(
                    f'Invalid value encountered in {key}. '
                    f'Value must be in the range of {min(choices)} and {max(choices)}.'
                )

    if type(value) == list:

        expected_type = reference_dict['list_type']
        for v in value:
            if not isinstance(v, expected_type):
                raise TypeError(
                    f"Value in list from configuration parameter {key} does not have the right type ({type(v)}) . "
                    f"It should be {str(expected_type)}."
                )

        if type(choices) == range or type(choices) == list:
            if any([True if v in choices else False for v in value]):
                return
            else:
                raise ValueError(
                    f'Invalid valid encountered in {key} list. Allowed values are {reference_dict[key]}'
                )

        elif type(choices) == dict:
            if any([True if choices['min'] < v < choices['max'] else False for v in value]):
                return
            else:
                raise ValueError(
                    f'Invalid valid encountered in {key}. '
                    f'Value needs to be in the range of {choices["min"]} to {choices["max"]}'
                )


def update_config_file(config_file, config_dict):

    logger.info('Verifying parameter settings...')
    # check for any false entries
    for key in config_dict.keys():
        _check_value(key, config_dict[key], REFERENCE_DICT[key])

    # dump to file
    with open(config_file, 'w') as f:
        json.dump(config_dict, f)


def get_default_config(project_dir=None, level='default'):

    default_conf_file = Path(__file__).parent.parent.joinpath('auxfiles/default_config.json')

    if project_dir:
        conf_file = Path(project_dir).joinpath('config.json')

        if conf_file.exists():
            # read config file
            logging.info(f'Using config file from project directory {project_dir}')
            with open(conf_file) as f:
                config_dict = json.load(f)
                return config_dict
        else:
            logging.info(
                f'No configuration file found inside the project directory {project_dir}. '
                'Loading the default parameter settings.'
            )
            with open(default_conf_file) as f:
                config_dict = json.load(f)
                return config_dict

    else:
        logging.info('No project directory given. Loading the default parameter settings.')
        with open(default_conf_file) as f:
            config_dict = json.load(f)
            return config_dict
