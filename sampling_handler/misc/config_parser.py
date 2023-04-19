from pyproj import Proj
from datetime import datetime as dt

from pyproj.crs import CRSError
import ee


def check_value(key, value, reference_dict):
    if not value:
        return

    if key == 'project_params':
        return
    # if it is a sub dictionary
    if type(value) == dict:
        for subkey in value.keys():
            # print(value[subkey], subkey, reference_dict[subkey])
            check_value(subkey, value[subkey], reference_dict[subkey])

        return

    # custom types
    expected_type = reference_dict['type']
    if expected_type == 'date':
        try:
            d = dt.strptime(value, '%Y-%m-%d')
            return
        except ValueError as e:
            raise e

    if expected_type == 'projection':
        try:
            p = Proj(value)
            return
        except CRSError as e:
            raise e

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
                f'Invalid valid encountered in {key}. '
                f'Value needs to be in the range of {choices["min"]} to {choices["max"]}'
            )

    if type(value) == int:
        if type(choices) == dict:

            if choices['min'] < value < choices['max']:
                return
            else:
                raise ValueError(
                    f'Invalid valid encountered in {key}. '
                    f'Value needs to be in the range of {choices["min"]} to {choices["max"]}'
                )

        if type(choices) == range:
            if value in choices:
                return
            else:
                raise ValueError(
                    f'Invalid valid encountered in {key}. '
                    f'Value must be in the range of {min(choices)} and {max(choices) - 1}.'
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
