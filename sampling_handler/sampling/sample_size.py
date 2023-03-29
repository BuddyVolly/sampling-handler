import math

import ee
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from retrying import retry


@retry(stop_max_attempt_number=3, wait_random_min=5000, wait_random_max=10000)
def gfc_area_statistics(
        aoi, start=2001, end=2022, tree_cover=20, scale=30
):
    """Extract area for AOI, Forest and Deforestation from GFC data

    Args:
        aoi (ee.Geometry): An Earth Engine geometry representing the
                           area of interest.
        start (int, optional): The start year (inclusive) of the analysis.
                               Defaults to 2001.
        end (int, optional): The end year (inclusive) of the analysis.
                             Defaults to 2022.
        tree_cover (int, optional): The minimum tree cover percentage to
                                    consider as forest. Defaults to 20.
        scale (int, optional): The spatial resolution of the analysis
                               in meters. Defaults to 30.

    Returns:
        dict: A dictionary containing the following keys:
            - 'total_area': The total area of the AOI in square kilometers.
            - 'forest_area': The forest area of the AOI in square kilometers.
            - 'change_area': The deforested area of the AOI in
                             square kilometers.

    Raises:
        ee.EEException: If an Earth Engine API error occurs.
    """

    # load hansen image
    hansen = ee.Image("UMD/hansen/global_forest_change_2021_v1_9")
    # create change layer for start and end date (inclusive)
    loss = hansen.select("lossyear").unmask(0)
    # get the mask right (weird decimal values in mask)
    loss = loss.updateMask(loss.mask().eq(1))
    # get masked change
    change = loss.gte(ee.Number(start).subtract(2000)).And(
        loss.lte(ee.Number(end).subtract(2000))
    )

    # extract forest with subtracted change before start year
    forest = (
        hansen.select("treecover2000")
        .updateMask(loss.gte(ee.Number(start).subtract(2000)).Or(loss.eq(0)))
        .gt(tree_cover)
        .unmask(0)
    )

    # create pixel area stack for forest, change and full aoi
    pixel_areas = (
        forest
        .addBands(change)
        .addBands(ee.Image(1))
        .multiply(ee.Image.pixelArea())
        .rename(["forest_area", "change_area", "total_area"])
    )

    # extract areas for the given aoi at the given scale
    areas = pixel_areas.reduceRegion(**{
        "reducer": ee.Reducer.sum(),
        "geometry": aoi,
        "scale": scale,
        "maxPixels": 1e14,
    })

    # turn result into a dictionary
    d, areas = {}, areas.getInfo()
    for area in areas.keys():
        d[area] = np.round(areas[area] / 1000000, 2)

    # create a timespan to calculate annual average
    timespan = end - start + 1
    print(
        f"According to the GFC product, the Area of Interest covers "
        f"an area of {d['total_area']} square kilometers, "
        f"of which {d['forest_area']} square kilometers have been forested "
        f"in {start} ({np.round(d['forest_area']/d['total_area']*100, 2)} %). "
        f"Between {start} and {end}, {d['change_area']} "
        f"square kilometers have been deforested."
        f"That corresponds to {d['change_area']/timespan} "
        f"square kilometers of annual deforestation in average."
    )

    # return values as dictionary
    return d


def cochran_sample_size(precision, confidence_level, population_proportion):
    """
    Calculates the sample size required to estimate a population proportion
    using the Cochran formula.

    Parameters:
    precision (float): the desired level of precision (margin of error)
    confidence_level (float): the desired level of confidence
    population_proportion (float): the proportion of the population being
                                   estimated

    Returns:
    The required sample size (int)
    """

    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))
    p_hat = population_proportion
    q_hat = 1 - p_hat
    e = precision * population_proportion

    n = ((z_score**2) * p_hat * q_hat) / (e**2)

    return math.ceil(n)


def cochran_margin_of_error(
        sample_size, confidence_level, population_proportion
):
    """
    Calculates the maximum margin of error for estimating a population
    proportion using a given sample size and confidence level,
    using the Cochran formula.

    Parameters:
    sample_size (int): the sample size being used
    confidence_level (float): the desired level of confidence (0-1)
    population_proportion (float): the proportion of the population being
                                   estimated

    Returns:
    The maximum margin of error (float)
    """

    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))
    p_hat = population_proportion
    q_hat = 1 - p_hat

    e = z_score * math.sqrt(
        (p_hat * q_hat) / sample_size
    ) / population_proportion

    return e


def determine_minimum_sample_size(
        area_dict, max_error_margin, confidence_level
):
    """
    Determines the margin of errors for forest and deforestation areas
    at a range of given sample sizes respective grid spacing and
    returns the DataFrame and the DataFrame row that is just below
    the maximum error.

    Args:
        area_dict: A dictionary containing the total area, forest area,
                   and deforestation area of interest.
        max_error_margin: A float specifying the maximum margin of error
                          allowed as a relative value. Default is 0.1.
        confidence_level: A float specifying the desired confidence level
                          as proportion. Default is 0.9.

    Returns:
        A tuple containing a pandas DataFrame object and the DataFrame row of
        the selected spacing below the maximum error margin.
    """

    d = {}
    for idx, sample_size in enumerate(range(10000, 500000, 10000)):

        # calculate the error at given sample size for deforestation areas
        change_proportion = area_dict["change_area"] / area_dict["total_area"]
        change_error = cochran_margin_of_error(
            sample_size=sample_size,
            confidence_level=confidence_level,
            population_proportion=change_proportion
        ) * 100  # multiply by 100 to get percentage

        # calculate the error at given sample size for forest area
        forest_proportion = area_dict["forest_area"] / area_dict["total_area"]
        forest_error = cochran_margin_of_error(
            sample_size=sample_size,
            confidence_level=confidence_level,
            population_proportion=forest_proportion
        ) * 100  # multiply by 100 to get percentage

        # calculate grid spacing for given sample size
        spacing = np.sqrt(area_dict["total_area"] / sample_size)
        # add calculations to a dictionary
        d[idx] = sample_size, forest_error, change_error, spacing

    # turn all dictionary entries into a pd DataFrame object
    df = pd.DataFrame.from_dict(d, orient="index")
    df.columns = [
        "Sample Size",
        "Margin of Error (Forest)",
        "Margin of Error (Deforestation)",
        "Grid Spacing",
    ]

    # get he DataFrame row that is just below the maximum error
    selected_spacing = df[
        df["Margin of Error (Deforestation)"] < max_error_margin*100
    ].head(1)

    # return both DataFrame and row of selected
    return df, selected_spacing


def display_minimum_sample_size(df, selected_spacing, savefile=None):
    """Displays the sample size versus margin of error derived from Cochran's
    theorem.

    This function takes in a pandas DataFrame with sample sizes, margin of
    errors for forest and deforestation areas, and grid spacing calculated
    by a given total area and displays them in two scatterplots.

    Args:
        df: A pandas DataFrame object containing sample sizes, margin of errors
            for forest and deforestation areas, and grid spacing.
        selected_spacing: A pandas DataFrame object containing a row with a
                          sample size that is just below the maximum error.
        savefile: A string specifying the path to save the output scatterplot.
                  Default is None.

    Returns:
        None.
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    # plot Margin of error for Forest
    axes[0] = sns.scatterplot(
        data=df,
        x='Sample Size',
        y='Margin of Error (Forest)',
        color='green',
        ax=axes[0]
    )

    # plot Margin of error for Deforestation
    axes[0] = sns.scatterplot(
        data=df,
        x='Sample Size',
        y='Margin of Error (Deforestation)',
        color='orange',
        ax=axes[0]
    )

    # plot Ideal Margin of error for Deforestation
    axes[0] = sns.scatterplot(
        data=selected_spacing,
        x='Sample Size',
        y='Margin of Error (Deforestation)',
        color='blue',
        ax=axes[0]
    )

    # plot Grid Spacing
    axes[1] = sns.scatterplot(
        data=df,
        x='Sample Size',
        y='Grid Spacing',
        color='white',
        ax=axes[1]
    )

    # plot Ideal Grid Spacing
    axes[1] = sns.scatterplot(
        data=selected_spacing,
        x='Sample Size',
        y='Grid Spacing',
        color='blue',
        ax=axes[1]
    )

    # set some formatting
    axes[0].legend(['Stable Forest', 'Forest Change', 'Ideal'])
    axes[0].set_ylabel('Margin of Error (in %)')
    axes[0].set_facecolor("gainsboro")
    axes[0].grid(color='white')
    axes[0].autoscale(enable=True, axis='both', tight=True)
    axes[1].legend(['All spacings', 'Ideal'])
    axes[1].set_facecolor("gainsboro")
    axes[1].grid(color='white')
    axes[1].set_box_aspect(1)
    axes[1].autoscale(enable=True, axis='both', tight=True)

    # save to a file
    if savefile:
        fig.savefig(savefile)


@retry(stop_max_attempt_number=5, wait_random_min=5000, wait_random_max=15000)
def gfc_sampling_simulation(
        aoi,
        start,
        end,
        area_dict,
        nr_of_runs_per_grid,
        grid_spacings,
        random_seed,
        scale=30,
        confidence_level=0.95
):
    # create random seeds
    np.random.seed(random_seed)
    seeds = np.random.random(nr_of_runs_per_grid)
    seeds = list(np.round(np.multiply(seeds, 100), 0))

    # get lossyear
    loss = ee.Image("UMD/hansen/global_forest_change_2021_v1_9").select(
        'lossyear'
    ).unmask(0)
    # filter for time of interest
    loss = loss.gte(ee.Number(start).subtract(2000)).And(
        loss.lte(ee.Number(end).subtract(2000)))
    # re-scale
    if scale != 30:
        loss = loss.reduceResolution(**{
                "reducer": ee.Reducer.mean(), "maxPixels": 65536
            }).reproject(loss.projection().atScale(scale)).mask().gt(0.5)

    # -----------------------------------------------------------------
    # nested function for getting proportional change per grid size
    def sample_simulation(grid_spacing):

        # set grid spacing as forced pixel size
        proj_at_spacing = ee.Projection("EPSG:3857").atScale(grid_spacing)

        # get overall sample size
        overall_sample_size = ee.Image(1).rename('sample_size').reproject(
            proj_at_spacing
        ).reduceRegion(**{
                'reducer': ee.Reducer.sum(),
                'geometry': aoi,
                'scale': grid_spacing,
                'maxPixels': 1e14
            }).get('sample_size')

        # -----------------------------------------------------------------
        # nested function for getting proportional change per seed and grid
        def sample_change(seed, proj):

            # create a subsample of our change image
            cells = ee.Image.random(seed).multiply(1000000).int().reproject(
                proj)
            random = ee.Image.random(seed).multiply(1000000).int()
            maximum = cells.addBands(random).reduceConnectedComponents(
                ee.Reducer.max()
            )
            points = random.eq(maximum).selfMask().reproject(
                proj.atScale(scale)
            )

            # create a stack with change and total pixels as 1
            stack = (
                loss.updateMask(points)  # masked sample change
                .addBands(points)  # all samples
                .multiply(
                    ee.Image.pixelArea()
                ).rename(['sampled_change', 'sampled_area'])
            )

            # sum them up
            areas = stack.reduceRegion(**{
                'reducer': ee.Reducer.sum(),
                'geometry': aoi,
                'scale': scale,
                'maxPixels': 1e14
            })

            # calculate proportional change to entire sampled area
            proportional_change_sampled = ee.Number(
                areas.get('sampled_change')).divide(
                ee.Number(areas.get('sampled_area'))).getInfo()

            return proportional_change_sampled

        # -----------------------------------------------------------------
        # get sample error mean and stddev
        proportional_changes = [
            sample_change(seed, proj_at_spacing) for seed in seeds
        ]

        # add to a dict of all grids
        return proportional_changes, overall_sample_size.getInfo()

    d, dfs = {}, []
    # we map over all different grid sizes
    print(" Running the sampling error simulation. "
          "Please be patient, this can take a while.")
    for idx, spacing in enumerate(grid_spacings):
        print(
            f" Running {nr_of_runs_per_grid} times the sample error"
            f" simulation with a grid spacing of {spacing}"
            f" meters at a scale of {scale}."
        )

        sampled_change, sample_size = sample_simulation(spacing)
        d['idx'] = idx
        d['spacing'] = spacing
        d['sample_size'] = sample_size
        d['sampled_change'] = sampled_change
        dfs.append(pd.DataFrame([d]))

    # concatenate all dataframes
    df = pd.concat(dfs)
    # get actual change
    actual_change = area_dict['change_area']/area_dict['total_area']
    # add bias and uncertainty calculations
    df[[
        'mean_bias', 'sd_bias', 'mean_sampled_area', 'sd_sampled_area',
        'uncertainty'
    ]] = df.apply(
        lambda x: add_statistical_measures(x, actual_change, confidence_level),
        axis=1, result_type='expand'
    )

    return df


def add_statistical_measures(row, actual_change, confidence_level=0.95):
    """
    Calculates bias and uncertainty for a given row of sampled data and actual
    change value.

    Args:
        row: A pandas DataFrame row containing the sampled change data.
        actual_change: A float representing the actual change value for the
                       area of interest.
        confidence_level: A float specifying the desired confidence level
                          as a percentage. Default is 0.95.

    Returns:
        A tuple containing the mean deviation, standard deviation of deviation,
        mean area, standard deviation of area, and uncertainty.
    """

    # bias calculation
    abs_errors = [
        np.abs(np.subtract(i, actual_change)) for i in row['sampled_change']
    ]
    mean_dev, sd_dev = np.nanmean(abs_errors), np.nanstd(abs_errors)

    # uncertainty calculation
    mean_area = np.nanmean(row['sampled_change'])
    sd_area = np.nanstd(row['sampled_change'])
    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))
    ci = z_score * sd_area
    uncertainty = ci / mean_area

    return mean_dev, sd_dev, mean_area, sd_area, uncertainty
