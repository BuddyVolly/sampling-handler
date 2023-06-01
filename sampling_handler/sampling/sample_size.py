import math
import json
import logging
import itertools
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geemap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from retrying import retry

from ..esbae import Esbae
from ..misc import py_helpers, config
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


class SampleSize(Esbae):

    def __init__(self, project_name, aoi, start, end, tree_cover, mmu):
        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        self.start = start
        self.end = end
        self.tree_cover = tree_cover
        self.mmu = mmu

        self.out_dir = str(Path(self.project_dir).joinpath('01_Sample_Size'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # placeholders for area statistics
        self.area_dict = None
        self.loss_df = None
        self.fig_ann_deforest = None

        # target error and confidence interval for calculation
        self.target_error = self.config_dict['stats_params']['target_error']
        self.confidence = self.config_dict['stats_params']['confidence']
        self.samples_min = 10000
        self.samples_max = 500000
        self.samples_step = 49

        # outputs from sample size calculation
        self.calculated = None
        self.selected = None
        self.fig_cochran = None

        # inputs for simulation
        self.spacings = self.config_dict['stats_params']['spacings']
        self.scales = self.config_dict['stats_params']['scales']
        self.runs = self.config_dict['stats_params']['runs']
        self.random_seed = self.config_dict['stats_params']['random_seed']

        # outputs for simulation
        self.simulated = None
        self.fig_simulation = None

    def gfc_areas(self, save_figure=True):

        self.config_dict['stats_params']['outdir'] = str(self.out_dir)
        self.config_dict['stats_params']['start'] = self.start
        self.config_dict['stats_params']['end'] = self.end
        self.config_dict['stats_params']['tree_cover'] = self.tree_cover
        self.config_dict['stats_params']['mmu'] = self.mmu
        self.config_dict['stats_params']['area_dict'] = self.area_dict

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        # from hectare to pixel size
        self.area_dict, self.loss_df, self.fig_ann_deforest = get_area_statistics(
            self.aoi, self.start, self.end, self.tree_cover, self.mmu
        )

        if save_figure:
            self.fig_ann_deforest.savefig(
                Path(self.out_dir).joinpath('annual_deforestation.png')
            )

    def minimum_sample_size(self, save_figure=True):

        # update conf
        self.config_dict['stats_params']['target_error'] = self.target_error
        self.config_dict['stats_params']['confidence'] = self.confidence

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        self.calculated, self.selected = determine_minimum_sample_size(
            self.area_dict, self.target_error/100, self.confidence/100,
            self.samples_min, self.samples_max, self.samples_step
        )

        self.fig_cochran = display_minimum_sample_size(
            self.calculated, self.selected)

        if save_figure:
            self.fig_cochran.savefig(
                Path(self.out_dir).joinpath('cochran_sample_size.png')
            )

        self.config_dict['stats_params']['optimal_spacing'] = int(
            self.selected['Grid Spacing'].values[0]*1000
        )

        # update conf
        self.config_dict['stats_params']['target_error'] = self.target_error
        self.config_dict['stats_params']['confidence'] = self.confidence

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

    def simulated_sampling_error(self, save_figure=True):

        # we need to get the AOI right with the CRS
        aoi = py_helpers.read_any_aoi_to_single_row_gdf(self.aoi, incrs=self.aoi_crs)

        self.config_dict['stats_params']['spacings'] = self.spacings
        self.config_dict['stats_params']['scales'] = self.scales
        self.config_dict['stats_params']['random_seed'] = self.random_seed
        self.config_dict['stats_params']['runs'] = self.runs

        # update conf file with set parameters before running
        config.update_config_file(self.config_file, self.config_dict)

        self.simulated = gfc_sampling_simulation(
            aoi, self.start, self.end, self.area_dict, self.tree_cover, self.mmu,
            self.runs, self.spacings, self.random_seed, self.scales
        )

        self.fig_simulation = display_simulation(
            self.simulated, self.area_dict, self.calculated
        )
        if save_figure:
            self.fig_simulation.savefig(
                (Path(self.out_dir).joinpath('simulated_sample_size.png'))
            )


def _plot_loss(loss_df, start, end):

    # initialize plot
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(12.5, 7.5))

    # barplot all years
    ax = sns.barplot(loss_df, x='year', y='deforest', color='darkgrey', ax=ax)

    # annotate all years
    for g in ax.patches:
        ax.annotate(
            format(g.get_height(), '.1f'),
            (g.get_x() + g.get_width() / 2, g.get_y() + g.get_height()),
            ha='center',
            va='center',
            xytext=(0, 16),
            textcoords='offset points'
        )

    # create copy of df to highlight years of selected period
    new_df = loss_df.copy()
    new_df.loc[(new_df.year < start) | (new_df.year > end), 'deforest'] = 0
    ax = sns.barplot(new_df, x='year', y='deforest', color='orange', ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual deforestation (km\u00B2)')
    ax.set_title('Annual deforestation rates according to GFC')

    # a custom legend for a custom plot
    legend_elements = [
        Line2D([0], [0], color='orange', label='Selected Period'),
        Line2D([0], [0], color='darkgrey', label='Other years')
    ]
    ax.legend(handles=legend_elements)

    # make the plot nice
    sns.despine(offset=10, trim=True, ax=ax)

    return fig


@retry(stop_max_attempt_number=3, wait_random_min=5000, wait_random_max=10000)
def get_area_statistics(aoi, start, end, tree_cover=20, mmu=70):

    logger.info(
        'Extracting areas of forest and tree cover loss from Hansen\'s Global Forest Change '
        'product. This may take a moment...'
    )
    if not isinstance(aoi, ee.FeatureCollection):
        aoi = py_helpers.read_any_aoi_to_single_row_gdf(aoi)
        # and uplaod as FC to EE
        aoi = geemap.geopandas_to_ee(aoi)

    # load hansen image
    hansen = ee.Image('UMD/hansen/global_forest_change_2022_v1_10')
    # apply tree cover threshhold
    forest_mask = hansen.select('treecover2000').gt(tree_cover).rename('forest_area')

    # apply mmu
    mmu_pixel = int(np.floor(np.sqrt(mmu*10000)))
    mmu_mask = forest_mask.gt(0).connectedPixelCount(
        ee.Number(mmu_pixel).add(2)
    ).gte(ee.Number(mmu_pixel))
    forest_mask = forest_mask.updateMask(mmu_mask)

    # rescale the right way, if scale is different from original
    scale = mmu_pixel
    if scale > 30:
        forest_mask = forest_mask.reduceResolution(**{
            "reducer": ee.Reducer.mean(),
            "maxPixels": 65536
        }).reproject(
            forest_mask.projection().atScale(scale)
        ).gt(0.5).rename('forest_area')

    # create a pixel area image for area of full aoi
    aoi_area = (
        ee.Image(1).reproject(forest_mask.projection().atScale(scale)).rename('aoi_area')
    )

    # get actual forest area
    layer = forest_mask.addBands(aoi_area)
    forest_area_2000 = layer.multiply(ee.Image.pixelArea()).reduceRegion(**{
        "reducer": ee.Reducer.sum(),
        "geometry": aoi,
        "scale": scale,
        "maxPixels": 1e14,
    }).select(['forest_area', 'aoi_area']).getInfo()

    def yearly_loss(aoi, year):

        # create change layer for start and end date (inclusive)
        loss = hansen.select("lossyear").unmask(0)
        # get the mask right (weird decimal values in mask)
        loss = loss.updateMask(loss.mask().eq(1))
        # get loss of the year and apply forest mask
        loss = loss.eq(
            ee.Number(year).subtract(2000)
        ).updateMask(forest_mask).unmask(0)

        # rescale the right way, if scale is different from original
        if scale > 30:
            loss = loss.reduceResolution(**{
                "reducer": ee.Reducer.mean(),
                "maxPixels": 65536
            }).reproject(
                loss.projection().atScale(scale)
            ).gt(0.5)

        # get area
        loss_area = loss.multiply(ee.Image.pixelArea()).reduceRegion(**{
            "reducer": ee.Reducer.sum(),
            "geometry": aoi,
            "scale": scale,
            "maxPixels": 1e14,
        }).select(['lossyear'])

        return [year, np.round(loss_area.getInfo()['lossyear'] / 1000000, 2)]

    gfc_args = []
    for year in range(2001, 2023, 1):
        gfc_args.append([aoi, year])

    results = py_helpers.run_in_parallel(yearly_loss, gfc_args, 15)

    # aggregate to dataframe
    loss_df = pd.DataFrame(results)
    loss_df.columns = ['year', 'deforest']
    loss_df.sort_values('year', inplace=True)

    deforest_before_start = loss_df.deforest[loss_df.year < start].sum()
    deforest_period = loss_df.deforest[(loss_df.year >= start) & (loss_df.year <= end)].sum()
    forest_area = np.round(forest_area_2000['forest_area'] / 1000000, 2) - deforest_before_start

    area_dict = {
        'total_area': np.round(forest_area_2000['aoi_area'] / 1000000, 2),
        'forest_area': forest_area,
        'change_area': deforest_period
    }

    fig = _plot_loss(loss_df, start, end)
    return area_dict, loss_df, fig


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
        area_dict, max_error_margin, confidence_level, samples_min, samples_max, samples_step
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
    steps = int((samples_max - samples_min) / samples_step)
    for idx, sample_size in enumerate(range(samples_min, samples_max, steps)):

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


def display_minimum_sample_size(df, selected_spacing, xmax=None, savefile=None):
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

    axes[0].legend(['Stable Forest', 'Forest Change', 'Ideal'])
    axes[0].set_ylabel('Margin of Error (in %)')
    axes[0].set_facecolor("gainsboro")
    axes[0].grid(color='white')
    axes[0].autoscale(enable=True, axis='both', tight=False)

    axes[1].set_ylabel('Grid Spacing (in km)')
    axes[1].legend(['All spacings', 'Ideal'])
    axes[1].set_facecolor("gainsboro")
    axes[1].grid(color='white')
    axes[1].set_box_aspect(1)
    axes[1].autoscale(enable=True, axis='both', tight=False)

    sns.despine(offset=10, trim=True, ax=axes[0])
    sns.despine(offset=10, trim=True, ax=axes[1])
    return fig


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
        np.abs(np.subtract(i, actual_change)) for i in row['Sampled Change']
    ]
    mean_dev, sd_dev = np.nanmean(abs_errors), np.nanstd(abs_errors)

    # uncertainty calculation
    mean_area = np.nanmean(row['Sampled Change'])
    sd_area = np.nanstd(row['Sampled Change'])
    z_score = abs(stats.norm.ppf((1 - confidence_level) / 2))
    ci = z_score * sd_area
    uncertainty = ci / mean_area

    return mean_dev*100, sd_dev*100, mean_area, sd_area, uncertainty*100


def gfc_sampling_simulation(
        aoi,
        start,
        end,
        area_dict,
        tree_cover,
        mmu,
        nr_of_runs_per_grid,
        grid_spacings,
        random_seed,
        scale=30,
        confidence_level=0.95
):

    if not isinstance(scale, list):
        scale = [scale]

    if not isinstance(aoi, ee.FeatureCollection):
        aoi = py_helpers.read_any_aoi_to_single_row_gdf(aoi)
        # and upload as FC to EE
        aoi = geemap.geopandas_to_ee(aoi)

    # create random seeds
    np.random.seed(random_seed)
    seeds = np.random.random(nr_of_runs_per_grid)
    seeds = list(np.round(np.multiply(seeds, 100), 0))

    # load hansen image
    hansen = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")
    # apply tree cover threshhold
    forest_mask = hansen.select('treecover2000').gt(tree_cover).rename('forest_area')

    # apply mmu
    mmu_pixel = int(np.floor(np.sqrt(mmu*10000)))
    mmu_mask = forest_mask.gt(0).connectedPixelCount(ee.Number(mmu_pixel).add(2)).gte(ee.Number(mmu_pixel))
    forest_mask = forest_mask.updateMask(mmu_mask)

    # get lossyear
    loss = ee.Image('UMD/hansen/global_forest_change_2022_v1_10').select('lossyear').unmask(0)

    # get the mask right (weird decimal values in mask)
    loss = loss \
        .updateMask(forest_mask) \
        .updateMask(loss.mask().eq(1))

    # filter for time of interest
    loss = loss.gte(ee.Number(start).subtract(2000)).And(
        loss.lte(ee.Number(end).subtract(2000))
    )

    # -----------------------------------------------------------------
    # nested function for getting proportional change per grid size
    def sample_simulation(grid_spacing, _scale, _loss):

        # resample loss layer
        if _scale != 30:
            _loss = loss.reduceResolution(**{
                "reducer": ee.Reducer.mean(),
                "maxPixels": 65536
            }).reproject(
                loss.projection().atScale(_scale)
            ).gt(0.5)

        # set grid spacing at forced pixel size
        proj_at_spacing = loss.projection().atScale(grid_spacing)

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
        @retry(stop_max_attempt_number=5, wait_random_min=5000, wait_random_max=15000)
        def sample_change(seed, proj):

            # create a subsample of our change image
            cells = ee.Image.random(seed).multiply(1000000).int().reproject(proj)
            random = ee.Image.random(seed).multiply(1000000).int().reproject(
                loss.projection().atScale(_scale))

            maximum = cells.addBands(random).reduceConnectedComponents(ee.Reducer.max())
            points = random.eq(maximum).selfMask().reproject(proj.atScale(_scale))

            # create a stack with change and total pixels as 1
            stack = (
                _loss.updateMask(points)  # masked sample change
                .addBands(points)  # all samples
                .multiply(
                    ee.Image.pixelArea()
                ).rename(['sampled_change', 'sampled_area'])
            )

            # sum them up
            areas = stack.reduceRegion(**{
                'reducer': ee.Reducer.sum(),
                'geometry': aoi,
                'scale': _scale,
                'maxPixels': 1e14
            })

            # calculate proportional change to entire sampled area
            proportional_change_sampled = ee.Number(
                areas.get('sampled_change')).divide(
                ee.Number(areas.get('sampled_area'))).getInfo()

            return proportional_change_sampled

        # -----------------------------------------------------------------
        # get sample error mean and stddev
        proportional_changes = [sample_change(seed, proj_at_spacing) for seed in seeds]

        # add to a dict of all grids
        return proportional_changes, overall_sample_size.getInfo(), grid_spacing, _scale

    d, dfs = {}, []
    # we map over all different grid sizes
    logger.info('Running the sampling error simulation. This can take a while...')

    # create list of args for parallel execution
    args = []
    for arguments in itertools.product(grid_spacings, scale):
        base_args = list(arguments)
        base_args.append(loss)
        args.append(base_args)

    # extract in parallel
    results = py_helpers.run_in_parallel(sample_simulation, args, 15)

    # turn to dataframe
    stats_df = pd.DataFrame(results)
    stats_df.columns = ['Sampled Change', 'Sample Size', 'Grid Spacing', 'Scale']

    # get actual change
    actual_change = area_dict['change_area'] / area_dict['total_area']

    # add bias and uncertainty calculations
    stats_df[[
        'Bias (Mean)', 'Bias (StdDev)', 'Sampled Area (Mean)', 'Sampled Area (StdDev)',
        'Margin of Error'
    ]] = stats_df.apply(
        lambda x: add_statistical_measures(x, actual_change, confidence_level),
        axis=1,
        result_type='expand'
    )

    return stats_df


def display_simulation(simulated_df, area_dict, calculated_df=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    palette = sns.color_palette("flare", as_cmap=True)

    axes[0] = sns.scatterplot(
        data=simulated_df,
        x='Sample Size',
        y='Margin of Error',
        hue='Grid Spacing',
        size='Scale',
        palette=palette,
        ax=axes[0]
    )
    axes[0] = sns.scatterplot(
        data=calculated_df,
        x='Sample Size',
        y='Margin of Error (Deforestation)',
        hue='Grid Spacing',
        marker='*', s=100,
        legend=False,
        palette=palette,
        ax=axes[0]
    )
    axes[0].set_facecolor("gainsboro")
    axes[0].grid(color='white')
    axes[0].set_box_aspect(1)
    axes[0].autoscale(enable=True, axis='both', tight=False)
    sns.despine(offset=10, trim=False, ax=axes[0])

    # prep for violin plot
    tmp_dfs = []
    for i, row in simulated_df.iterrows():
        tmp_dfs.append(pd.DataFrame([
            (
                sampled_area * area_dict['total_area'],
                row['Grid Spacing'],
                row['Scale']
            ) for sampled_area in row['Sampled Change']
        ]))

    tmp_df = pd.concat(tmp_dfs)
    tmp_df.columns = ['Sampled Change', 'Grid Spacing', 'Scale']
    # plot
    palette = sns.color_palette("flare")
    axes[1] = sns.violinplot(
        data=tmp_df,
        x='Grid Spacing',
        y='Sampled Change',
        hue='Scale',
        ax=axes[1],
        palette=palette
    )
    # Drawing a horizontal line at point 1.25
    axes[1].axhline(area_dict['change_area'], c='black', linestyle='dotted')
    axes[1].set_facecolor("gainsboro")
    axes[1].grid(color='white')
    axes[1].set_box_aspect(1)
    axes[1].autoscale(enable=True, axis='both', tight=False)
    sns.despine(offset=10, trim=True)

    # add linelegend
    # where some data has already been plotted to ax
    handles, labels = axes[1].get_legend_handles_labels()

    # manually define a new patch
    line = Line2D([0], [0], color='black', label='True Change', linestyle='dotted')
    # handles is a list, so append manual patch
    handles.append(line)
    axes[1].legend(handles=handles, title='Scale')
    return fig
