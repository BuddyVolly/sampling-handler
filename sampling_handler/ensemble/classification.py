# ruff: noqa: E402
# pylint: disable=wrong-import-position
import warnings as _warnings

_original_warn = None

# We get LOOOOOOTS fo awful warnings from Imblearn
# Only thing that work to silence the is the following code here
# (as warnings does not captures warnings from underlying libs)


def _warn(
        message:str, category:str='', stacklevel:int=1, source:str=''
): # need hints to work with pytorch
    pass # In the future, we can implement filters here. For now, just mute everything.


def please():
    global _original_warn
    _original_warn = _warnings.warn
    _warnings.warn = _warn


please()


import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from imblearn import FunctionSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import shapely
from shapely import wkt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
from skopt import BayesSearchCV

from ..esbae import Esbae
from ..misc import py_helpers, plt_helpers, config
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class EnsembleClassification(Esbae):

    def __init__(
            self,
            project_name,
            training_file='False',
            binary_change_column='False',
            predictors='all',
            classifier='BalancedRandomForest',
            random_state=42,
            binary_stable_forest_column=False,
            satellite='Landsat',
            aoi=None
    ):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('06_Probability_Classification'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.training_file = training_file
        self.binary_change_column = binary_change_column
        self.predictors = predictors
        self.classifier = classifier
        self.random_state = random_state
        self.binary_stable_forest_column = binary_stable_forest_column
        self.satellite = satellite

        # get params from before steps (or default values)
        conf = self.config_dict
        self.pid = conf['design_params']['pid']
        self.start_year = conf['da_params']['start_monitor'][:4]
        self.end_year = conf['da_params']['end_monitor'][:4]

        logger.info('Loading augmented dataset...')
        self.augmented_df = self.load_augmented_dataset()

        logger.info('Loading training data...')
        if not self.training_file:
            logger.info('No training data found. Creating training data from global datasets...')
            raise FileNotFoundError('No training data found...')
            #self.merged_df = self.load_global_training_data()
        else:
            self.training_df = pd.read_csv(self.training_file)

            logger.info('Merging training data with augmented dataset...')
            if self.pid in self.training_df:
                self.training_df.rename(columns={self.pid: f'{self.pid}_training'}, inplace=True)

            try:
                self.training_df = gpd.GeoDataFrame(self.training_df, geometry=gpd.points_from_xy(self.training_df.lon, self.training_df.lat), crs='epsg:4326')
            except:
                self.training_df = gpd.GeoDataFrame(self.training_df, geometry=self.training_df['geometry'].apply(wkt.loads), crs='epsg:4326')

            self.merged_df = gpd.sjoin_nearest(
                self.augmented_df,
                self.training_df[[self.pid, self.binary_change_column, 'geometry']],
                how='left',
                max_distance=0.001
            )

    def load_augmented_dataset(self):

        da_dic = Path(self.config_dict['da_params']['outdir']).joinpath(self.satellite)
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
        return df

    def outlier_rejection(self, X, y, random_state):

        """This will be our function used to resample our dataset."""
        model = IsolationForest(max_samples=100, contamination=0.1, random_state=random_state, n_jobs=-1, bootstrap=True)
        model.fit(X)
        y_pred = model.predict(X)
        return X[y_pred == 1], y[y_pred == 1]

    def binary_change_probability(
                self, train_test_split=False, outlier_removal_training=True, bayes=False, random_state=42,
        ):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

            warnings.filterwarnings(
                'ignore', 'This import path', FutureWarning
            )

            warnings.filterwarnings(
                'ignore', 'to make it possible to propagate', UserWarning
            )
            warnings.filterwarnings(action='once')


            # select columsn that are used by Classification
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
                'gfc_tc00', 'gfc_gain', 'gfc_loss', 'gfc_lossyear',
                'tmf_defyear', 'tmf_degyear', 'tmf_main', 'tmf_sub',
                'lang_tree_height', 'potapov_tree_height',
                'esri_lc_17', 'esri_lc_18', 'esri_lc_19', 'esri_lc_20', 'esri_lc_21',
                'esa_lc_20', 'esa_lc_21',
                'dw_class_mode', 'dw_tree_prob__max', 'dw_tree_prob__min',
                'dw_tree_prob__stdDev', 'dw_tree_prob_mean',
                'elevation', 'slope', 'aspect'
            ]

            if self.predictors == 'all':
                self.predictors = [col for col in esbae_cols if col in self.merged_df.columns]

            # prepare predictive variables
            logger.info('Preparing predictive variables...')
            self.merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Impute missing values
            ct = ColumnTransformer(
                [("imp", SimpleImputer(strategy="mean"), self.predictors)]
            )
            self.merged_df[self.predictors] = ct.fit_transform(self.merged_df)
            logger.info(f'Total number of data entries: {len(self.merged_df)}')

            # get target variable for training data
            y = self.binary_change_column
            train_df = self.merged_df[~self.merged_df[y].isna()]
            logger.info(f'Number of total reference data entries: {len(train_df)}')

            if train_test_split:
                logger.info('Splitting into train and test dataset')
                train_df, test_df = train_test_split(train_df, test_size=train_test_split, random_state=random_state)
                logger.info(f'Train dataset size: {len(train_df)}')
                logger.info(f'Test dataset size: {len(test_df)}')

            y_train = train_df[y]
            x_train = train_df[self.predictors]

            if len(np.unique(y_train)) != 2:
                raise ValueError('Target variable y shall only have 2 values (0 and 1)')

            if outlier_removal_training:
                reject_sampler = FunctionSampler(func=self.outlier_rejection, kw_args={'random_state': random_state})
                x_train, y_train = reject_sampler.fit_resample(x_train, y_train)

            # compute class weights
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y_train), y=y_train
            )

            # model optimization
            parameters = {
                'n_estimators': [250, 500],
                'max_depth': [5, 10, 25, 50],
                'min_samples_split': [5, 10, 25],
                'min_samples_leaf': [10, 25]
            }

            if bayes:
                brf = BayesSearchCV(
                    BalancedRandomForestClassifier(
                        random_state=random_state,
                        oob_score=True,
                        class_weight=dict(enumerate(class_weights)),
                        #n_jobs=-1
                    ),
                    search_spaces=parameters,
                    n_iter=5,
                    cv=5,
                    random_state=random_state
                )
                # run classifier
                brf.fit(x_train, y_train)
                print(f'OOB Score is {brf.best_estimator_.oob_score_}')
            else:
                brf = BalancedRandomForestClassifier(
                    n_estimators=1500,
                    random_state=random_state,
                    oob_score=True,
                    class_weight=dict(enumerate(class_weights)),
                    n_jobs=-1
                )
                brf.fit(x_train, y_train)
                print(f'OOB Score is {brf.oob_score_}')


            # run permutation, to get most important features
            print('Running feature importance analysis')
            result = permutation_importance(
                brf,
                x_train,
                y_train,
                n_repeats=25,
                random_state=random_state,
                n_jobs=-1
            )
            # turn permutation results into a dataframe
            perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in self.predictors])
            perm['AVG_Importance'] = result.importances_mean
            perm['STD_Importance'] = result.importances_std
            perm.sort_values('AVG_Importance', ascending=False, inplace=True)

            # plot feature importance
            fig, ax = plt.subplots(figsize=(12,7))
            sns.barplot(perm, x=perm.index, y='AVG_Importance', yerr=perm.STD_Importance, ax=ax)
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(f'{self.out_dir}/feat_imp.png', bbox_inches='tight')

            # predict probability
            class_probabilities = brf.predict_proba(self.merged_df[self.predictors])
            self.merged_df['chg_probability'] = class_probabilities.T[1]
            return self.merged_df[[self.pid, 'chg_probability', 'geometry']], brf

    def plot_probability(self, markersize=2, figsize=(15, 15), basemap=None, title='Change Probability Map'):

        if isinstance(
                self.merged_df.head(1).geometry.values[0],
                shapely.geometry.polygon.Polygon
        ):
            plot_df = self.merged_df.copy()
            plot_df['geometry'] = plot_df.geometry.centroid
            plot_df.set_crs('epsg:4326', inplace=True)

        ax = plt_helpers.plot_map_continous(
        plot_df,
        'chg_probability',
        markersize=markersize,
        #vmin=np.percentile(training_df[column], 2),
        #vmax=np.percentile(training_df[column], 98),
        figsize=figsize,
        #cbar_label='Range (Change Probability)',
        basemap=basemap,
        title=title
    )


def get_binary_change(row, start_year, end_year, consider_tmf=True, gfc_gain=True):
    """Extracts change from global products.

    Args:
        row (pandas.Series): A row of data containing information about the pixel being analyzed.
        start_year (int): The start year of the time period being analyzed.
        end_year (int): The end year of the time period being analyzed.
        consider_tmf (bool, optional): Whether to consider tree cover loss and degradation from the Treecover Loss and
            Gain dataset. Defaults to True.
        gfc_gain (bool, optional): Whether to consider tree cover gain from the Global Forest Change dataset. Defaults
            to True.

    Returns:
        tuple: A tuple containing the binary FNF value (1 for forest, 0 for non-forest) and the type of change (0 for
            no change, 1 for loss, 2 for gain, 3 for degradation, or 4 for both loss and degradation).
    """
    loss_year = np.nan_to_num(row.gfc_lossyear)
    gfc_loss = 1 if int(start_year[2:]) < int(loss_year) < int(end_year[2:]) else 0
    gfc_gain = 1 if row.gfc_gain and gfc_gain else 0
    if consider_tmf:
        tmf_def = 1 if int(start_year) < row.tmf_defyear < int(end_year) else 0
        tmf_deg = 1 if int(start_year) < row.tmf_degyear < int(end_year) else 0
    else:
        tmf_def, tmf_deg = 0, 0

    # get any change
    change = np.max([gfc_loss, tmf_def, tmf_deg, gfc_gain])

    # extract forest/non-forest
    ones = []
    if hasattr(row, 'lang_tree_height'):
        ones.append(1 if row.lang_tree_height > 5 else 0)

    if hasattr(row, 'potapov_tree_height'):
        ones.append(1 if row.potapov_tree_height > 5 else 0)

    if hasattr(row, 'gfc_tc00'):
        ones.append(1 if row.gfc_tc00 > 10 else 0)

    if hasattr(row, 'tmf_main'):
        ones.append(1 if row.tmf_main == 10 else 0)

    if hasattr(row, 'dw_tree_prob__min'):
        ones.append(1 if row.dw_tree_prob__min > 50 else 0)

    if hasattr(row, 'esa_lc20'):
        ones.append(1 if row.esa_lc20 == 10 or row.esa_lc20 == 95 else 0)

    if hasattr(row, 'esa_lc21'):
        ones.append(1 if row.esa_lc21 == 10 or row.esa_lc21 == 95 else 0)

    for year in range(2017, 2022, 1):
        year = str(year)
        if hasattr(row, f'esri_lc{year[:2]}'):
            ones.append(
                1 if row[f'esri_lc{year[:2]}'] == 2 or row[f'esri_lc{year[:2]}'] == 3 else 0)

    fnf = 1 if all(ones) else 0 if any(ones) != 1 else np.nan
    return fnf, change


def add_global_target(df, start, end, tmf, gain):

    # turn nan to 0
    df['gfc_lossyear'] = np.nan_to_num(df['gfc_lossyear'])

    df[['FNF', 'CNC']] = df.apply(
        lambda row: get_binary_change(row, start, end, tmf, gain),
        axis=1,
        result_type='expand'
    )
    return df


def get_stats(row, band, start, end):

    idx = [start*10000 < int(d) < end*10000 for d in row.dates]
    ts = np.array(row.ts[band])[idx]

    return np.nanmean(ts), np.nanstd(ts)


def add_yearly_reflectance(df, start, end):

    bands = df.ts.head(1).values[0].keys()
    for year in range(int(start), int(end)):

        to_class = df[[
            'point_id', 'elevation', 'slope', 'aspect', 'FNF'
        ]].copy()
        # prepare full dataframe for classification
        print('Calculating annual stats for FNF classification')
        for band in bands:
            to_class[[f'{band}_{year}_mean', f'{band}_{year}_sd']] = df.apply(
                lambda row: get_stats(row, band, year, year + 1), axis=1, result_type='expand'
            )

            #df[f'fnf_prob_{year}'] = classify(to_class)
        return

def outlier_rejection(X, y):

    """This will be our function used to resample our dataset."""
    model = IsolationForest(max_samples=100, contamination=0.1)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]


def binary_probability_classification(
        df, y, predictors=None, class_prob=1, outlier=True, random=42, bayes=False
):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    warnings.filterwarnings(
        'ignore', 'This import path', FutureWarning
    )

    warnings.filterwarnings(
        'ignore', 'to make it possible to propagate', UserWarning
    )
    warnings.filterwarnings(action='once')
    # get target variable
    y_train = df[y][~df[y].isna()]

    if len(np.unique(y_train)) != 2:
        raise ValueError('Target variable y can only have 2 values')

    # prepare predictive variable
    x_df = df[predictors].copy() if predictors else df.copy()
    imp = SimpleImputer(strategy="mean")
    x_df = imp.fit_transform(x_df)

    x_train = x_df[~df[y].isna()]

    if outlier:
        reject_sampler = FunctionSampler(func=outlier_rejection)
        x_train, y_train = reject_sampler.fit_resample(x_train, y_train)

    # compute class weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train), y=y_train
    )

    #print(len(y_train[y_train == 1]))
    #print(len(y_train[y_train == 0]))
    # model optimization
    parameters = {
        'n_estimators': [250, 500],
        'max_depth': [5, 10, 25, 50],
        'min_samples_split': [5, 10, 25],
        'min_samples_leaf': [10, 25]
    }

    if bayes:
        brf = BayesSearchCV(
            BalancedRandomForestClassifier(random_state=random),
            search_spaces=parameters,
            n_iter=5,
            cv=5
        )
        # run classifier
        brf.fit(x_train, y_train)
    else:
        brf = BalancedRandomForestClassifier(
            n_estimators=1500,
            random_state=random,
            oob_score=True,
            class_weight=dict(enumerate(class_weights)),
            n_jobs=-1
        )
        brf.fit(x_train, y_train)
        print(f'OOB Score is {brf.oob_score_}')

    # predict probability
    class_probabilities = brf.predict_proba(x_df)
    return class_probabilities.T[class_prob]
