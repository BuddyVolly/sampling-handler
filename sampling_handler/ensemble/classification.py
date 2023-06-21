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
import warnings
from pathlib import Path

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn import FunctionSampler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from skopt import BayesSearchCV
from ..esbae import Esbae
from ..misc.settings import setup_logger
from ..misc import py_helpers


# Create a logger object
logger = logging.getLogger(__name__)
LOGFILE = setup_logger(logger)


class EnsembleClassification(Esbae):

    def __init__(
            self,
            project_name,
            satellite,
            ts_start,
            ts_end,
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
        self.out_dir = str(Path(self.project_dir).joinpath('05b_Supervised_Global'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.start = ts_start
        self.end = ts_end
        self.satellite = satellite
        self.outlier = False
        self.bayes_optim = False

        # get params from befre steps (or default values)
        conf = self.config_dict
        self.pid = conf['design_params']['pid']
        self.sample_asset = conf['design_params']['ee_samples_fc']

        # load default params
        self.lsat_params = conf['ts_params']['lsat_params']
        self.workers = conf['ts_params']['ee_workers']
        self.max_points_per_chunk = conf['ts_params']['max_points_per_chunk']
        self.grid_size_levels = conf['ts_params']['grid_size_levels']

        logger.info('Aggregating ')


def get_binary_change(row, start_year, end_year, consider_tmf=True, gfc_gain=True):

    # extract change
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
    #print(len(y_train[y_train == 1]))
    #print(len(y_train[y_train == 0]))
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
        brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
        brf.fit(x_train, y_train)

    # predict probability
    class_probabilities = brf.predict_proba(x_df)
    return class_probabilities.T[class_prob]


def get_stats(row, band, start, end):
    idx = [start * 10000 < int(d) < end * 10000 for d in row.dates]
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
