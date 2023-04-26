import warnings
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn import FunctionSampler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from skopt import BayesSearchCV


def get_binary_change(row, start_year, end_year, consider_tmf=True, gfc_gain=True):

    # extract change
    loss_year = np.nan_to_num(row.gfc_lossyear)
    gfc_loss = 1 if int(start_year[2:]) < int(row.gfc_lossyear) < int(end_year[2:]) else 0
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
        'n_estimators': [50, 100, 250, 500],
        'max_depth': [5, 10, 25, 50],
        'min_samples_split': [2, 5, 10, 25],
        'min_samples_leaf': [1, 3, 5, 10, 25]
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
