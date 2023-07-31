import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def sample_allocation(n, div, Nh, SDh, N):
    """Allocates a sample size to a stratum using Neyman and proportional allocation.

    Args:
        n (int): The total sample size.
        div (float): The sum of the product of stratum size and stratum standard deviation.
        Nh (int): The size of the stratum.
        SDh (float): The standard deviation of the stratum.
        N (int): The total population size.

    Returns:
        tuple: A tuple containing the Neyman allocation and the proportional allocation.

    """
    neyman = np.multiply(n, np.divide(np.multiply(Nh,SDh), div))
    proportional = np.multiply(n, np.divide(Nh,N))
    return int(neyman), int(proportional)


def bayesian_update(prior, likelihood):
    """Updates a prior probability using Bayes' theorem.

    Args:
        prior (float): The prior probability.
        likelihood (float): The likelihood of the evidence.

    Returns:
        float: The updated probability.

    """
    fac = np.multiply(prior, likelihood)
    return np.divide(
        fac, np.add(
            fac,
            np.multiply(
                np.subtract(1, prior),
                np.subtract(1, likelihood)
            )
        )
    )


def combine_probabilities(i, j):
    """Combines two probabilities using the square root of their product.

    Args:
        i (float): The first probability.
        j (float): The second probability.

    Returns:
        float: The combined probability.

    """
    return np.sqrt(np.multiply(i, j))


def kmeans_stratifier(df, x, strata, sample_size, random_state=42):
    """Stratifies a continous variable using KMeans clustering.

    Args:
        df (pandas.DataFrame): The input dataframe.
        x (str): The column name of the continous variable to use for clustering.
        strata (int): The number of strata to create.
        sample_size (int): The desired sample size.
        random_state (int): The random state to use for reproducibility.

    Returns:
        tuple: A tuple containing the stratified dataframe and a summary dataframe.

    """
    # add kmeans cluster result to df
    df['stratum'] = KMeans(
        n_clusters=strata,
        n_init="auto",
        random_state=random_state
    ).fit_predict(df[x].values.reshape(-1, 1))

    # prepare for
    sample_df = pd.DataFrame(
        [(
            i,
            len(df[df.stratum == i]),
            np.mean(df[x][df.stratum == i]),
            np.std(df[x][df.stratum == i]),
            len(df[df.stratum == i])*np.std(df[x][df.stratum == i])
        )
            for i in df.stratum.unique()], columns=['KMeans', 'Nh', 'mean', 'SD', 'div'])

    sample_df.sort_values('mean', inplace=True)
    sample_df['stratum'] = [i + 1 for i in range(len(sample_df))]
    sample_df[['neyman', 'proportional']] = sample_df.apply(lambda row: sample_allocation(
        sample_size,
        sample_df['div'].sum(),
        row.Nh,
        row.SD,
        len(df)
    ), axis=1, result_type='expand')

    # replace KMeans with Strata
    df['stratum'].replace(sample_df.KMeans.tolist(), sample_df.stratum.tolist(), inplace=True)
    sample_df.drop(['KMeans'], axis=1, inplace=True)

    return df, sample_df


def plot_cluster(df, prob_column, strata_column, bins=10):
    """Plots a scatter plot and histogram of a probability column, stratified by a strata column.

    Args:
        df (pandas.DataFrame): The input dataframe.
        prob_column (str): The column name of the probability column to plot.
        strata_column (str): The column name of the strata column to use for stratification.
        bins (int): The number of bins to use for the histogram.

    Returns:
        None
    """
    bounds = []
    for i in np.sort(df[strata_column].unique()):
        bounds.append(df[prob_column][df[strata_column] == i].max())

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    df.plot(prob_column, markersize=0.15, ax=ax[0], legend=True, cmap='magma')
    df.hist(prob_column, ax=ax[1], bins=bins)

    for bound in np.sort(bounds)[:len(df[strata_column].unique()) - 1]:
        ax[1].axvline(bound, color='orange')
