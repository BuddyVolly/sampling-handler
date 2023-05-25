import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns

def sample_allocation(n, div, Nh, SDh, N):

    neyman = np.multiply(n, np.divide(np.multiply(Nh,SDh), div))
    proportional = np.multiply(n, np.divide(Nh,N))
    return int(neyman), int(proportional)


def bayesian_update(prior, likelihood):
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
    return np.sqrt(np.multiply(i, j))


def kmeans_stratifier(df, x, strata, sample_size):

    # add kmeans cluster result to df
    df['kmeans'] = KMeans(
        n_clusters=strata,
        n_init="auto"
    ).fit_predict(df[x].values.reshape(-1, 1))

    # prepare for
    sample_df = pd.DataFrame(
        [(
            i,
            len(df[df.kmeans == i]), 
            np.mean(df[x][df.kmeans == i]),
            np.std(df[x][df.kmeans == i]),
            len(df[df.kmeans == i])*np.std(df[x][df.kmeans == i])
        )
            for i in df.kmeans.unique()], columns=['KMeans', 'Nh', 'mean', 'SD', 'div'])
    
    sample_df.sort_values('mean', inplace=True)
    sample_df['Strata'] = [i + 1 for i in range(len(sample_df))]
    sample_df[['neyman', 'proportional']] = sample_df.apply(lambda row: sample_allocation(
        sample_size,
        sample_df['div'].sum(),
        row.Nh,
        row.SD,
        len(df)
    ), axis=1, result_type='expand')

    return df, sample_df


def plot_cluster(df, prob_column, strata_column):

    bounds = []
    for i in np.sort(df[strata_column].unique()):
        bounds.append(df[prob_column][df[strata_column] == i].max())

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    df.plot(prob_column, markersize=0.15, ax=ax[0], legend=True, cmap='magma')
    df.hist(prob_column, ax=ax[1])

    for bound in np.sort(bounds)[:len(df[strata_column].unique()) - 1]:
        ax[1].axvline(bound, color='orange')
