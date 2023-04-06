import numpy as np


def cusum_calculation(residuals):
    # do cumsum calculation
    cumsum = np.nancumsum(residuals, axis=0)
    s_max = np.nanmax(cumsum, axis=0)
    s_min = np.nanmin(cumsum, axis=0)
    s_diff = s_max - s_min

    # get podition of max value
    argmax = np.argmax(cumsum, axis=0)

    return s_diff, argmax


def bootstrap_cusum(stack, s_diff, nr_bootstraps):

    # intialize iteration variables
    i, comparison_array, change_sum = 0, np.zeros(s_diff.shape), np.zeros(s_diff.shape)

    while i < nr_bootstraps:

        # shuffle first axis
        shuffled_index = np.random.choice(
            stack.shape[0],
            size=stack.shape[0],
            replace=False
        )

        # run cumsum on re-shuffled stack
        s_diff_bs, _ = cusum_calculation(stack[shuffled_index])

        # compare if s_diff_bs is greater and sum up
        comparison_array += np.greater(s_diff, s_diff_bs).astype("float32")

        # sum up random change magnitude s_diff_bs
        change_sum += s_diff_bs

        # set counter
        i += 1

    # calculate final confidence and significance
    confidences = np.divide(comparison_array, nr_bootstraps, out=np.zeros_like(comparison_array), where=nr_bootstraps!=0)
    signficance = 1 - np.divide(np.divide(change_sum, nr_bootstraps), s_diff, out=np.zeros_like(s_diff), where=s_diff!=0)

    # calculate final confidence level
    change_point_confidence = np.multiply(confidences, signficance)

    return change_point_confidence


def cusum_deforest(data, dates, point_id, nr_bootstraps):
    """
    Calculates Page's Cumulative Sum Test according to NASA SERVIR Handbook's implementation
    """

    if data:
        # calculate residuals (broadcasting here)
        residuals = np.subtract(data, np.nanmean(data))

        # get original cumsum caluclation and dates
        magnitude, argmax = cusum_calculation(residuals)

        # get dates into change array
        date = dates[argmax]

        # get confidence from bootstrap procedure
        confidence = bootstrap_cusum(residuals, magnitude, nr_bootstraps)

        return date, confidence, magnitude, point_id

    else:
        return -1, -1, -1, point_id
