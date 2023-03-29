import numpy as np
import dask_geopandas as dgpd


def sfc_subsample(gdf, target_point_size, seed=None):
    """Space-filling curve

    This function takes a GeoDataFrame with Point geometries,
    and will return a
    """

    # Check if all geometries are points
    is_only_points = all(geom.type == "Point" for geom in gdf.geometry)

    # Raise an error if not all geometries are points
    if not is_only_points:
        raise ValueError("GeoDataFrame contains non-point geometries")

    # create hilbert curve and sort by distance
    dgdf = dgpd.from_geopandas(gdf, npartitions=4)
    gdf["dist"] = dgdf.geometry.hilbert_distance()
    gdf = gdf.sort_values("dist").reset_index()

    # get index and taol population size from which to susample
    idx = gdf.index.values
    initial_points_size = len(gdf)

    # get a division factor (sort of the equivalent to each nth point, but as float)
    division_factor = target_point_size / initial_points_size

    ### RANDOM INITIALIZATION ###
    # a list of points for random start, determined by target and total initial size
    max_start = np.ceil(initial_points_size / target_point_size)

    # create a list from 0 to max_start
    start_list = range(int(max_start))

    # define a seed for re-production
    if seed:
        np.random.seed(seed)

    # randomly select a sample for start
    start_sample = np.random.choice(start_list)

    ### FINAL SELECTION PROCESS ###
    # some initialization
    to_include, ceil = [], 0

    # we need a new starting point after random initialization, as it cannot be i
    j = 1
    # loop over all points
    for i in range(initial_points_size):

        # skip until randomly start point
        if i < start_sample:
            continue

        # here we convert the floating division factor into integer
        # the first sample will always be included as ceil is 0, j is 1, and division factor is greater than 0
        # so 0 != to soemthing > 0
        if ceil != np.ceil(j * division_factor):

            # we update the ceil for the next selection
            ceil = np.ceil(j * division_factor)

            # append sample
            to_include.append(idx[i])

        j += 1

    return gdf.loc[to_include]
