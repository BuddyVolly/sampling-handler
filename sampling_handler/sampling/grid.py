""" TO DO DOC"""

from pathlib import Path
import uuid
import logging

import ee
import geemap
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.geometry import box, Point

from ..misc.py_helpers import run_command, read_any_aoi_to_single_row_gdf
from ..misc.settings import setup_logger


# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


def random_point(geometry):
    """ TO DO DOC"""
    bounds = geometry.bounds
    while True:

        x = (bounds[2] - bounds[0]) * np.random.random_sample(1) + bounds[0]
        y = (bounds[3] - bounds[1]) * np.random.random_sample(1) + bounds[1]
        if Point(x, y).within(geometry):
            break

    return Point(x, y)


def squared_grid(aoi, spacing, crs="ESRI:54017", sampling_strategy="systematic"):
    """ TO DO DOC"""

    logger.info("Reading AOI.")
    aoi = read_any_aoi_to_single_row_gdf(aoi, crs)

    # get bounds
    bounds = aoi.bounds

    # get origin point
    originx = bounds.minx.values[0]
    originy = bounds.miny.values[0]

    # get widht and height of aoi bounds
    width = bounds.maxx - bounds.minx
    height = bounds.maxy - bounds.miny

    # calculate how many cols and row are those
    columns = int(np.floor(float(width) / spacing))
    rows = int(np.floor(float(height) / spacing))

    # create grid cells
    logger.info("Creating grid cells.")
    i, _list = 1, []
    for column in range(0, columns + 1):
        x = originx + (column * spacing)
        for row in range(0, rows + 1):
            y = originy + (row * spacing)
            cell = box(x, y, x + spacing, y + spacing)
            if cell.intersects(aoi.iloc[0]["geometry"]):
                _list.append(cell)
                i += 1

    # and turn into geodataframe
    logger.debug("Turning grid cells into GeoDataFrame...")
    df = pd.DataFrame(_list, columns=["geometry"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    # add points
    logger.info("Creating sampling points.")
    if sampling_strategy == "systematic":
        # take centroid
        gdf["sample_points"] = gdf.geometry.centroid

    elif sampling_strategy == "random":
        # create rand points in each grid
        gdf["sample_points"] = gdf.geometry.apply(lambda shp: random_point(shp))

    # add point id
    logger.info("Adding a unique point ID...")
    gdf["point_id"] = [i for i in range(len(gdf.index))]

    # divide to grid and point df
    grid_gdf = gdf.drop(["sample_points"], axis=1)
    gdf["geometry"] = gdf["sample_points"]
    point_gdf = gdf.drop(["sample_points"], axis=1)

    logger.info("Remove points outside AOI...")
    point_gdf = point_gdf[point_gdf.geometry.within(aoi.iloc[0]["geometry"])]

    logger.info(f"Final sampling grid consists of {len(point_gdf)} points.")
    return grid_gdf, point_gdf


def hexagonal_grid(
    aoi,
    resolution,
    sampling_strategy="systematic",
    centroid_crs='ESRI:54017',
    outcrs='EPSG:4326' ,
    projection="ISEA3H",
    grid_only=False,

):
    """ TO DO DOC"""

    logger.info("Reading AOI.")
    aoi = read_any_aoi_to_single_row_gdf(aoi, outcrs)

    # create a unique id
    uuid_str = str(uuid.uuid4())

    # decalaring tmp file paths
    tmp_folder = Path.home().joinpath("tmp")
    tmp_meta = tmp_folder.joinpath(f"tmp_meta_{uuid_str}")
    tmp_extent = tmp_folder.joinpath(f"tmp_extent_{uuid_str}.shp")
    tmp_outfile = tmp_folder.joinpath(f"tmp_file_{uuid_str}")

    # force lat/lon for dggrid
    aoi = aoi.to_crs("EPSG:4326")
    aoi.to_file(tmp_extent)

    logger.info("Creating hexagonal grid...")
    # Create a list of lines
    lines = [
        "dggrid_operation GENERATE_GRID",
        f"dggs_type {projection}",
        f"dggs_res_spec {resolution}",
        "clip_subset_type SHAPEFILE",
        f"clip_region_files {tmp_extent}",
        "cell_output_type GEOJSON",
        f"cell_output_file_name {tmp_outfile}",
    ]

    # Open the file for writing
    with open(tmp_meta, "w") as file:
        # Write each line to the file
        for line in lines:
            file.write(line + "\n")

    # run dggrid
    _ = run_command(f"dggrid {tmp_meta}", stdout=False, stderr=False)

    # read dggrid output into geodataframe
    gdf = gpd.read_file(tmp_outfile.with_suffix(".geojson")).to_crs(outcrs)

    if grid_only:
        return gdf

    # remove temp files
    tmp_meta.unlink()
    tmp_extent.unlink()
    tmp_outfile.with_suffix(".geojson").unlink()

    if sampling_strategy == "systematic":
        logger.info("Creating centroids based on Behrmann's equal area projection...")
        # centroid calculation shall be done on projected CRS
        gdf = gdf.to_crs(centroid_crs)
        # get centroids
        gdf["sample_points"] = gdf.geometry.centroid

    elif sampling_strategy == "random":
        # create rand points in each grid
        gdf["sample_points"] = gdf.geometry.apply(
            lambda shp: random_point(shp)
        )

    # add point id
    logging.info("Adding a unique point ID...")
    gdf["point_id"] = [i for i in range(len(gdf.index))]

    # divide to grid and point df
    grid_gdf = gdf.drop(["sample_points"], axis=1).to_crs(outcrs)
    gdf["geometry"] = gdf["sample_points"]
    point_gdf = gdf.drop(["sample_points"], axis=1).to_crs(outcrs)

    logging.info("Remove points outside AOI...")
    point_gdf = point_gdf[point_gdf.geometry.within(aoi.iloc[0]["geometry"])]

    logging.info(f"Sampling grid consists of {len(point_gdf)} points.")
    return grid_gdf.to_crs(outcrs), point_gdf.to_crs(outcrs)


def plot_samples(aoi, sample_points, grid_cells=None):
    """ TO DO DOC"""

    fig, ax = plt.subplots(1, 1, figsize=(25, 25))

    aoi = read_any_aoi_to_single_row_gdf(aoi, sample_points.crs)
    aoi.plot(ax=ax, alpha=0.25)

    if grid_cells is not None:
        grid_cells.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.1)

    sample_points.plot(ax=ax, facecolor="red", markersize=0.5)
