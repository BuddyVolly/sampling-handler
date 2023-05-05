""" TO DO DOC"""

from pathlib import Path
import os
import uuid
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import shapely
from shapely.geometry import box, Point

from ..esbae import Esbae
from ..misc import py_helpers, ee_helpers, config
from ..misc.settings import setup_logger


# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


class SampleDesign(Esbae):

    def __init__(self, project_name, shape, strategy, grid_crs, out_crs, aoi=None):

        # ------------------------------------------
        # 1 Get Generic class attributes
        super().__init__(project_name, aoi)

        # we need to get the AOI right with the CRS
        self.aoi = py_helpers.read_any_aoi_to_single_row_gdf(
            self.aoi, incrs=self.aoi_crs
        )

        # here is where out files are stored
        self.out_dir = str(Path(self.project_dir).joinpath('02_Sample_Design'))
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # TODO add point id to grid generation
        self.pid = self.config_dict['design_params']['pid']
        self.grid_shape = shape
        self.sampling_strategy = strategy
        self.grid_crs = grid_crs
        self.out_crs = out_crs

        # placeholders for grid generation
        self.squared_grid_size = None
        self.dggrid_resolution = None
        self.dggrid_projection = None

        # placeholders
        self.cell_grid = None
        self.points = None
        self.plot_figure = None
        self.ee_grid_fc = None
        self.ee_points_fc = None

    def generate_samples(self, upload_to_ee, save_as_ceo=None):

        # update configuration dict and file
        self.config_dict['design_params']['outdir'] = str(self.out_dir)
        self.config_dict['design_params']['sampling_strategy'] = self.sampling_strategy
        self.config_dict['design_params']['grid_shape'] = self.grid_shape
        self.config_dict['design_params']['grid_size'] = self.squared_grid_size
        self.config_dict['design_params']['grid_crs'] = self.grid_crs
        self.config_dict['design_params']['out_crs'] = self.out_crs
        self.config_dict['design_params']['ee_grid_fc'] = self.ee_grid_fc
        self.config_dict['design_params']['ee_samples_fc'] = self.ee_points_fc
        self.config_dict['design_params']['dggrid']['resolution'] = self.dggrid_resolution
        self.config_dict['design_params']['dggrid']['projection'] = self.dggrid_projection
        config.update_config_file(self.config_file, self.config_dict)

        if self.grid_shape == 'hexagonal':

            # create hex grid
            self.cell_grid, self.points = parallel_hexgrid(
                self.aoi, self.dggrid_resolution, self.sampling_strategy,
                self.dggrid_projection, self.grid_crs, self.out_crs
            )

            cell_name = (
                f'{self.sampling_strategy}_hex_'
                f'{self.dggrid_projection}_{self.dggrid_resolution}'
            )
            points_name = (
                f'{self.sampling_strategy}_samples_'
                f'{self.dggrid_projection}_{self.dggrid_resolution}'
            )

        if self.grid_shape == 'squared':
            self.cell_grid, self.points = squared_grid(
                self.aoi, self.squared_grid_size, self.sampling_strategy,
                self.grid_crs, self.out_crs
            )

            cell_name = f'{self.sampling_strategy}_squared_' \
                        f'{self.dggrid_projection}_{self.dggrid_resolution}'
            points_name = f'{self.sampling_strategy}_samples_' \
                          f'{self.dggrid_projection}_{self.dggrid_resolution}'

        # save files
        logger.info('Grid cells are saved locally.')
        ceo_file = Path(self.out_dir).joinpath(f'{points_name}.csv') if save_as_ceo else None
        py_helpers.save_gdf_locally(
            self.cell_grid, self.out_dir, gpkg=f'{cell_name}.gpkg'
        )
        logger.info('Point samples are saved locally as GeoPackage and CEO file.')
        py_helpers.save_gdf_locally(
            self.points, self.out_dir, gpkg=f'{points_name}.gpkg', ceo_csv=ceo_file
        )

        if upload_to_ee:
            logger.info('Grid cells are being uploaded to GEE. This may take a while...')
            self.ee_grid_fc = ee_helpers.export_to_ee(
                self.cell_grid, cell_name, self.project_name, 10000
            )
            logger.info('Point samples are being uploaded to GEE. This may take a while...')
            self.ee_points_fc = ee_helpers.export_to_ee(
                self.points, points_name, self.project_name
            )

        return self.cell_grid, self.points

    def plot_samples(self, save_figure=True):

        self.plot_figure = plot_samples(self.aoi, self.points, self.cell_grid)
        self.plot_figure.savefig(Path(self.out_dir).joinpath('grid.png'))


def random_point(geometry):
    """ TO DO DOC"""
    bounds = geometry.bounds
    while True:

        x = (bounds[2] - bounds[0]) * np.random.random_sample(1) + bounds[0]
        y = (bounds[3] - bounds[1]) * np.random.random_sample(1) + bounds[1]
        if Point(x, y).within(geometry):
            break

    return Point(x, y)


def squared_grid(
        aoi,
        spacing,
        sampling_strategy="centroid",
        grid_crs="ESRI:54017",
        out_crs="ESRI:54017"
):
    """ TO DO DOC"""

    logger.info("Reading AOI.")
    aoi = py_helpers.read_any_aoi_to_single_row_gdf(aoi, outcrs=grid_crs)

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
    logger.info("Creating squared grid...")
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
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=grid_crs)

    # add points
    logger.info("Placing samples within grid...")
    if sampling_strategy == "centroid":
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

    logger.info("Removing points outside AOI...")
    point_gdf = point_gdf[point_gdf.geometry.within(aoi.iloc[0]["geometry"])]

    logger.info(f"Final sampling grid consists of {len(point_gdf)} samples.")
    return grid_gdf.to_crs(out_crs), point_gdf.to_crs(out_crs)


def hexagonal_grid(
    aoi,
    resolution,
    sampling_strategy="centroid",
    projection="ISEA3H",
    centroid_crs='ESRI:54017',
    outcrs='EPSG:4326' ,
    grid_only=False
):
    """ TO DO DOC"""

    logger.debug("Reading AOI.")
    aoi = py_helpers.read_any_aoi_to_single_row_gdf(aoi, outcrs)

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

    logger.debug('Creating hexagonal grid...')
    # Create a list of lines
    lines = [
        'dggrid_operation GENERATE_GRID',
        f'dggs_type {projection}',
        f'dggs_res_spec {resolution}',
        'clip_subset_type SHAPEFILE',
        f'clip_region_files {tmp_extent}',
        'cell_output_type GEOJSON',
        f'cell_output_file_name {tmp_outfile}',
    ]

    # Open the file for writing
    with open(tmp_meta, 'w') as file:
        # Write each line to the file
        for line in lines:
            file.write(line + '\n')

    # run dggrid
    _ = py_helpers.run_command(f'dggrid {tmp_meta}', elapsed=False, stdout=False, stderr=False)

    # read dggrid output into geodataframe
    gdf = gpd.read_file(tmp_outfile.with_suffix('.geojson')).to_crs(outcrs)
    gdf['geometry'] = convert_3d_to_2d(gdf['geometry'])

    if grid_only:
        return gdf

    # remove temp files
    tmp_meta.unlink()
    tmp_extent.unlink()
    tmp_outfile.with_suffix('.geojson').unlink()

    if sampling_strategy == 'centroid':
        logger.debug('Creating centroids based on Behrmann\'s equal area projection...')
        # centroid calculation shall be done on projected CRS
        gdf = gdf.to_crs(centroid_crs)
        # get centroids
        gdf['sample_points'] = gdf.geometry.centroid

    elif sampling_strategy == 'random':
        # create rand points in each grid
        gdf['sample_points'] = gdf.geometry.apply(
            lambda shp: random_point(shp)
        )

    # add point id
    logging.debug('Adding a unique point ID...')
    gdf['point_id'] = [i for i in range(len(gdf.index))]

    # divide to grid and point df
    grid_gdf = gdf.drop(['sample_points'], axis=1).to_crs(outcrs)
    gdf['geometry'] = gdf['sample_points']
    point_gdf = gdf.drop(['sample_points'], axis=1).to_crs(outcrs)

    logging.debug('Remove points outside AOI...')
    point_gdf = point_gdf[point_gdf.geometry.within(aoi.iloc[0]['geometry'])]

    logging.debug(f'Sampling grid consists of {len(point_gdf)} points.')
    return grid_gdf.to_crs(outcrs), point_gdf.to_crs(outcrs)


def convert_3d_to_2d(geom):
    return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))


def parallel_hexgrid(
        aoi,
        resolution,  # refers to the resolution of the grid
        sampling_strategy='centroid',  # choices are 'random' or 'systematic'
        dggrid_proj="ISEA3H",
        centroid_crs="ESRI:54017",
        outcrs='EPSG:4326'
):

    logger.info('Creating hexagonal grid...')
    if resolution < 12:
        hexs, grids = hexagonal_grid(
            aoi, resolution, sampling_strategy, dggrid_proj, centroid_crs, outcrs
        )
        return hexs, grids

    # else we parallelize
    base_resolution = np.ceil(resolution / 2)
    base_hexs = hexagonal_grid(aoi, base_resolution, dggrid_proj, grid_only=True)

    aoi_gdf = py_helpers.read_any_aoi_to_single_row_gdf(aoi)
    base_hexs['geometry'] = [
        geom.buffer(0.1).intersection(aoi_gdf.geometry)[0]
        for geom in convert_3d_to_2d(base_hexs.geometry)
    ]

    dggrid_args = []
    for i, row in base_hexs.iterrows():
        dggrid_args.append([
            gpd.GeoDataFrame([base_hexs.loc[i]]).set_crs('EPSG:4326'),
            resolution, sampling_strategy, dggrid_proj, centroid_crs, outcrs,
        ])

    results = py_helpers.run_in_parallel(
        hexagonal_grid, dggrid_args, os.cpu_count(), 'processes'
    )

    hexs, samples = [], []
    for r in results:
        if r:
            hexs.append(r[0])
            samples.append(r[1])

    hexs = pd.concat(hexs)
    hexs.drop_duplicates('name', inplace=True)

    centroids = pd.concat(samples)
    centroids.drop_duplicates('name', inplace=True)

    # add point id
    logger.info("Adding a unique point ID...")
    centroids['point_id'] = [i for i in range(len(centroids))]

    logger.info(f"Final sampling design consists of {len(centroids)} samples.")
    return hexs, centroids


def plot_samples(aoi, sample_points, grid_cells=None, basemap=cx.providers.Esri.WorldImagery):
    """ TO DO DOC"""

    fig, ax = plt.subplots(1, 1, figsize=(25, 25))

    aoi = py_helpers.read_any_aoi_to_single_row_gdf(aoi, sample_points.crs)
    aoi.plot(ax=ax, alpha=0.25)

    if grid_cells is not None:
        grid_cells.plot(ax=ax, facecolor="none", edgecolor="black", lw=0.1)

    sample_points.plot(ax=ax, facecolor="red", markersize=0.5)
    cx.add_basemap(ax, crs=aoi.crs.to_string())
    ax.add_artist(ScaleBar(py_helpers.get_scalebar_distance(sample_points)))
    ax.set_title('Sample Design')
    return fig
