import time
import logging

import ee
import geemap
import geopandas as gpd

from ..misc.py_helpers import split_dataframe
from ..misc.settings import setup_logger

# Create a logger object
logger = logging.getLogger(__name__)
setup_logger(logger)


def processing_grid(aoi, grid_size):

    boundbox = (
        aoi.geometry().bounds().buffer(distance=1, proj=ee.Projection("EPSG:4326"))
    )

    # return the list of coordinates
    list_cords = ee.Array.cat(boundbox.coordinates(), 1)

    # get the X and Y -coordinates
    x_cords = list_cords.slice(1, 0, 1)
    y_cords = list_cords.slice(1, 1, 2)

    # reduce the arrays to find the max (or min) value
    x_min = x_cords.reduce("min", [0]).get([0, 0])
    x_max = x_cords.reduce("max", [0]).get([0, 0])
    y_min = y_cords.reduce("min", [0]).get([0, 0])
    y_max = y_cords.reduce("max", [0]).get([0, 0])

    xx = ee.List.sequence(
        x_min, ee.Number(x_max).subtract(ee.Number(grid_size).multiply(0.9)), grid_size
    )
    yy = ee.List.sequence(
        y_min, ee.Number(y_max).subtract(ee.Number(grid_size).multiply(0.9)), grid_size
    )

    def map_over_x(_x):
        def map_over_y(_y):
            x1 = ee.Number(_x)
            x2 = ee.Number(_x).add(ee.Number(grid_size))
            y1 = ee.Number(_y)
            y2 = ee.Number(_y).add(ee.Number(grid_size))

            cords = ee.List([x1, y1, x2, y2])
            rect = ee.Algorithms.GeometryConstructors.Rectangle(
                cords, "EPSG:4326", False
            )
            return ee.Feature(rect)

        return yy.map(map_over_y)

    cells = xx.map(map_over_x).flatten()

    return ee.FeatureCollection(cells).filterBounds(aoi)


def get_random_point(feature, seed=None):

    feat = ee.Feature(feature)
    process_dict = {"region": feat.geometry(), "points": 1, "maxError": 100}
    if seed:
        process_dict.update(seed=seed)

    point = ee.Feature(
        ee.FeatureCollection.randomPoints(**process_dict).first()
    ).set("point_id", feat.id())

    return point.set("LON", point.geometry().coordinates().get(0)).set(
        "LAT", point.geometry().coordinates().get(1)
    )


def get_center_point(feature):
    feat = ee.Feature(feature)
    point = feat.centroid(10).set("point_id", feat.id())
    return point.set("LON", point.geometry().coordinates().get(0)).set(
        "LAT", point.geometry().coordinates().get(1)
    )


def set_id(feature):
    point = feature.set("point_id", feature.id())
    return point.set("LON", point.geometry().coordinates().get(0)).set(
        "LAT", point.geometry().coordinates().get(1)
    )


def _ee_export_table(ee_fc, description, asset_id, sub_folder=None, wait_until_end=True):

    # get users asset root
    asset_root = ee.data.getAssetRoots()[0]["id"]

    # if there is any subfolder,create it
    if sub_folder:
        try:
            # create temporary folder
            ee.data.createAsset(
                {"type": "folder"}, f"{asset_root}/{sub_folder}"
            )
        except ee.EEException as e:  # already exists
            if "Cannot overwrite asset subfolder" in str(e):
                logger.debug(str(e))
                pass

        asset_root = f"{asset_root}/{sub_folder}"

    # final asset name
    asset_id = f"{asset_root}/{asset_id}"

    # check if asset already exists
    for asset in ee.data.listAssets({"parent": asset_root})["assets"]:
        if asset["name"] == asset_id:
            logger.warning("Feature Collection asset already exists.")
            return 'exists', asset_id

    # create and start export task
    export_task = ee.batch.Export.table.toAsset(
        collection=ee_fc,
        description=description,
        assetId=asset_id,
    )
    export_task.start()

    if wait_until_end:
        check_finished_tasks([export_task], 5)

    return export_task, asset_id


def check_finished_tasks(tasks, wait=30):

    # check on status
    finished = False
    while not finished:
        time.sleep(wait)
        for task in tasks:
            state = task.status()['state']
            if state == 'COMPLETED':
                finished = True
            elif state in ['FAILED', 'CANCELLED']:
                if task.status()['error_message'] == 'Table is empty.':
                    finished = True
                else:
                    raise RuntimeError('Upload failed')
            elif state in ['UNSUBMITTED', 'SUBMITTED', 'READY', 'RUNNING', 'CANCEL_REQUESTED']:
                finished = False
                break

    return


def delete_sub_folder(sub_folder):

    # get users asset root
    asset_root = ee.data.getAssetRoots()[0]["id"]
    asset_folder = f"{asset_root}/{sub_folder}"

    logger.info(f"Removing assets within asset folder {asset_folder}")
    child_assets = ee.data.listAssets({"parent": f"{asset_folder}"})["assets"]
    for i, ass in enumerate(child_assets):
        logger.debug(f'Removing asset {ass["id"]} within {asset_folder}')
        ee.data.deleteAsset(ass["id"])

    logger.info(f"Removing asset folder {asset_folder}.")
    ee.data.deleteAsset(f"{asset_root}/{sub_folder}")


def cleanup_tmp_esbae():

    asset_root = ee.data.getAssetRoots()[0]["id"]
    for a in ee.data.listAssets({"parent": f"{asset_root}"})["assets"]:
        if a["name"].split('/')[-1][:9] == "tmp_esbae":
            delete_sub_folder(a["name"].split('/')[-1])


def merge_fcs(sub_folder):

    # get users asset root
    asset_root = ee.data.getAssetRoots()[0]["id"]

    logger.info(f"Merging the Feature Collections within {sub_folder}")
    child_assets = ee.data.listAssets({"parent": f"{asset_root}/{sub_folder}"})["assets"]
    for i, ass in enumerate(child_assets):
        # aggregate collections
        coll = ee.FeatureCollection(ass["id"])
        ee_fc = coll if i == 0 else ee_fc.merge(coll)

    return ee_fc


def export_to_ee(gdf, asset_name, ee_sub_folder=None, chunk_size=25000):
    """Function to export any kind of GeoDataFrame or temporary Feature Collection
    as an Earth Engine asset



    """

    # create a unique id
    gmt = time.strftime("%y%m%d_%H%M%S", time.gmtime())

    # if it is already a feature collection
    # (e.g. a gdf turned into a fc by geemap beforehand)
    if isinstance(gdf, ee.FeatureCollection):

        logger.info(f"Exporting Feature Collection {gdf}")
        _, asset_id = _ee_export_table(
            gdf=gdf,
            description=f"esbae_table_upload_{gmt}",
            asset_id=asset_name
        )
        return asset_id

    # check if input is a geodataframe
    elif isinstance(gdf, gpd.geodataframe.GeoDataFrame):
        pass
    else:
        raise ValueError(
            "Input must be a either a ee.FeatureCollection or a gpd.GeoDataFrame object"
        )

    if len(gdf) > chunk_size:

        logger.info("Need to run splitted upload routine as the dataframe has more than 25000 rows")
        #
        tmp_folder = f"tmp_esbae_{gmt}"
        logger.debug(f"Creating a temporary asset folder {tmp_folder}")

        # upload chunks of data to avoid hitting upload limitations
        logger.info(f"Uploading chunks of the dataframe into temporary assets")
        chunks, tasks = split_dataframe(gdf, chunk_size=chunk_size), []
        for i, chunk in enumerate(chunks):

            # turn gdf into Feature Collection
            chunk_fc = geemap.geopandas_to_ee(chunk)

            # run upload
            export_task, _ = _ee_export_table(
                ee_fc=chunk_fc,
                description=f"part_{i}_esbae_table_upload_{gmt}",
                asset_id=f"table_part_{i}",
                sub_folder=tmp_folder,
                wait_until_end=False
            )
            tasks.append(export_task)

        # wait until all tasks have been finished
        check_finished_tasks(tasks)

        # merge assets
        logger.info('Aggregating the temporary assets')
        ee_fc = merge_fcs(tmp_folder)

        logger.debug(f"Exporting aggregated assets to final EE asset {asset_name}")
        _, asset_id = _ee_export_table(
            ee_fc=ee_fc,
            description=f"esbae_final_table_upload_{gmt}",
            asset_id=asset_name,
            sub_folder=ee_sub_folder
        )

        delete_sub_folder(tmp_folder)
        logger.info(f"Upload completed. You can find the samples at {asset_name}")
    else:

        # turn into FC
        ee_fc = geemap.geopandas_to_ee(gdf.to_crs("EPSG:4326"))

        logger.debug(f"Exporting GeoDataFrame to Earth Engine asset {asset_name}")
        _, asset_id = _ee_export_table(
            ee_fc=ee_fc,
            description=f"esbae_upload_table_{gmt}",
            asset_id=asset_name,
            sub_folder=ee_sub_folder
        )
        logger.debug("Successfully exported table as an Earth Engine asset.")

    return asset_id
