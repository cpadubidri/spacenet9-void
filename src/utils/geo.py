""" utility functions for geospatial data """
from typing import List, Tuple

import numpy as np
import shapely
from shapely import Point
from pyproj import CRS, Transformer
import pandas as pd
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.ops import transform
from osgeo import gdal, osr

def row_col_to_y_x(row: int,
                   col: int,
                   geotransform: Tuple[float, float, float, float, float]) -> Tuple[float, float]:
    """ translate row and column image coordinate to geographic coordinate based on the given geotransform """
    y = geotransform[3] + (row * geotransform[5])
    x = geotransform[0] + (col * geotransform[1])
    return y, x

def y_x_to_row_col(y: float,
                   x: float,
                   geotransform: Tuple[float, float, float, float, float]) -> Tuple[int, int]:
    """ translate y and x geo coordinates to row and column image coordinates based on the given geotransform """
    row = int((y - geotransform[3]) / geotransform[5])
    col = int((x - geotransform[0]) / geotransform[1])
    return row, col

def translate_tiepoints_to_geo_coords(tiepoints_row: List[int],
                                      tiepoints_col: List[int],
                                      geotransform: Tuple[float, float, float, float, float]) -> Tuple[List[float], List[float]]:
    """ translates list of tiepoint rows and cols (in image coordinates) to geo coordinates based on the given geotransform """
    xs = []
    ys = []
    for i in range(len(tiepoints_row)):
        y, x = row_col_to_y_x(tiepoints_row[i], tiepoints_col[i], geotransform)
        ys.append(y)
        xs.append(x)
    return ys, xs

def translate_tiepoints_to_image_coords(tiepoint_ys: List[float],
                                        tiepoint_xs: List[float],
                                        geotransform: Tuple[float, float, float, float, float]) -> Tuple[List[int], List[int]]:
    """ translates list of tiepoint geo coordinate xs and ys to image coordinates (rows and columns) based on give geotransform """
    rows = []
    cols = []
    for i in range(len(tiepoint_ys)):
        row, col = y_x_to_row_col(tiepoint_ys[i], tiepoint_xs[i], geotransform)
        rows.append(row)
        cols.append(col)
    return rows, cols

def ys_xs_to_shapely_points(ys: List[float], xs: List[float]) -> List[Point]:
    """ convert lists of y and x coordinates into a list of shapely points. """
    return [Point(xs[i], ys[i]) for i in range(len(ys))]

def shapely_points_to_ys_xs(points: List[Point]) -> Tuple[List[float], List[float]]:
    out = np.array([[i.y, i.x] for i in points])
    return list(out[:,0]), list(out[:,1])

def get_tiepoint_distances(sar_tiepoints_y: List[float],
                           sar_tiepoints_x: List[float],
                           opt_tiepoints_y: List[float],
                           opt_tiepoints_x: List[float]) -> np.ndarray:
    """ returns an array of values indicating the distance between each tiepoint """
    distances = []
    for i in range(len(sar_tiepoints_y)):
        sar_y, sar_x = sar_tiepoints_y[i], sar_tiepoints_x[i]
        opt_y, opt_x = opt_tiepoints_y[i], opt_tiepoints_x[i]
        d = shapely.distance(Point(sar_x,sar_y), Point(opt_x,opt_y))
        distances.append(d)
    return np.array(distances)

def _tiepoints_file_to_geo_coords(sar_image_filepath: str,
                            optical_image_filepath: str,
                            tiepoints_filepath: str,
                            output_tiepoints_filepath: str):
    """ translate tiepoints file csv from row/col to y/x coords. """
    sar_ds = gdal.Open(str(sar_image_filepath))
    opt_ds = gdal.Open(str(optical_image_filepath))
    sar_geotran = list(sar_ds.GetGeoTransform())
    opt_geotran = list(opt_ds.GetGeoTransform())
    
    im_srs = opt_ds.GetProjection()
    srs_wkt = osr.SpatialReference(im_srs)
    opt_epsg = int(srs_wkt.GetAttrValue('authority', 1))

    im_srs = sar_ds.GetProjection()
    srs_wkt = osr.SpatialReference(im_srs)
    sar_epsg = int(srs_wkt.GetAttrValue('authority', 1))

    df = pd.read_csv(tiepoints_filepath)
    sar_row = df['sar_row'].to_list()
    sar_col = df['sar_col'].to_list()
    opt_row = df['optical_row'].to_list()
    opt_col = df['optical_col'].to_list()
    sar_ys, sar_xs = translate_tiepoints_to_geo_coords(sar_row, sar_col, sar_geotran) # translate row/col to geo coords based on the respective image's geotransform
    opt_ys, opt_xs = translate_tiepoints_to_geo_coords(opt_row, opt_col, opt_geotran)

    #assert(sar_epsg == opt_epsg), f"sar_epsg != opt_epsg. {sar_epsg}!={opt_epsg}"
    if sar_epsg != opt_epsg: # reproject optical x/y coords to the same coordinate reference system as the sar x/y coords.
        points = [Point(opt_xs[i], opt_ys[i]) for i in range(len(opt_ys))]
        new_geoms = transform_to_epsg(points, opt_epsg, sar_epsg)
        opt_xs = [p.x for p in new_geoms]
        opt_ys = [p.y for p in new_geoms]

    df = pd.DataFrame.from_dict({"sar_x":sar_xs,
                                 "sar_y":sar_ys,
                                 "optical_x":opt_xs,
                                 "optical_y":opt_ys})
    df.to_csv(output_tiepoints_filepath, index=False)
    print(df.head())

def write_tiff(array: np.ndarray,
               geotran: Tuple[float, float, float, float, float],
               epsg: int,
               out_filepath: str,
               dtype = gdal.GDT_Byte,
               no_data_value = None) -> None:
    """ write and save a new GeoTIFF to out_filepath location """
    driver = gdal.GetDriverByName("GTiff")
    
    arr_shape = array.shape
    if array.ndim == 3:
        bands = arr_shape[0]
        rows = arr_shape[1]
        cols = arr_shape[2]
    else:
        bands = 1
        rows = arr_shape[0]
        cols = arr_shape[1]

    out_ds = driver.Create(out_filepath, cols, rows, bands, dtype)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromEPSG(epsg)

    out_ds.SetGeoTransform(geotran)
    out_ds.SetProjection(raster_srs.ExportToWkt())
    band_idx = 0
    while band_idx < bands:
        outband = out_ds.GetRasterBand(band_idx + 1)
        if bands == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[band_idx])
        if no_data_value:
            outband.SetNoDataValue(no_data_value)
        outband.FlushCache()
        band_idx += 1
    out_ds = None

def transform_to_epsg(geometries: List[shapely.Geometry],
                      source_epsg: int,
                      destination_epsg: int):
    """Parameters
    ----------
    geometries : list
        a list of shapely.Geometry objects to transform
    source_epsg : int
         the source epsg. the given geometries are currently projected into this EPSG
    destination_epsg : int
        the epsg to transform geometries into.

    Returns
    -------
    out_geometries : list
        a list of shapely.Geometry objects. these are geometries transformed into the destination_epsg

    """
    project = get_transform(source_epsg, destination_epsg)
    out_geometries = []
    for idx, geometry in enumerate(geometries):
        transformed_geom = transform(project, geometry)
        out_geometries.append(transformed_geom)
    return out_geometries


def get_transform(source_epsg: int,
                  destination_epsg: int):
    """Parameters
    ----------
    source_epsg : int
         the source epsg. the given geometries are currently projected into this EPSG
    destination_epsg : int
        the epsg to transform geometries into.

    Returns
    -------
    project :
        the pyroj transform object to use for reprojecting geometries from source_epsg to destination_epsg

    """
    if source_epsg is None:
        raise ValueError(
            ("Unable to perform transformation. " "A source EPSG for this ENC cell is not defined.")
        )
    crs_source = CRS.from_epsg(source_epsg)
    crs_target = CRS.from_epsg(destination_epsg)

    project = Transformer.from_crs(crs_source, crs_target, always_xy=True).transform
    return project