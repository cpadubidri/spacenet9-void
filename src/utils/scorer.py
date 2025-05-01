""" Evaluate an optical pixel offset image against geo-tiepoints csv """
import argparse
from typing import List, Tuple, Callable, Optional
import logging

import numpy as np
import pandas as pd
from osgeo import gdal, osr

gdal.UseExceptions()

from geo import translate_tiepoints_to_image_coords, get_tiepoint_distances, translate_tiepoints_to_geo_coords, ys_xs_to_shapely_points, transform_to_epsg, shapely_points_to_ys_xs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_tiepoints", type=str, help="filepath to the geo tiepoints csv", required=True)
    parser.add_argument("--output_offset_image", type=str, help="filepath to the pixel offset image", required=True)
    parser.add_argument("--optical_image", type=str, help="Path to the optical image", required=True)
    parser.add_argument("--sar_image", type=str, help="Path to the SAR image", required=True)
    parser.add_argument("--score_file", type=str, help="Path to the text file fot outputting the score", required=False)
    args = parser.parse_args()
    return args

def create_offset_map_from_reference(optical_image_filepath: str,
                                     sar_image_filepath: str,
                                     tiepoints_filepath: str):
    sar_ds = gdal.Open(str(sar_image_filepath))
    opt_ds = gdal.Open(str(optical_image_filepath))
    sar_geotran = list(sar_ds.GetGeoTransform())
    opt_geotran = list(opt_ds.GetGeoTransform())
    opt_srs = opt_ds.GetProjection()
    opt_wkt = osr.SpatialReference(opt_srs)
    opt_epsg = int(opt_wkt.GetAttrValue('authority', 1))
    opt_array = opt_ds.ReadAsArray()

    sar_srs = sar_ds.GetProjection()
    sar_wkt = osr.SpatialReference(sar_srs)
    sar_epsg = int(sar_wkt.GetAttrValue('authority', 1))
    
    sar_ds = None
    opt_ds = None

    df = pd.read_csv(tiepoints_filepath)
    sar_cols = df['sar_col'].to_list() # col in reference to original sar scene
    sar_rows = df['sar_row'].to_list() # row in reference to original sar scene
    opt_cols = df['optical_col'].to_list()
    opt_rows = df['optical_row'].to_list()

    sar_ys, sar_xs = translate_tiepoints_to_geo_coords(sar_rows, sar_cols, sar_geotran)
    opt_ys, opt_xs = translate_tiepoints_to_geo_coords(opt_rows, opt_cols, opt_geotran)
    
    if sar_epsg != opt_epsg:
        print("[INFO] => SAR coordinate reference system is not the same as optical coordinate reference system")
        points = ys_xs_to_shapely_points(sar_ys, sar_xs)
        transformed_points = transform_to_epsg(points, sar_epsg, opt_epsg)
        sar_ys, sar_xs = shapely_points_to_ys_xs(transformed_points)
    
    y_dif_geo = np.array(sar_ys) - np.array(opt_ys)
    x_dif_geo = np.array(sar_xs) - np.array(opt_xs)
    y_dif_pixels = np.rint(y_dif_geo / opt_geotran[5]).astype(int)
    x_dif_pixels = np.rint(x_dif_geo / opt_geotran[1]).astype(int)

    #NODATA_VALUE = 99999
    ref_shape = opt_array.shape
    nrows = ref_shape[1]
    ncols = ref_shape[2]
    #out = np.ones(shape=(2, nrows, ncols), dtype=np.int32) * NODATA_VALUE
    out = np.full((2, nrows, ncols), np.nan)
    for idx in range(len(opt_rows)):
        source_row = opt_rows[idx]
        source_col = opt_cols[idx]
        out[0, source_row, source_col] = x_dif_pixels[idx]
        out[1, source_row, source_col] = y_dif_pixels[idx]
    return out

def evaluate_solution(prediction_offset_array, optical_image_filepath, sar_image_filepath, tiepoints_filepath):
    reference_offset_array = create_offset_map_from_reference(optical_image_filepath, sar_image_filepath, tiepoints_filepath)
    match_size(prediction_offset_array, reference_offset_array)

    mask = ~np.isnan(reference_offset_array[0])

    ref_xs = reference_offset_array[0][mask]
    ref_ys = reference_offset_array[1][mask]

    pred_xs = prediction_offset_array[0][mask]
    pred_ys = prediction_offset_array[1][mask]

    dists = [np.sqrt((pred_xs[i] - ref_xs[i])**2 + (pred_ys[i] - ref_ys[i])**2) for i in range(len(ref_xs))]
    return dists

def match_size(array1, array2):
    shape1 = array1.shape
    nrows1 = shape1[1]
    ncols1 = shape1[2]
    shape2 = array2.shape*
    nrows2 = shape2[1]
    ncols2 = shape2[2]
    if nrows1 != nrows2 or ncols1 != ncols2:
        raise ValueError(f"Offset image dimensions ({ncols1}x{nrows1}) do not match optical image dimensions ({ncols2}x{nrows2})!")


if __name__ == "__main__":
    args = parse_args()
    
    offset_image_filepath = args.output_offset_image
    evaluation_tiepoints_csv = args.evaluation_tiepoints
    optical_image = args.optical_image
    sar_image = args.sar_image

    fmt = "%(asctime)s [%(levelname)s] => %(message)s"
    level = getattr(logging, "INFO", None)
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")

    raw_score = -1
    logging.info("Reading offset image")
    try:
        ds = gdal.Open(offset_image_filepath)
        offset_array = ds.ReadAsArray()
        ds = None
        
        if offset_array.ndim != 3:
            raise ValueError(f"The offset image must have 3 dimensions (2 bands x height x width), not {offset_array.ndim} dimensions!")
        if offset_array.shape[0] != 2:
            raise ValueError(f"The offset image does not have 2 bands, it has {offset_array.shape[0]} bands!")
        
        ds = gdal.Open(optical_image)
        opt_geotran = ds.GetGeoTransform()
        ds = None

        logging.info("Evaluating predicted pixel offset map to reference tiepoints")
        d = evaluate_solution(offset_array, optical_image, sar_image, evaluation_tiepoints_csv)
        
    except RuntimeError as e:
        logging.info(f"Error while reading offset image with gdal.Open(): {e}")
    except ValueError as e:
        logging.info(f"Offset image dimensions error: {e}")
    except Exception as e:
        logging.info(f"Unexpected error while reading offset image with gdal.Open(): {e}")
    else:
        logging.info(f"    mean euclidean distance between predicted offset map and reference tiepoints (meters): {np.mean(np.array(d) * opt_geotran[1])}")
        logging.info(f"    mean euclidean distance between predicted offset map and reference tiepoints (pixels): {np.mean(d)}")
        raw_score = np.mean(d)
        leaderboard_score = 100.0 / (1 + 0.01 * raw_score)
        logging.info(f"    leaderboard score: {leaderboard_score}")
    finally:
        if args.score_file is not None:
            with open(args.score_file, "a") as file:
                file.write(f"{raw_score}\n")
