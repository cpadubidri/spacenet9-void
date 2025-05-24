import sys
import argparse
import os
import rasterio
import numpy as np
from utils import resize_to_match_resolution, save_shift_map
from prediction import load_model, compute_shift_with_model

def main(optical_image, sar_image,  output_offset_image):
    print(f"optical_image: {os.path.exists(optical_image)}, sar_image: {os.path.exists(sar_image)}")

    #load images
    with rasterio.open(optical_image) as rgb_src:
        rgb_img = rgb_src.read().transpose(1, 2, 0)
        rgb_profile = rgb_src.profile

    with rasterio.open(sar_image) as sar_src:
        sar_img = sar_src.read(1)
        sar_profile = sar_src.profile

    #resize SAR to match RGB resolution
    sar_resized = resize_to_match_resolution(
        sar_img,
        current_res=sar_profile["transform"][0],
        target_res=rgb_profile["transform"][0]
    )

    #load model
    # model = load_model("model_weights.pth") 
    model = None

    #compute shift map using sliding window inference
    shift_map = compute_shift_with_model(model, sar_resized, rgb_img)

    #save the output as a GeoTIFF with dx and dy bands
    save_shift_map(shift_map, output_offset_image, rgb_profile)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trivial solution returning zero offset")
    parser.add_argument("--optical_image", type=str, help="Path to the optical image", required=True)
    parser.add_argument("--sar_image", type=str, help="Path to the SAR image", required=True)
    parser.add_argument("--output_offset_image", type=str, help="Path to the output offset image", required=True)
    args = parser.parse_args()

    main(args.optical_image, args.sar_image, args.output_offset_image)


# ./test.sh /home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/publictest/02_optical_publictest.tif /home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/publictest/02_sar_publictest.tif /home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void/deployment/solution/shift.tif