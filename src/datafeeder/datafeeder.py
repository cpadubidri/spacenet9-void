import os
import random
import uuid
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import box
import numpy as np


class CModaldata(Dataset):
    def __init__(self, config, debug_vis=False):
        self.data_path = config.data["data_path"]
        self.data_len = config.data["data_length"] #this is not actual length of dataset here, but noof steps in training loop, because we are generating data on the fly


        #some important parameters
        self.target_size = 256
        self.crop_size_m = config.data.get("image_size") #this determines the size of crop in meter, we need to decide on image size in pixels
        self.margin_m = 20 #this is the margin in meter to avoid edge effects (i.e. black pixels on corners of sar images)
        self.dissimialr_dist = 300 #this will control the dissimilar pair (rgb,sar). The value should be choose not very small(close patch) or not very large (far patch). Target is hardnegatives

        #debug and visualization
        self.debug_vis = debug_vis 
        if self.debug_vis: #creates folder and saves crop rgb and sar image in geotiff format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.vis_dir = os.path.join(os.getcwd(), f"debug_vis_{timestamp}")
            os.makedirs(self.vis_dir, exist_ok=True)
        else:
            self.vis_dir = None

        #load list of image pairs
        data_list = pd.read_csv(os.path.join(self.data_path, 'data_patch.csv'))['data_list']
        self.pairs = []
        for filename in data_list:
            sar_path = os.path.join(self.data_path, filename.replace('source', 'sar') + '.tif')
            rgb_path = os.path.join(self.data_path, filename.replace('source', 'optical') + '.tif')
            if os.path.exists(sar_path) and os.path.exists(rgb_path):
                self.pairs.append((sar_path, rgb_path)) #create sar and rgb pair


        if not self.pairs: #check if rgb and sar all OK
            raise RuntimeError("No valid SAR-RGB pairs found.") 

    def __len__(self):
        return self.data_len #this is not actual length of dataset here, but noof steps in training loop, because we are generating data on the fly

    def __getitem__(self, index):
        is_similar = random.choice([True, False]) #randomly choose if the image pair is similar or not. This will be used as label and also to decide how to sample the images
        
        #get image pair and load
        pair_idx = random.randint(0, len(self.pairs) - 1) 
        sar_path, rgb_path = self.pairs[pair_idx] #select rgb and sar pair out of 3 pairs

        try:
            if self.debug_vis:
                print(f"File exists: {os.path.exists(sar_path)}, {os.path.exists(rgb_path)}")
                print(f"Loading {sar_path} and {rgb_path}")
            with rio.open(sar_path) as sar_ds, rio.open(rgb_path) as rgb_ds:
                sar_bounds = self.get_bounds(sar_ds) #>> use sar to get the bounds, because sar has zero padding
                rgb_bounds = rgb_ds.bounds

                if sar_bounds is None or rgb_bounds is None: #which is impossible, so raise error and stop
                    raise ValueError("Invalid bounds")
                
                minx = max(sar_bounds[0], rgb_bounds[0])
                miny = max(sar_bounds[1], rgb_bounds[1])
                maxx = min(sar_bounds[2], rgb_bounds[2])
                maxy = min(sar_bounds[3], rgb_bounds[3])

                #abort if they don’t overlap enough
                if minx >= maxx or miny >= maxy:
                    raise ValueError("No overlap between SAR and RGB after margins")
                
                if is_similar: #if label is 1, the sample the same point from both images
                    cx, cy = self.random_point((minx, miny, maxx, maxy), sar_ds)
                    sar_crop, sar_meta, sar_np = self.read_crop(sar_ds, cx, cy) #<<<<<<<<after debugging the image remove meta to make it faster>>>>>>>>
                    rgb_crop, rgb_meta, rgb_np = self.read_crop(rgb_ds, cx, cy)
                else: #if label is 0, sample two different points from the same image, hard-negative
                    #random point for sar
                    cx1, cy1 = self.random_point((minx, miny, maxx, maxy), sar_ds)
                    
                    #based on sar random point, we need to get the random point for rgb image
                    angle = random.uniform(0, 2 * np.pi) 
                    dx = self.dissimialr_dist * np.cos(angle)
                    dy = self.dissimialr_dist * np.sin(angle)

                    cx2 = cx1 + dx
                    cy2 = cy1 + dy #chance of going outside the rgb image is high.

                    #*************************#
                    '''
                    This will make sure the random point dont go outside the rgb image
                    '''
                    res_x, res_y = abs(rgb_ds.res[0]), abs(rgb_ds.res[1])
                    crop_px = int(self.crop_size_m / res_x)
                    crop_py = int(self.crop_size_m / res_y)
                    x_margin = crop_px // 2 * res_x
                    y_margin = crop_py // 2 * res_y

                    rgb_safe_bounds = self.get_bounds(rgb_ds)
                    if rgb_safe_bounds is None:
                        raise ValueError("Invalid RGB safe bounds.")

                    valid_minx = rgb_safe_bounds[0] + x_margin
                    valid_maxx = rgb_safe_bounds[2] - x_margin
                    valid_miny = rgb_safe_bounds[1] + y_margin
                    valid_maxy = rgb_safe_bounds[3] - y_margin

                    cx2 = max(valid_minx, min(valid_maxx, cx2))
                    cy2 = max(valid_miny, min(valid_maxy, cy2))
                    #*************************#

                    sar_crop, sar_meta, sar_np = self.read_crop(sar_ds, cx1, cy1)
                    rgb_crop, rgb_meta, rgb_np = self.read_crop(rgb_ds, cx2, cy2)

                           
                if sar_crop is None or rgb_crop is None: #if the crop is empty, raise error
                    raise ValueError("Empty crop detected.")
                
                #save crops as TIFFs for debug
                if self.debug_vis:
                    pair_id = uuid.uuid4().hex[:8]
                    self.save_crop(sar_np, sar_meta, int(is_similar), "sar", pair_id)
                    self.save_crop(rgb_np, rgb_meta, int(is_similar), "rgb", pair_id)

                #stack and return
                stacked = [rgb_crop, sar_crop]


                #pending: need augmentation here
                '''
                torch transformations - flip, rotate, color jitter. Dont resize or crop this will kill our sar,rgb pairing
                also let the sar and rgb be in different dimension, we will handle that in the model    
                '''

                return stacked, torch.tensor(int(is_similar), dtype=torch.long) #return the stacked image and the label
        

        except Exception as e:
            print(f"Error loading images: {e}")
            # return np.zeros((256,256,3)),np.zeros((1))

    def get_bounds(self, ds):
        bounds = ds.bounds
        safe = box(
            bounds.left + self.margin_m, 
            bounds.bottom + self.margin_m,
            bounds.right - self.margin_m,
            bounds.top - self.margin_m
        ) # subtract some margin around the sar bound because sar images have black pixels on corners
        return None if safe.is_empty else safe.bounds

    def random_point(self, bounds, ds): #get random point in the image (in pixel coordinates)
        minx, miny, maxx, maxy = bounds
        res_x, res_y = abs(ds.res[0]), abs(ds.res[1])  #get img resolution, it around 0.4m.
        crop_px = int(self.crop_size_m / res_x) #converting to pixel coordinates
        crop_py = int(self.crop_size_m / res_y)

        x_margin = crop_px // 2 * res_x
        y_margin = crop_py // 2 * res_y

        x = random.uniform(minx + x_margin, maxx - x_margin) #making sure the random point is not too close to the edge of the image
        y = random.uniform(miny + y_margin, maxy - y_margin)
        return x, y

    def read_crop(self, ds, cx, cy): # need some work here. This has some issue
        try:
            transform = ds.transform
            res_x, res_y = abs(ds.res[0]), abs(ds.res[1])
            crop_w = int(self.crop_size_m / res_x)
            crop_h = int(self.crop_size_m / res_y)

            px, py = ds.index(cx, cy)
            left = px - crop_w // 2
            top = py - crop_h // 2

            
            if left < 0 or top < 0 or (left + crop_w) > ds.width or (top + crop_h) > ds.height:
                return None, None, None

            window = Window(left, top, crop_w, crop_h)
            img = ds.read(window=window, boundless=False).astype(np.float32)

            if ds.dtypes[0] == 'uint8':
                img /= 255.0

            if img.shape[1] == 0 or img.shape[2] == 0:
                return None, None, None

            img_t = torch.from_numpy(img).float()
            # img_t = F.interpolate(img_t.unsqueeze(0), size=(self.target_size, self.target_size),
            #                       mode='bilinear', align_corners=False).squeeze(0)

            new_transform = rio.windows.transform(window, transform)
            meta = {
                "transform": new_transform,
                "crs": ds.crs,
                "count": img.shape[0],
                "dtype": img.dtype
            }
            return img_t, meta, img
        
        except Exception as e:
            print(f"[read_crop failed]: {e}")
            return None, None, None
    
    def save_crop(self, img_array, meta, label, band_type, pair_id):  # <- added pair_id
        if self.vis_dir is None:
            return

        file_id = f"{label}_{pair_id}_{band_type}.tif"  # consistent base name
        out_path = os.path.join(self.vis_dir, file_id)

        meta.setdefault("driver", "GTiff")
        meta["height"] = img_array.shape[1]
        meta["width"] = img_array.shape[2]
        meta.setdefault("count", img_array.shape[0])
        meta.setdefault("dtype", str(img_array.dtype))

        with rio.open(out_path, "w", **meta) as dst:
            dst.write(img_array)

        print(f"Saved {band_type.upper()} crop for label={label} at: {out_path}")
      


def get_dataloader(config, debug_vis=False):
    dataset = CModaldata(config, debug_vis)
    return DataLoader(dataset,
                      batch_size=config.data["batch_size"],
                      shuffle=True,
                      num_workers=config.data["num_workers"],
                      drop_last=True)



if __name__=="__main__":
    pass