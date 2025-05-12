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
        self.crop_size_m = config.data.get("image_size") #this determines the size of crop in meter, we need to decide on image size in pixels
        self.margin_m = 200 #this is the margin in meter to avoid edge effects (i.e. black pixels on corners of sar images)
        

        #debug and visualization
        self.debug_vis = debug_vis 
        if self.debug_vis: #creates folder and saves crop rgb and sar image in geotiff format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.vis_dir = os.path.join(os.getcwd(), f"debug_vis_{timestamp}")
            os.makedirs(self.vis_dir, exist_ok=True)
        else:
            self.vis_dir = None

        # Load list of image pairs
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
        is_similar = random.choice([True, True]) #randomly choose if the image pair is similar or not. This will be used as label and also to decide how to sample the images
        
        #get image pair and load
        pair_idx = random.randint(0, len(self.pairs) - 1) 
        sar_path, rgb_path = self.pairs[pair_idx] #select rgb and sar pair out of 3 pairs





        return np.zeros((256,256,3)),np.zeros((1))

def get_dataloader(config, debug_vis=False):
    dataset = CModaldata(config, debug_vis)
    return DataLoader(dataset,
                      batch_size=config.data["batch_size"],
                      shuffle=True,
                      num_workers=config.data["num_workers"],
                      drop_last=True)



if __name__=="__main__":
    pass