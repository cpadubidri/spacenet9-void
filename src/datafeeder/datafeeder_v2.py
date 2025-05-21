import os
import random
import pandas as pd
import cv2
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ContrastiveDatafeeder(Dataset):
    def __init__(self, config, mode="Train", debug=False):
        self.debug = debug
        self.data_path = config.data["data_path"]
        self.image_size = config.data["image_size"]
        self.min_offset = 50 #>> this value is important, it decide hard negatives
        

        #load list of image pairs
        data_list = pd.read_csv(os.path.join(self.data_path, 'file_list.csv'))['file_name']
        self.pairs = []
        for filename in data_list:
            sar_path = os.path.join(self.data_path, filename.replace('source', 'sar'))
            rgb_path = os.path.join(self.data_path, filename.replace('source', 'optical'))
            if os.path.exists(sar_path) and os.path.exists(rgb_path):
                self.pairs.append((sar_path, rgb_path))

        #transformations (tensor, normalization) [The meand and std is calculated from the dataset, check prepare.ipynb]
        self.sar_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[40.2317/255.0], std=[37.8343/255.0])
        ])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[103.28885539/255.0, 111.16151289/255.0, 120.56394087/255.0], std=[65.82869639/255.0, 70.8654012/255.0, 77.10939367/255.0])
        ])

        self.mode = mode
        # print(self.mode)
        if mode == 'valid':
            # random.seed(42)
            # np.random.seed(42)
            self.valid_pts = []
            num_samples = config.data["val_length"]

            h, w = 1310, 1310

            for _ in range(num_samples):
                if random.random() < 0.5:
                    cx, cy = self.random_point_near_center(h, w, delta=256)
                    self.valid_pts.append((cx, cy, cx, cy, 1))  # label 1
                else:
                    cx1, cy1 = self.random_point_near_center(h, w, delta=256)
                    # cx2, cy2 = self.random_point_near_center(h, w, delta=400)
                    # min_offset = 32
                    dx = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
                    dy = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)

                    cx2 = np.clip(cx1 + dx, 0, w - 1)
                    cy2 = np.clip(cy1 + dy, 0, h - 1)
                    self.valid_pts.append((cx1, cy1, cx2, cy2, 0))  # label 0
            self.data_len = config.data["val_length"]
        else:
            self.data_len = config.data["data_length"]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.mode=="train":        
            is_similar = random.choice([True, False])
            pair_idx = random.randint(0, len(self.pairs) - 1)
            sar_path, rgb_path = self.pairs[pair_idx]
            # print(sar_path, rgb_path)

            #read rgb and sar images
            sar_image = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

            #resize SAR to match RGB dimensions
            sar_image = cv2.resize(sar_image, (rgb_image.shape[1], rgb_image.shape[0]))
            h, w = rgb_image.shape[:2]

            #generate full-size Gaussian once, centered
            gaussian_full = self.generate_gaussian(h, w)

            #choose crop centers
            if is_similar:
                cx, cy = self.random_point_near_center(h, w, delta=256)
                cx1, cy1 = cx, cy
                cx2, cy2 = cx, cy
            else:
                cx1, cy1 = self.random_point_near_center(h, w, delta=256)
                # cx2, cy2 = self.random_point_near_center(h, w, delta=400)
                min_offset =32
                dx = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
                dy = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)

                cx2 = np.clip(cx1 + dx, 0, w - 1)
                cy2 = np.clip(cy1 + dy, 0, h - 1)

            #crop SAR, RGB, and Gaussian
            sar_crop = self.safe_crop(sar_image, cx1, cy1, self.image_size)
            rgb_crop = self.safe_crop(rgb_image, cx2, cy2, self.image_size)
            
            #crop Gaussian
            if is_similar:
                g_crop = self.safe_crop(gaussian_full, cx1, cy1, self.image_size)
            else:
                g_crop = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            #convert to tensors
            sar_tensor = self.sar_transform(sar_crop)
            sar_tensor = sar_tensor.repeat(3, 1, 1)  # Repeat the SAR tensor to match RGB channels
            rgb_tensor = self.rgb_transform(rgb_crop)
            g_tensor = torch.from_numpy(g_crop).unsqueeze(0).float()

            return [rgb_tensor, sar_tensor, g_tensor], torch.tensor(is_similar, dtype=torch.float32)
        else: #make sure valid set is same during traing
            cx1, cy1, cx2, cy2, is_similar = self.valid_pts[idx]
            pair_idx = idx % len(self.pairs)
            sar_path, rgb_path = self.pairs[pair_idx]

            sar_image = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

            sar_image = cv2.resize(sar_image, (rgb_image.shape[1], rgb_image.shape[0]))
            h, w = rgb_image.shape[:2]
            gaussian_full = self.generate_gaussian(h, w)

            sar_crop = self.safe_crop(sar_image, cx1, cy1, self.image_size)
            rgb_crop = self.safe_crop(rgb_image, cx2, cy2, self.image_size)

            if is_similar:
                g_crop = self.safe_crop(gaussian_full, cx1, cy1, self.image_size)
            else:
                g_crop = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            sar_tensor = self.sar_transform(sar_crop)
            sar_tensor = sar_tensor.repeat(3, 1, 1)  # Repeat the SAR tensor to match RGB channels
            rgb_tensor = self.rgb_transform(rgb_crop) 
            g_tensor = torch.from_numpy(g_crop).unsqueeze(0).float()
            
            # rgb_tensor = rgb_tensor/255.0
            # sar_tensor = sar_tensor/255.0
            return [rgb_tensor, sar_tensor, g_tensor], torch.tensor(is_similar, dtype=torch.float32)

    def random_point_near_center(self, height, width, delta=20):
        center_x = width // 2
        center_y = height // 2
        cx = random.randint(center_x - delta, center_x + delta)
        cy = random.randint(center_y - delta, center_y + delta)
        return cx, cy

    def safe_crop(self, image, cx, cy, size):
        half = size // 2
        h, w = image.shape[:2]

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)

        cropped = image[y1:y2, x1:x2]

        if cropped.shape[0] != size or cropped.shape[1] != size:
            padded = np.zeros((size, size, cropped.shape[2]) if len(cropped.shape) == 3 else (size, size), dtype=cropped.dtype)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
            return padded
        return cropped

    def generate_gaussian(self, height, width, sigma_ratio=2):
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x*x + y*y)
        sigma = 1.0 / sigma_ratio
        gaussian = np.exp(-(d**2 / (2.0 * sigma**2)))
        gaussian /= gaussian.max()  # Normalize to [0, 1]
        return gaussian.astype(np.float32)



def get_dataloader_v2(config, mode="train", debug=False):
    dataset = ContrastiveDatafeeder(config, mode, debug)
    dataloader = DataLoader(dataset, batch_size=config.data["batch_size"], shuffle=True, num_workers=config.data["num_workers"])
    return dataloader