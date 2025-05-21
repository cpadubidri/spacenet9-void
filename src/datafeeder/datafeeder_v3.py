import os
import random
import pandas as pd
import cv2
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TripletDatafeeder(Dataset):
    def __init__(self, config, mode="train", debug=False):
        self.debug = debug
        self.data_path = config.data["data_path"]
        self.image_size = config.data["image_size"]
        self.min_offset = 50
        self.mode = mode.lower()
        
        data_list = pd.read_csv(os.path.join(self.data_path, 'file_list.csv'))['file_name']
        self.pairs = []
        for filename in data_list:
            sar_path = os.path.join(self.data_path, filename.replace('source', 'sar'))
            rgb_path = os.path.join(self.data_path, filename.replace('source', 'optical'))
            if os.path.exists(sar_path) and os.path.exists(rgb_path):
                self.pairs.append((sar_path, rgb_path))

        self.sar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[40.2317 / 255.0], std=[37.8343 / 255.0])
        ])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[103.28885539 / 255.0, 111.16151289 / 255.0, 120.56394087 / 255.0],
                                 std=[65.82869639 / 255.0, 70.8654012 / 255.0, 77.10939367 / 255.0])
        ])

        if self.mode == 'valid':
            self.val_triplets = []
            num_samples = config.data["val_length"]
            h, w = 1310, 1310  # Set based on expected image size

            for _ in range(num_samples):
                pair_idx = random.randint(0, len(self.pairs) - 1)
                cx, cy = self.random_point_near_center(h, w, delta=256)

                # Negative offset
                dx = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
                dy = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
                neg_cx = np.clip(cx + dx, 0, w - 1)
                neg_cy = np.clip(cy + dy, 0, h - 1)

                self.val_triplets.append((pair_idx, cx, cy, neg_cx, neg_cy))

            self.data_len = config.data["val_length"]
        else:
            self.data_len = config.data["data_length"]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.mode == "valid":
            pair_idx, cx, cy, neg_cx, neg_cy = self.val_triplets[idx]
        else:
            pair_idx = random.randint(0, len(self.pairs) - 1)
            cx, cy = self.random_point_near_center(1310, 1310, delta=256)
            dx = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
            dy = random.choice([-1, 1]) * random.randint(self.min_offset, self.min_offset + 50)
            neg_cx = np.clip(cx + dx, 0, 1310 - 1)
            neg_cy = np.clip(cy + dy, 0, 1310 - 1)

        sar_path, rgb_path = self.pairs[pair_idx]

        sar_image = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        sar_image = cv2.resize(sar_image, (rgb_image.shape[1], rgb_image.shape[0]))

        anchor_crop = self.safe_crop(rgb_image, cx, cy, self.image_size)
        positive_crop = self.safe_crop(sar_image, cx, cy, self.image_size)
        negative_crop = self.safe_crop(sar_image, neg_cx, neg_cy, self.image_size)

        anchor_tensor = self.rgb_transform(anchor_crop)
        positive_tensor = self.sar_transform(positive_crop).repeat(3, 1, 1)
        negative_tensor = self.sar_transform(negative_crop).repeat(3, 1, 1)

        return [anchor_tensor, positive_tensor, negative_tensor], torch.tensor(0)

    def random_point_near_center(self, height, width, delta=20):
        cx = random.randint(width // 2 - delta, width // 2 + delta)
        cy = random.randint(height // 2 - delta, height // 2 + delta)
        return cx, cy

    def safe_crop(self, image, cx, cy, size):
        half = size // 2
        h, w = image.shape[:2]
        x1, y1 = max(cx - half, 0), max(cy - half, 0)
        x2, y2 = min(cx + half, w), min(cy + half, h)
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] != size or crop.shape[1] != size:
            padded = np.zeros((size, size, crop.shape[2]) if len(crop.shape) == 3 else (size, size), dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            return padded
        return crop

def get_dataloader_v3(config, mode="train", debug=False):
    dataset = TripletDatafeeder(config, mode, debug)
    dataloader = DataLoader(dataset, batch_size=config.data["batch_size"], shuffle=True, num_workers=config.data["num_workers"])
    return dataloader