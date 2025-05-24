import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from utils import extract_patches, compute_cross_correlation, find_peak_shift

class ResNet50Triplet2D(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, H/32, W/32]

        self.projection = nn.Sequential(
            nn.Conv2d(2048, embedding_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, global_pool=False):
        feat_map = self.backbone(x)
        emb_map = self.projection(feat_map)

        if global_pool:
            pooled = F.adaptive_avg_pool2d(emb_map, 1).squeeze(-1).squeeze(-1)
            return F.normalize(pooled, p=2, dim=1)
        else:
            return F.normalize(emb_map, p=2, dim=1)


def load_model(weight_path, embedding_dim=512):
    model = ResNet50Triplet2D(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model



def compute_shift_with_model(model, sar_img, rgb_img):
    shift_map = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 2))

    (coords_sar_pairs, rgb_patches) = extract_patches(sar_img, rgb_img)

    for ((x, y), sar_patch), rgb_patch in zip(coords_sar_pairs, rgb_patches):\
        # with torch.no_grad():
        #     # Shape checks
        #     sar_tensor = torch.from_numpy(sar_patch).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        #     rgb_tensor = torch.from_numpy(rgb_patch).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]

        #     emb_sar = model(sar_tensor)
        #     emb_rgb = model(rgb_tensor)

        # ccm = compute_cross_correlation(emb_sar, emb_rgb)
        # dx, dy = find_peak_shift(ccm)
        # shift_map[y, x] = [dx, dy]
        dx, dy = 0, 0  # Stubbed out for demo run
        shift_map[y, x] = [dx, dy]

    return shift_map
