import numpy as np
import os
import cv2
import rasterio
from scipy.ndimage import zoom
from scipy.signal import correlate2d

def resize_to_match_resolution(img, current_res, target_res):
    scale = current_res / target_res
    return zoom(img, scale, order=1)

def extract_patches(sar_img, rgb_img, patch_size=512, stride=256):
    """
    Slide RGB over SAR (centered). Returns:
    - coords: (x, y)
    - sar_patches: cropped from resized SAR image
    - rgb_patches: RGB image patches
    """
    sar_patches, rgb_patches, coords = [], [], []

    half = patch_size // 2
    for y in range(0, rgb_img.shape[0] - patch_size + 1, stride):
        for x in range(0, rgb_img.shape[1] - patch_size + 1, stride):
            rgb_patch = rgb_img[y:y + patch_size, x:x + patch_size, :]
            sar_y = y + half
            sar_x = x + half

            # Check for SAR patch bounds
            sar_patch = sar_img[sar_y - patch_size: sar_y + patch_size,
                                sar_x - patch_size: sar_x + patch_size]
            if sar_patch.shape[0] == patch_size * 2 and sar_patch.shape[1] == patch_size * 2:
                sar_patches.append(sar_patch)
                rgb_patches.append(rgb_patch)
                coords.append((x, y))
    return zip(coords, sar_patches), rgb_patches

def compute_cross_correlation(emb1, emb2):
    """
    emb1, emb2: torch.Tensors with shape [1, C, H, W]
    Returns a 2D cross-correlation matrix
    """
    emb1_np = emb1.squeeze().cpu().numpy()  # [C, H, W]
    emb2_np = emb2.squeeze().cpu().numpy()  # [C, H, W]
    ccm = np.sum([
        correlate2d(emb1_np[c], emb2_np[c], mode='same')
        for c in range(emb1_np.shape[0])
    ], axis=0)
    return ccm

def find_peak_shift(ccm):
    """
    Find (dx, dy) from peak in CCM.
    """
    peak_idx = np.unravel_index(np.argmax(ccm), ccm.shape)
    center = np.array(ccm.shape) // 2
    dy, dx = peak_idx[0] - center[0], peak_idx[1] - center[1]
    return dx, dy

def save_shift_map(shift_map, output_path, profile):
    """
    Save (dx, dy) shift map as a 2-channel GeoTIFF
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving shift map to {output_path}")
    out_profile = profile.copy()
    out_profile.update({
        'count': 2,
        'dtype': 'float32'
    })
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(shift_map[..., 0].astype('float32'), 1)  # dx
        dst.write(shift_map[..., 1].astype('float32'), 2)  # dy
