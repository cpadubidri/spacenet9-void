## Concepts:

    Phase 1     Train ResNet encoder using contrastive loss with distance-weighted sampling  
    Phase 2     Use encoder in a Siamese setup to predict (dx,dy) via regression  
    Deploy      Option 1: Exhaustive search in embedding space  
                Option 2: Dense shift prediction via sliding window regression  


## Phase 1: Contrastive Learning for SAR–Optical Patch Alignment
Idea: Use ResNet encoder to learn modality-invariant, spatially-aware embeddings from SAR–Optical patch pairs using contrastive loss. And a Gaussian distance-weighted mask applied to utilize the keypoint annotations provided.

### Data Generation

- Start with 3 SAR–Optical image pairs (each ~6000×6000) and ~250 aligned keypoints.
- For each keypoint:
  - Extract a 2048×2048 patch from both SAR and Optical images centered on the match.
  - Precompute a 2048×2048 Gaussian mask with peak at the center.
- At training time:
  - Randomly crop 256×256 patches from the same spatial location in SAR, Optical, and Gaussian mask.
  - Downsample the mask to 16×16 to match encoder output resolution.


### Model & Loss

- Shared ResNet encoder produces feature maps of shape (B, C, x, y). x=y-> embedding dimension
- Compute cosine similarity per spatial location between SAR and Optical feature maps.
- Apply the downsampled (to x,y) Gaussian mask to weight central similarity scores.
- Use distance-weighted contrastive loss:
    
    $\mathcal{L} = y(1 - \text{sim})^2 + (1 - y) \cdot \text{ReLU}(\text{sim} - m)^2$

      where `sim` is the masked cosine similarity and `y` is the match label.


### Deployment (Phase 1 Only)

- At test time, for a given optical patch:
  - Generate multiple(~1000, different shift) SAR candidate patches with known synthetic shifts.
  - Pass all patches through the encoder to obtain embeddings.
  - Measure cosine similarity with the optical patch embedding.
  - Choose the SAR patch with maximum similarity — this indicates the best match and estimated shift.







#### Notes:
1. SAR and Optical image are cropped using (156.25*2) meter around the keypoint.