import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianWeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, output1, output2, label, gaussian_weight):
        """
        Args:
            output1: Tensor of shape (B, D) — embeddings from SAR
            output2: Tensor of shape (B, D) — embeddings from RGB
            label: Tensor of shape (B,) — 1 = similar, 0 = dissimilar
            gaussian_weight: Tensor of shape (B, 1, H, W) — Gaussian maps
        Returns:
            scalar loss
        """
        #compute Euclidean distance between embeddings
        distances = F.pairwise_distance(output1, output2, p=2)  # (B,)

        #contrastive loss per sample
        base_loss = label * distances.pow(2) + (1 - label) * F.relu(self.margin - distances).pow(2)

        g_mean = gaussian_weight.mean(dim=[1, 2, 3])  # (B,)

        #scale Gaussian mean to range [0.5, 1.0] to keep weights stable
        g_scaled = 0.5 + 0.5 * g_mean  # (B,) now in [0.5, 1.0]

        # Use the same scaled value for both positive and negative
        # So positive samples use g_scaled, negatives use g_scaled.mean()
        mean_neg_weight = g_scaled.mean().detach()  # fixed scalar for negatives

        effective_weight = label * g_scaled + (1 - label) * mean_neg_weight

        loss = base_loss * effective_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # returns (B,) if no reduction





def get_loss(name):
    if name == "contrastive":
        return GaussianWeightedContrastiveLoss(margin=1.0)
    else:
        raise NotImplementedError(f"Loss function {name} not implemented.")
