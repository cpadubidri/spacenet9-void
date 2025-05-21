import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger

log = logger(log_dir="/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void/experiments/exp_03/logs", 
             log_filename="loss_verbose.log")

class GaussianModulatedContrastiveLoss(nn.Module):
    def __init__(self, init_margin=1.0, reduction='mean', normalize_gaussian='max'):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(init_margin))
        self.reduction = reduction
        self.normalize_gaussian = normalize_gaussian

    def forward(self, rgb_embed, sar_embed, label, gaussian_mask):
        #norm embeddings
        rgb_norm = F.normalize(rgb_embed, dim=1)
        sar_norm = F.normalize(sar_embed, dim=1)

        log.info(f"RGB norm: min={rgb_norm.min().item():.4f}, max={rgb_norm.max().item():.4f}, mean={rgb_norm.mean().item():.4f}, std={rgb_norm.std().item():.4f}")
        log.info(f"SAR norm: min={sar_norm.min().item():.4f}, max={sar_norm.max().item():.4f}, mean={sar_norm.mean().item():.4f}, std={sar_norm.std().item():.4f}")

        #cosine similarity
        cos_sim = (rgb_norm * sar_norm).sum(dim=1, keepdim=True)

        #normalize gaussian mask >> dont use this, we do it in the dataloader
        if self.normalize_gaussian == 'max':
            gaussian_mask = gaussian_mask / (gaussian_mask.amax(dim=[2, 3], keepdim=True) + 1e-6)
        elif self.normalize_gaussian == 'sum':
            gaussian_mask = gaussian_mask / (gaussian_mask.sum(dim=[2, 3], keepdim=True) + 1e-6)

        #prepare label
        label = label.view(-1, 1, 1, 1).float()
        label = label.expand_as(cos_sim)

        #cosine similarity stats
        sim_mask = label.bool()
        dis_mask = ~sim_mask
        log.info(f"Cos Sim - Similar: min={cos_sim[sim_mask].min().item():.4f}, max={cos_sim[sim_mask].max().item():.4f}, mean={cos_sim[sim_mask].mean().item():.4f}")
        log.info(f"Cos Sim - Dissimilar: min={cos_sim[dis_mask].min().item():.4f}, max={cos_sim[dis_mask].max().item():.4f}, mean={cos_sim[dis_mask].mean().item():.4f}")

        #gaussian map, should be 0 for dissimilar pairs
        log.info(f"Gaussian - Similar: mean={gaussian_mask[sim_mask].mean().item():.4f}, std={gaussian_mask[sim_mask].std().item():.4f}")
        log.info(f"Gaussian - Dissimilar: mean={gaussian_mask[dis_mask].mean().item():.4f}, std={gaussian_mask[dis_mask].std().item():.4f}")

        #possitive and negative losses
        pos_loss = (1 - cos_sim) * (gaussian_mask + 0.1)
        neg_loss = F.relu(cos_sim - self.margin).pow(2)

        log.info(f"Pos loss: mean={pos_loss[sim_mask].mean().item():.6f}, sum={pos_loss[sim_mask].sum().item():.6f}")
        log.info(f"Neg loss: mean={neg_loss[dis_mask].mean().item():.6f}, sum={neg_loss[dis_mask].sum().item():.6f}")

        #total loss
        loss = label * pos_loss + (1 - label) * neg_loss
        log.info(f"Total loss (before reduction): mean={loss.mean().item():.6f}, sum={loss.sum().item():.6f}")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class HeatmapRegressionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred_heatmap, target_heatmap):
        return self.loss_fn(pred_heatmap, target_heatmap)


def get_loss(name):
    if name == "contrastive":
        return GaussianModulatedContrastiveLoss(margin=1.0)
    elif name == "heatmap":
        return HeatmapRegressionLoss()
    else:
        raise NotImplementedError(f"Loss function {name} not implemented.")
