import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logger

# Use environment variable for log directory or default to a local path
log_dir = os.path.join(os.environ.get('SAVE_PATH', os.path.abspath('../../experiments/exp_01')), 'logs')
log = logger(log_dir=log_dir, log_filename="loss_verbose.log")

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
        
        #possitive and negative losses
        pos_loss = (1 - cos_sim) * (gaussian_mask + 0.1)
        neg_loss = (F.relu(cos_sim - self.margin).pow(2))#*0.1

        rgb_magnitude = rgb_norm.norm(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        sar_magnitude = sar_norm.norm(dim=1, keepdim=True)


        # Log for similar pairs
        if sim_mask.any():
            log.info(f"--- Similar Pairs ---")
            log.info(f"RGB norm (Similar): min={rgb_magnitude[sim_mask].min().item():.4f}, max={rgb_magnitude[sim_mask].max().item():.4f}, mean={rgb_magnitude[sim_mask].mean().item():.4f}, std={rgb_magnitude[sim_mask].std().item():.4f}")
            log.info(f"SAR norm (Similar): min={sar_magnitude[sim_mask].min().item():.4f}, max={sar_magnitude[sim_mask].max().item():.4f}, mean={sar_magnitude[sim_mask].mean().item():.4f}, std={sar_magnitude[sim_mask].std().item():.4f}")
            log.info(f"Cos Sim: min={cos_sim[sim_mask].min().item():.4f}, max={cos_sim[sim_mask].max().item():.4f}, mean={cos_sim[sim_mask].mean().item():.4f}")
            log.info(f"Gaussian: mean={gaussian_mask[sim_mask].mean().item():.4f}, std={gaussian_mask[sim_mask].std().item():.4f}")
            log.info(f"Pos loss: mean={pos_loss[sim_mask].mean().item():.6f}, sum={pos_loss[sim_mask].sum().item():.6f}")
        else:
            log.info("No similar pairs in this batch.")

        # Log for dissimilar pairs
        if dis_mask.any():
            log.info(f"--- Dissimilar Pairs ---")
            log.info(f"RGB norm (Similar): min={rgb_magnitude[dis_mask].min().item():.4f}, max={rgb_magnitude[dis_mask].max().item():.4f}, mean={rgb_magnitude[dis_mask].mean().item():.4f}, std={rgb_magnitude[dis_mask].std().item():.4f}")
            log.info(f"SAR norm (Similar): min={sar_magnitude[dis_mask].min().item():.4f}, max={sar_magnitude[dis_mask].max().item():.4f}, mean={sar_magnitude[dis_mask].mean().item():.4f}, std={sar_magnitude[dis_mask].std().item():.4f}")

            log.info(f"Cos Sim: min={cos_sim[dis_mask].min().item():.4f}, max={cos_sim[dis_mask].max().item():.4f}, mean={cos_sim[dis_mask].mean().item():.4f}")
            log.info(f"Gaussian: mean={gaussian_mask[dis_mask].mean().item():.4f}, std={gaussian_mask[dis_mask].std().item():.4f}")
            log.info(f"Neg loss: mean={neg_loss[dis_mask].mean().item():.6f}, sum={neg_loss[dis_mask].sum().item():.6f}")
        else:
            log.info("No dissimilar pairs in this batch.")



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


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_pos = F.pairwise_distance(anchor, positive, p=2) #>> pos distance
        distance_neg = F.pairwise_distance(anchor, negative, p=2) #>> neg distance
        # print(f"Distance pos: {distance_pos.mean()}, Distance neg: {distance_neg.mean()}")
        return F.relu(distance_pos - distance_neg + self.margin).mean() #>> triplet loss



class CosineTripletLoss(nn.Module):
    def __init__(self, reg_weight=0.1, margin=0.3, reduction='mean'):
        super().__init__()
        self.reg_weight = reg_weight
        self.reduction = reduction
        self.margin = margin

    def forward(self, anchor, positive, negative):
        #cosine similarity losses
        cosine_pos = F.cosine_similarity(anchor, positive, dim=1)
        cosine_neg = F.cosine_similarity(anchor, negative, dim=1)

        loss = F.relu(cosine_neg - cosine_pos + self.margin).mean()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        #positive Euclidean distance regularization
        pos_reg = (anchor - positive).pow(2).sum(dim=1).mean()
        loss += self.reg_weight * pos_reg

        return loss



def get_loss(name):
    if name == "contrastive":
        return GaussianModulatedContrastiveLoss(init_margin=1.0)
    elif name == "heatmap":
        return HeatmapRegressionLoss()
    else:
        raise NotImplementedError(f"Loss function {name} not implemented.")
