import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void', 'src')))
from datafeeder import get_dataloader_v3
from training import train_triplet
from models import resnet18, ResNet50Triplet2D
from utils import Configuration



if __name__=="__main__":
    config_path = "experiments/exp_11/config.json"


    model = ResNet50Triplet2D(embedding_dim=512)

    config = Configuration(config_path) 

    train_loader = get_dataloader_v3(config, mode="train")
    val_loader = get_dataloader_v3(config, mode="valid")



    train_triplet(config_path=config_path, model=model, 
          train_loader=train_loader, 
          val_loader=val_loader)