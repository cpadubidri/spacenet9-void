import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void', 'src')))
from datafeeder import get_dataloader_v2
from training import train
from models import resnet18, ResNet50Spatial
from utils import Configuration



if __name__=="__main__":
    config_path = "experiments/exp_03/config.json"


    model = ResNet50Spatial(embedding_dim=64)

    config = Configuration('experiments/exp_03/config.json') 

    train_loader = get_dataloader_v2(config, mode="train")
    val_loader = get_dataloader_v2(config, mode="valid")



    train(config_path=config_path, model=model, 
          train_loader=train_loader, 
          val_loader=val_loader)