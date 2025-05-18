import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void', 'src')))
from datafeeder import get_dataloader_v2
from training import train
from models import resnet18
from utils import Configuration



if __name__=="__main__":
    config_path = "experiments/exp_01/config.json"


    model = resnet18(embedding_dim=128)

    config = Configuration('experiments/exp_01/config.json') 

    train_loader = get_dataloader_v2(config, mode="train")
    val_loader = get_dataloader_v2(config, mode="valid")



    train(config_path=config_path, model=model, 
          train_loader=train_loader, 
          val_loader=val_loader)