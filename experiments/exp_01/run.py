import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('/home/savvas/SUPER-NAS/USERS/Chirag/PROJECTS/202504-spacenet9/data/spacenet9-void', 'src')))
from datafeeder.datafeeder import get_dataloader
from training import train
from models import resnet18
from utils import Configuration



if __name__=="__main__":
    config_path = "experiments/exp_01/config.json"


    model = resnet18(embedding_dim=128)

    config = Configuration('experiments/exp_01/config.json') 

    train_loader = get_dataloader(config)
    test_loader = get_dataloader(config)



    # train(config_path=config_path, model=model, 
    #       train_loader=train_loader, 
    #       test_loader=test_loader)