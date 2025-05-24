import sys
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Add src directory to path
sys.path.append(os.path.join(project_root, 'src'))
from datafeeder import get_dataloader_v2
from training import train
from models import resnet18
from utils import Configuration



if __name__=="__main__":
    # Use absolute path for config file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))

    model = resnet18(embedding_dim=128)

    config = Configuration(config_path) 

    train_loader = get_dataloader_v2(config, mode="train")
    val_loader = get_dataloader_v2(config, mode="valid")



    train(config_path=config_path, model=model, 
          train_loader=train_loader, 
          val_loader=val_loader)