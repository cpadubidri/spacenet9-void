from utils import Configuration, logger

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from .loss_fn import get_loss


def train(config_path, model, train_loader, val_loader):
    #init config
    config = Configuration(config_path)

    #init logger
    log = logger(log_dir=config.log_path, 
                 log_filename=f"log_{config.experiment_name}.log")
    log.info(f"Experiment Name: {config.experiment_name}")
    log.info("Starting training pipeline...")

    #config device
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)

    #init optimizer and loss function
    log.info(f"Setting Loss_fn: {config.training['loss']} and Optimizer with Lr: {config.training['lr']}")
    criterion = get_loss(config.training['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'])

    #init TensorBoard
    tensorboard_log_path = os.path.join(config.training['savepath'], 'tensorboard')
    os.makedirs(tensorboard_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_path)
    log.info(f"TensorBoard initialized at {tensorboard_log_path}")

    #training loop
    best_val_loss = float('inf')
    log.info(f"Starting training for {config.training['epochs']} epochs")

    for epoch in range(config.training['epochs']):
        model.train()
        log.info(f"Epoch {epoch + 1}/{config.training['epochs']}")
        train_loss = 0.0

        #train loop
        for batch in tqdm(train_loader, total=len(train_loader), desc="Training", unit="batch"):
            inputs, labels = batch
            rgb, sar, gaussian = inputs
            rgb, sar, gaussian, labels = rgb.to(device), sar.to(device), gaussian.to(device), labels.to(device)

            optimizer.zero_grad()

            rgb_embed = model(rgb)
            sar_embed = model(sar)

            # print(rgb_embed.shape, sar_embed.shape, labels.shape, gaussian.shape)

            loss = criterion(rgb_embed, sar_embed, labels, gaussian)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        log.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        #validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc="Validating", unit="batch"):
                inputs, labels = batch
                rgb, sar, gaussian = inputs
                rgb, sar, gaussian, labels = rgb.to(device), sar.to(device), gaussian.to(device), labels.to(device)

                rgb_embed = model(rgb)
                sar_embed = model(sar)

                loss = criterion(rgb_embed, sar_embed, labels, gaussian)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        log.info(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")

        #save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_base_path = os.path.join(config.training['savepath'],"models") 
            os.makedirs(best_model_base_path, exist_ok=True)
            best_model_path = best_model_base_path + f"/best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Saved new best model to {best_model_path}")

        #save checkpoint every 5 epochs
        if (epoch + 1) % config.training.get('checkpoint_interval', 5) == 0:
            checkpoint_base_path = os.path.join(config.training['savepath'], "checkpoints")
            os.makedirs(checkpoint_base_path, exist_ok=True)
            checkpoint_path = checkpoint_base_path+f"/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"Checkpoint saved at {checkpoint_path}")

    log.info("Training complete.")
    writer.close()
