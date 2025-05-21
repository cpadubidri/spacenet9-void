from utils import Configuration, logger

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from .loss_fn import GaussianModulatedContrastiveLoss


def train(config_path, model, train_loader, val_loader):
    config = Configuration(config_path)

    #logger
    log = logger(
        log_dir=config.log_path, 
        log_filename=f"log_{config.experiment_name}.log"
    )
    log.info(f"Experiment Name: {config.experiment_name}")
    log.info("Starting training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)

    #init loss, dont norm gaussian we do it in the dataloader, init margin 0.5
    criterion = GaussianModulatedContrastiveLoss(
        init_margin=0.5,
        reduction='mean',
        normalize_gaussian='none'
    ).to(device)

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'])

    #init tensorboard
    tensorboard_log_path = os.path.join(config.training['savepath'], 'tensorboard')
    os.makedirs(tensorboard_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_path)
    log.info(f"TensorBoard initialized at {tensorboard_log_path}")

    best_val_loss = float('inf')
    log.info(f"Starting training for {config.training['epochs']} epochs")

    #epochs
    for epoch in range(config.training['epochs']):
        model.train()
        log.info(f"Epoch {epoch + 1}/{config.training['epochs']}")
        train_loss = 0.0

        #train loop
        for batch in tqdm(train_loader, total=len(train_loader), desc="Training"):
            inputs, labels = batch
            rgb, sar, gaussian = inputs
            rgb, sar, gaussian, labels = rgb.to(device), sar.to(device), gaussian.to(device), labels.to(device)

            optimizer.zero_grad()

            rgb_embed = model(rgb)
            sar_embed = model(sar)

            gaussian = F.interpolate(gaussian, size=rgb_embed.shape[2:], mode='bilinear', align_corners=False)
            
            log.info(f"*******************************************")
            log.info(f"Similar count: {(labels == 1).sum().item()}, Dissimilar count: {(labels == 0).sum().item()}")

            loss = criterion(rgb_embed, sar_embed, labels, gaussian)
            log.info(f"*******************************************")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch) #>>write loss to tb
        writer.add_histogram('Embedding/RGB', rgb_embed, epoch) #>>write rgb embedding to tb
        writer.add_histogram('Embedding/SAR', sar_embed, epoch) #>>write sar embedding to tb
        writer.add_scalar('Margin', criterion.margin.item(), epoch) #>>margin is learnable so visualize it

        log.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss}")

        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                rgb, sar, gaussian = inputs
                rgb, sar, gaussian, labels = rgb.to(device), sar.to(device), gaussian.to(device), labels.to(device)

                rgb_embed = model(rgb)
                sar_embed = model(sar)

                gaussian = F.interpolate(gaussian, size=rgb_embed.shape[2:], mode='bilinear', align_corners=False)
                
                log.info(f"*******************************************")
                loss = criterion(rgb_embed, sar_embed, labels, gaussian)
                log.info(f"*******************************************")
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        log.info(f"Epoch {epoch + 1} - Valid Loss: {avg_val_loss}")
        log.info(f"Epoch {epoch+1}: Current margin = {criterion.margin.item():.4f}")


        #save best model if loss is lower over epochs
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_base_path = os.path.join(config.training['savepath'], "models") 
            os.makedirs(best_model_base_path, exist_ok=True)
            best_model_path = os.path.join(best_model_base_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Saved best model to {best_model_path}")

        #save checkpoint (for safety)
        if (epoch + 1) % config.training.get('checkpoint_interval', 5) == 0:
            checkpoint_base_path = os.path.join(config.training['savepath'], "checkpoints")
            os.makedirs(checkpoint_base_path, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_base_path, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"Checkpoint saved at {checkpoint_path}")

    log.info("Training complete.")
    writer.close()
