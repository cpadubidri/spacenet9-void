from utils import Configuration, logger

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from .loss_fn import TripletLoss


def train_triplet(config_path, model, train_loader, val_loader):
    config = Configuration(config_path)

    #logger
    log = logger(
        log_dir=config.log_path,
        log_filename=f"log_{config.experiment_name}.log"
    )
    log.info(f"Experiment Name: {config.experiment_name}")
    log.info("Starting TripletLoss training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model.to(device)

    #init loss & optimizer
    criterion = TripletLoss(margin=config.training.get("margin", 1.0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'])

    #init tensorBoard
    tensorboard_log_path = os.path.join(config.training['savepath'], 'tensorboard')
    os.makedirs(tensorboard_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_path)
    log.info(f"TensorBoard initialized at {tensorboard_log_path}")

    best_val_loss = float('inf')
    log.info(f"Training for {config.training['epochs']} epochs")

    for epoch in range(config.training['epochs']):
        model.train()
        log.info(f"Epoch {epoch + 1}/{config.training['epochs']}")
        train_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        batch_count = 0

        #train loop
        for batch in tqdm(train_loader, desc="Training"):
            anchor, positive, negative = batch[0]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            anchor_vec = F.adaptive_avg_pool2d(anchor_embed, 1).view(anchor_embed.size(0), -1)
            positive_vec = F.adaptive_avg_pool2d(positive_embed, 1).view(positive_embed.size(0), -1)
            negative_vec = F.adaptive_avg_pool2d(negative_embed, 1).view(negative_embed.size(0), -1)

            distance_pos = F.pairwise_distance(anchor_vec, positive_vec, p=2)
            distance_neg = F.pairwise_distance(anchor_vec, negative_vec, p=2)

            loss = criterion(anchor_vec, positive_vec, negative_vec)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_pos_dist += distance_pos.mean().item()
            total_neg_dist += distance_neg.mean().item()
            batch_count += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_pos_dist = total_pos_dist / batch_count
        avg_neg_dist = total_neg_dist / batch_count

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_histogram('Embedding/Anchor', anchor_vec.detach().cpu(), epoch)
        writer.add_histogram('Embedding/Positive', positive_vec.detach().cpu(), epoch)
        writer.add_histogram('Embedding/Negative', negative_vec.detach().cpu(), epoch)
        writer.add_scalar('Distance/Train_Pos', avg_pos_dist, epoch)
        writer.add_scalar('Distance/Train_Neg', avg_neg_dist, epoch)

        log.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.8f}, Pos Dist: {avg_pos_dist:.4f}, Neg Dist: {avg_neg_dist:.4f}")

        #validation loop: For log embeddings to TensorBoard Projector every epoch
        embedding_list = []
        metadata_list = []
        label_img_list = []

        triplet_count = 0
        max_triplets = 5
        with torch.no_grad():
            for batch in val_loader:
                anchor, sar, rgb = batch[0]
                anchor, sar, rgb = anchor.to(device), sar.to(device), rgb.to(device)

                anchor_vec = F.adaptive_avg_pool2d(model(anchor), 1).view(anchor.size(0), -1)
                sar_vec = F.adaptive_avg_pool2d(model(sar), 1).view(sar.size(0), -1)
                rgb_vec = F.adaptive_avg_pool2d(model(rgb), 1).view(rgb.size(0), -1)

                for i in range(anchor_vec.size(0)):
                    if triplet_count >= max_triplets:
                        break

                    embedding_list.extend([
                        anchor_vec[i].cpu(),
                        sar_vec[i].cpu(),
                        rgb_vec[i].cpu()
                    ])
                    metadata_list.extend([
                        f"triplet_{triplet_count}",
                        f"triplet_{triplet_count}",
                        f"triplet_{triplet_count}"
                    ])
                    label_img_list.extend([
                        anchor[i].cpu(),
                        sar[i].cpu(),
                        rgb[i].cpu()
                    ])
                    triplet_count += 1
                if triplet_count >= max_triplets:
                    break

        embeddings = torch.stack(embedding_list)
        label_imgs = torch.stack(label_img_list)
        writer.add_embedding(
            mat=embeddings,
            metadata=metadata_list,
            label_img=label_imgs,
            global_step=epoch,
            tag="TripletProjection"
        ) #this can be visualized in tensorboard projector


        #validation loop for calualting loss
        val_loss = 0.0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        batch_count = 0
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = batch[0]
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)

                anchor_vec = F.adaptive_avg_pool2d(anchor_embed, 1).view(anchor_embed.size(0), -1)
                positive_vec = F.adaptive_avg_pool2d(positive_embed, 1).view(positive_embed.size(0), -1)
                negative_vec = F.adaptive_avg_pool2d(negative_embed, 1).view(negative_embed.size(0), -1)

                distance_pos = F.pairwise_distance(anchor_vec, positive_vec, p=2)
                distance_neg = F.pairwise_distance(anchor_vec, negative_vec, p=2)

                loss = criterion(anchor_vec, positive_vec, negative_vec)
                val_loss += loss.item()
                total_pos_dist += distance_pos.mean().item()
                total_neg_dist += distance_neg.mean().item()
                batch_count += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_pos_dist = total_pos_dist / batch_count
        avg_val_neg_dist = total_neg_dist / batch_count

        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        log.info(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.8f}, Pos Dist: {avg_val_pos_dist:.4f}, Neg Dist: {avg_val_neg_dist:.4f}")

        #save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_dir = os.path.join(config.training['savepath'], 'models')
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(model_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Saved best model to {best_model_path}")

        #save checkpoint
        if (epoch + 1) % config.training.get("checkpoint_interval", 5) == 0:
            ckpt_dir = os.path.join(config.training['savepath'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint at {ckpt_path}")

    log.info("Training complete.")
    writer.close()
