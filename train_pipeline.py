import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.backbone import KimiaNetBackbone
from models.fcn_heads import MultiBiomarkerFCN
from models.loss import composite_loss
from data.dataset import WSIPatchDataset


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for level, mag in zip(['local', 'global'], config['data']['magnifications']):
        print(f"\n--- Training {level} model (Magnification {mag}x) ---")

        # Dataset
        dataset = WSIPatchDataset(
            patch_dir=config['data']['patch_dir'],
            label_csv=config['data']['metadata_csv'],
            level=level
        )
        train_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)

        # Model
        backbone = KimiaNetBackbone().to(device)
        fcn_heads = MultiBiomarkerFCN(input_dim=512).to(device)
        params = list(backbone.parameters()) + list(fcn_heads.parameters())
        optimizer = torch.optim.Adam(params, lr=config['train']['learning_rate'])

        # Training loop
        for epoch in range(config['train']['epochs']):
            backbone.train()
            fcn_heads.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                features = backbone(images)
                preds = fcn_heads(features)
                loss = composite_loss(
                    preds, labels,
                    alpha=config['loss']['alpha'],
                    beta=config['loss']['beta'],
                    gamma=config['loss']['gamma']
                )

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (batch_idx + 1) % config['train']['log_interval'] == 0:
                    print(f"Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            print(f"Epoch {epoch+1}/{config['train']['epochs']}, Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint
        os.makedirs(config['train']['save_dir'], exist_ok=True)
        torch.save({
            'backbone': backbone.state_dict(),
            'fcn_heads': fcn_heads.state_dict()
        }, os.path.join(config['train']['save_dir'], f"model_{level}.pt"))

        print(f"Model saved for {level} at {config['train']['save_dir']}/model_{level}.pt")
