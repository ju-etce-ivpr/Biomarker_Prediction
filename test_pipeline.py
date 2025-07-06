import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.backbone import KimiaNetBackbone
from models.fcn_heads import MultiBiomarkerFCN
from data.dataset import WSIPatchDataset


def average_predictions(pred_list):
    """Average patch-level predictions to WSI-level probability vector."""
    stacked = torch.stack(pred_list, dim=0)  # [N_patches x 5]
    return torch.mean(stacked, dim=0)  # [5]


def evaluate_model(config, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wsi_probs = {}

    for level in ['local', 'global']:
        print(f"\n--- Evaluating {level} model ---")

        # Load model
        backbone = KimiaNetBackbone().to(device)
        fcn_heads = MultiBiomarkerFCN().to(device)
        ckpt = torch.load(os.path.join(checkpoint_dir, f"model_{level}.pt"), map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        fcn_heads.load_state_dict(ckpt['fcn_heads'])
        backbone.eval()
        fcn_heads.eval()

        # Dataset
        dataset = WSIPatchDataset(
            patch_dir=config['data']['patch_dir'],
            label_csv=config['data']['metadata_csv'],
            level=level
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for image, label in loader:
            image = image.to(device)
            with torch.no_grad():
                feat = backbone(image)
                out_list = fcn_heads(feat)
                probs = torch.cat(out_list, dim=1).squeeze(0).cpu()

            # WSI name parsing
            path = dataset.samples[loader.dataset.samples.index((dataset.samples[0]))][0]
            wsi_id = os.path.basename(path).split('_')[0]
            if wsi_id not in wsi_probs:
                wsi_probs[wsi_id] = {}
            if level not in wsi_probs[wsi_id]:
                wsi_probs[wsi_id][level] = []

            wsi_probs[wsi_id][level].append(probs)

    print("\n--- Soft Voting Results ---")
    for wsi_id in wsi_probs:
        local_avg = average_predictions(wsi_probs[wsi_id]['local'])
        global_avg = average_predictions(wsi_probs[wsi_id]['global'])

        # Modulated Rank Averaging (weights based on training performance)
        w_local = 0.6  # placeholder
        w_global = 0.4  # placeholder

        combined = w_local * local_avg + w_global * global_avg
        binary_prediction = (combined > 0.5).int()

        print(f"{wsi_id}: {binary_prediction.tolist()}")
