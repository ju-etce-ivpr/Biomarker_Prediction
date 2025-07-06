import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class WSIPatchDataset(Dataset):
    """
    PyTorch Dataset for WSI patches with biomarker labels.
    Supports loading either local or global magnification level.
    """
    def __init__(self, patch_dir, label_csv, level='local', transform=None):
        self.patch_dir = patch_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.labels_df = pd.read_csv(label_csv)
        self.level = level

        # Build list of image paths and match with labels
        self.samples = []
        for patient_id in self.labels_df['ID']:
            patch_folder = os.path.join(patch_dir, f"{self.level}_patches", patient_id)
            if not os.path.exists(patch_folder):
                continue
            for fname in os.listdir(patch_folder):
                if fname.endswith(".png"):
                    fpath = os.path.join(patch_folder, fname)
                    label = self.labels_df[self.labels_df['ID'] == patient_id].iloc[0, 1:].values.astype(int)
                    self.samples.append((fpath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label
