import torch
import torch.nn as nn


class FCNHead(nn.Module):
    """
    FCN for predicting a single biomarker from feature vector.
    """
    def __init__(self, input_dim=512):
        super(FCNHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class MultiBiomarkerFCN(nn.Module):
    """
    Module with 5 parallel FCNs, one for each biomarker.
    Returns: list of [B x 1] outputs, one per biomarker
    """
    def __init__(self, input_dim=512, num_biomarkers=5):
        super(MultiBiomarkerFCN, self).__init__()
        self.heads = nn.ModuleList([FCNHead(input_dim) for _ in range(num_biomarkers)])

    def forward(self, x):
        return [head(x) for head in self.heads]
