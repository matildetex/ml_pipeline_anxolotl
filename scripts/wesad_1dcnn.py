"""WESAD LOSO — 1D-CNN (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn as nn
from _dl_loso import run_dl_model


class CNN1D(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


if __name__ == '__main__':
    run_dl_model(
        model_fn_wrist=lambda c: CNN1D(in_channels=c),
        model_fn_all=lambda c: CNN1D(in_channels=c),
        name='1D-CNN',
        out_name='wesad_1dcnn',
    )
