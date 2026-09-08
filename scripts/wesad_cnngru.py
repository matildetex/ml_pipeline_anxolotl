"""WESAD LOSO — CNN-GRU (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn as nn
from _dl_loso import run_dl_model


class CNNGRUHybrid(nn.Module):
    def __init__(self, in_channels=16, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, 128, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)           # (B, 128, T)
        x = x.permute(0, 2, 1)   # (B, T, 128)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


if __name__ == '__main__':
    run_dl_model(
        model_fn_wrist=lambda c: CNNGRUHybrid(in_channels=c),
        model_fn_all=lambda c: CNNGRUHybrid(in_channels=c),
        name='CNN-GRU',
        out_name='wesad_cnngru',
    )
