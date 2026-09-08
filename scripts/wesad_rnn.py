"""WESAD LOSO — Bidirectional GRU (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn as nn
from _dl_loso import run_dl_model


class RNN1D(nn.Module):
    def __init__(self, in_channels=5, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.rnn = nn.GRU(in_channels, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T) → RNN wants (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])


if __name__ == '__main__':
    run_dl_model(
        model_fn_wrist=lambda c: RNN1D(in_channels=c),
        model_fn_all=lambda c: RNN1D(in_channels=c),
        name='RNN (BiGRU)',
        out_name='wesad_rnn',
    )
