"""WESAD LOSO — Transformer classifier (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from _dl_loso import run_dl_model


class TransformerClassifier(nn.Module):
    def __init__(self, in_channels=5, seq_len=60, num_classes=2,
                 d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, T, C)
        x = self.input_proj(x)           # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


if __name__ == '__main__':
    run_dl_model(
        model_fn_wrist=lambda c: TransformerClassifier(in_channels=c),
        model_fn_all=lambda c: TransformerClassifier(in_channels=c),
        name='Transformer',
        out_name='wesad_transformer',
    )
