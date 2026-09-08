"""WESAD LOSO — SSM/Mamba classifier (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from _dl_loso import run_dl_model


class SSMBlock(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
        self.D = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        A = torch.exp(-torch.exp(self.A))
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outputs = []
        for t in range(x.size(1)):
            h = A * h + self.B(x[:, t, :])
            outputs.append(self.C(h))
        out = torch.stack(outputs, dim=1)
        return self.norm(out + self.D(x))


class SSMClassifier(nn.Module):
    def __init__(self, in_channels=5, d_model=128, num_layers=2,
                 num_classes=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList([SSMBlock(d_model) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B, T, C)
        x = self.input_proj(x)
        for layer in self.layers:
            x = self.drop(layer(x))
        return self.classifier(x.mean(dim=1))


if __name__ == '__main__':
    run_dl_model(
        model_fn_wrist=lambda c: SSMClassifier(in_channels=c),
        model_fn_all=lambda c: SSMClassifier(in_channels=c),
        name='Mamba (SSM)',
        out_name='wesad_mamba',
    )
