"""WESAD LOSO — DANN with subject-domain GRL (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm
from common import load_wesad_seq, report_metrics, save_results

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_WRIST_IDX = [0, 5, 6, 7, 8]


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


class FeatureExtractor(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        # nhead must divide d_model; pick largest power-of-2 that does, capped at 4
        nhead = max(h for h in [1, 2, 4] if n_channels % h == 0)
        enc = nn.TransformerEncoderLayer(
            d_model=n_channels, nhead=nhead,
            dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.proj = nn.Linear(n_channels, 128)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, T, C)
        x = self.transformer(x)
        return self.proj(x.mean(dim=1))


class DANN(nn.Module):
    def __init__(self, n_channels, n_subjects):
        super().__init__()
        self.extractor = FeatureExtractor(n_channels)
        self.stress_cls = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 2))
        self.domain_dis = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, n_subjects))

    def forward(self, x, lam):
        z = self.extractor(x)
        stress = self.stress_cls(z)
        domain = self.domain_dis(GRL.apply(z, lam))
        return stress, domain


def run_loso(X_seq, y_seq, groups_seq, label):
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X_seq, y_seq, groups_seq))
    _pbar = tqdm(enumerate(_splits), total=len(_splits),
                 desc="LOSO", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in _pbar:
        subj = groups_seq[te[0]]
        print(f'  fold {fold+1:2d} | subject: {subj}')

        X_tr, y_tr = X_seq[tr], y_seq[tr]
        X_te, y_te = X_seq[te], y_seq[te]
        d_tr = groups_seq[tr]

        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

        le = LabelEncoder()
        d_enc = le.fit_transform(d_tr)
        n_subj = len(le.classes_)
        n_ch = X_tr.shape[1]

        Xt = torch.FloatTensor(X_tr)
        yt = torch.LongTensor(y_tr)
        dt = torch.LongTensor(d_enc)

        model = DANN(n_ch, n_subj).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        w = torch.FloatTensor([1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
        loss_s = nn.CrossEntropyLoss(weight=w)
        loss_d = nn.CrossEntropyLoss()

        dl = DataLoader(TensorDataset(Xt, yt, dt), batch_size=256, shuffle=True)
        model.train()
        for epoch in range(100):
            lam = 2 / (1 + np.exp(-10 * epoch / 250)) - 1
            for xb, yb, db in dl:
                xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
                so, do = model(xb, lam)
                loss = loss_s(so, yb) + 0.1 * loss_d(do, db)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        test_dl = DataLoader(TensorDataset(torch.FloatTensor(X_te)), batch_size=256)
        preds, probs = [], []
        with torch.no_grad():
            for (xb,) in test_dl:
                logits, _ = model(xb.to(DEVICE), lam=0)
                preds.extend(logits.argmax(1).cpu().numpy())
                probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())

        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te); all_prob.extend(probs)
        print(f'           F1={f1:.4f}')

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


if __name__ == '__main__':
    print(f'Loading WESAD sequences... device={DEVICE}')
    X_seq, y_seq, groups_seq = load_wesad_seq()
    all_results = []

    print('\n' + '='*65)
    print('  WESAD LOSO — DANN — WRIST (5 channels)')
    print('='*65)
    r = run_loso(X_seq[:, SEQ_WRIST_IDX, :], y_seq, groups_seq,
                 'DANN WESAD — wrist (5 ch)')
    all_results.append(r)

    print('\n' + '='*65)
    print('  WESAD LOSO — DANN — ALL (16 channels)')
    print('='*65)
    r = run_loso(X_seq, y_seq, groups_seq, 'DANN WESAD — all (16 ch)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_dann.json'))
