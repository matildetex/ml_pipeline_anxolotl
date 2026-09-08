"""Shared DANN-tabular LOSO helper.
Domain = subject identity (subject-invariant features via GRL).
Applies BL-norm per fold internally.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from common import report_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x
    @staticmethod
    def backward(ctx, g): return g.neg() * ctx.lam, None


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=128, dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class _DANN(nn.Module):
    def __init__(self, in_dim, n_subjects, feat_dim=128):
        super().__init__()
        self.ext  = _MLP(in_dim, hidden=(256, 128), out_dim=feat_dim)
        self.cls  = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, 64),
                                   nn.ReLU(), nn.Linear(64, 2))
        self.disc = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, 64),
                                   nn.ReLU(), nn.Linear(64, n_subjects))

    def forward(self, x, lam):
        z = self.ext(x)
        return self.cls(z), self.disc(_GRL.apply(z, lam))

    def proba(self, x):
        z = self.ext(x)
        return torch.softmax(self.cls(z), dim=1)[:, 1]


def dann_tabular_loso(X_arr, y, groups, label,
                       bl_train_fn, bl_test_fn,
                       epochs=300, lr=1e-3,
                       domain_weight=0.1, batch_size=256):
    """
    X_arr      : raw (N, F) numpy array — BL-norm applied per fold inside here.
    bl_train_fn: fn(X, y, groups) -> X_normalised   (training subjects)
    bl_test_fn : fn(X, groups)    -> X_normalised   (test subject)
    """
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    for fold, (tr, te) in enumerate(logo.split(X_arr, y, groups)):
        X_tr_bl = bl_train_fn(X_arr[tr], y[tr], groups[tr])
        X_te_bl = bl_test_fn(X_arr[te], groups[te])

        imp = SimpleImputer(strategy='mean')
        X_tr_n = imp.fit_transform(X_tr_bl)
        X_te_n = imp.transform(X_te_bl)
        y_tr, y_te = y[tr], y[te]

        le    = LabelEncoder()
        d_enc = le.fit_transform(groups[tr])
        n_subj = len(le.classes_)
        n_feat  = X_tr_n.shape[1]

        w1 = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
        cls_w = torch.FloatTensor([1.0, w1]).to(DEVICE)

        Xt = torch.FloatTensor(X_tr_n); yt = torch.LongTensor(y_tr)
        dt = torch.LongTensor(d_enc)
        dl = DataLoader(TensorDataset(Xt, yt, dt), batch_size=batch_size,
                        shuffle=True, drop_last=True)

        model   = _DANN(n_feat, n_subj).to(DEVICE)
        opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        cls_fn  = nn.CrossEntropyLoss(weight=cls_w)
        disc_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            lam = 2 / (1 + np.exp(-10 * epoch / epochs)) - 1
            for xb, yb, db in dl:
                xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
                so, do = model(xb, lam)
                loss = cls_fn(so, yb) + domain_weight * disc_fn(do, db)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        tl = DataLoader(TensorDataset(torch.FloatTensor(X_te_n)), batch_size=512)
        probs = []
        with torch.no_grad():
            for (xb,) in tl:
                probs.extend(model.proba(xb.to(DEVICE)).cpu().numpy())
        probs = np.array(probs)
        f1 = f1_score(y_te, (probs >= 0.5).astype(int), average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te); all_prob.extend(probs)
        print(f'  fold {fold+1:2d} | {groups[te[0]]}  F1={f1:.3f}')

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)
