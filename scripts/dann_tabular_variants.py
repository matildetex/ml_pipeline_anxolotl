"""DANN tabular reproduction — Table 4.9 (WESAD-Full, 137 feat) and
Table 4.10 (WESAD Wrist-Only, 55 feat) "DANN + IG + SMOTE" rows.

V0_Baseline is the only variant used for these two thesis rows (V1-V8
remain defined below as the architecture/hyperparameter ablation they
were originally written for, but are not exercised by __main__ here).
20 features via mutual_info_classif (per fold, train-only), SMOTE per
fold (train-only). Seeded (torch + numpy) for reproducibility; results
are saved via common.save_results.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.autograd import Function
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from common import load_wesad_tabular, report_metrics, save_results

DEVICE = torch.device('cpu')
TOP_K  = 20
EPOCHS = 250
LR     = 1e-3
SEED   = 42

# Canonical wrist-55 definition (Table 4.10) -- SAME 5 channels as
# 1_wesad/wrist_only/run_wesad_wrist55_blnorm.py. Deliberately NOT
# common.py's wrist_cols(), which is a 7-channel/77-feature definition
# (adds eda_wrist + temp_wrist) used elsewhere for a different purpose.
WRIST_PREFIXES_55 = ('acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_')

# Old-schema 137-feat snapshot (Table 4.9, WESAD-Full) -- same file used
# by scripts/wesad_ensemble.py and 1_wesad/classical/wesad_autogluon_rerun_fixed.py.
WESAD_137_SNAPSHOT = os.path.join(os.path.dirname(__file__), '..', 'processed_data',
                                   'wesad_archive', 'processed_data_137feat_snapshot.pkl')


def load_wesad_137():
    with open(WESAD_137_SNAPSHOT, 'rb') as f:
        d = pickle.load(f)
    X = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
    y = np.array(d['y']); groups = np.array(d['groups']); feat_cols = list(d['feature_cols'])
    assert len(feat_cols) == 137, (
        f'Expected 137 features from the old-schema WESAD snapshot (thesis Section 4.2), '
        f'got {len(feat_cols)} -- see DATA.md.'
    )
    return X, y, groups, feat_cols


# ── GRL ──────────────────────────────────────────────────────────────────────
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


# ── Architectures ─────────────────────────────────────────────────────────────
class MLP_128_64(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU())
    def forward(self, x): return self.net(x)


class MLP_BN_Drop(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x): return self.net(x)


class MLP_256_128_64(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU())
    def forward(self, x): return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d); self.relu = nn.ReLU(); self.fc2 = nn.Linear(d, d)
    def forward(self, x):
        return self.relu(x + self.fc2(self.relu(self.fc1(x))))

class MLP_Residual(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.proj = nn.Linear(n_feat, 128)
        self.res1 = ResidualBlock(128)
        self.down = nn.Linear(128, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.proj(x))
        x = self.res1(x)
        return self.relu(self.down(x))


class DANN(nn.Module):
    def __init__(self, extractor, n_subjects):
        super().__init__()
        self.ext = extractor
        self.stress = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.domain = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_subjects))
    def forward(self, x, lam):
        z = self.ext(x)
        return self.stress(z), self.domain(GRL.apply(z, lam))


# ── LOSO runner ───────────────────────────────────────────────────────────────
def run_variant(X_df, y_arr, groups_arr, feat_cols, cfg, label):
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    for fold, (tr, te) in enumerate(logo.split(X_df[feat_cols], y_arr, groups_arr)):
        X_tr_raw = X_df[feat_cols].values[tr].astype(float)
        X_te_raw = X_df[feat_cols].values[te].astype(float)
        y_tr, y_te = y_arr[tr], y_arr[te]
        d_tr = groups_arr[tr]

        imp = SimpleImputer(strategy='mean')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr_raw))
        X_te = imp.transform(sc.transform(X_te_raw))

        # IG feature selection
        scores  = mutual_info_classif(X_tr, y_tr, random_state=42)
        idx     = np.argsort(scores)[::-1][:TOP_K]
        X_tr_k  = X_tr[:, idx]
        X_te_k  = X_te[:, idx]

        # Encode group strings to ints before SMOTE concat
        le_fold = LabelEncoder()
        d_tr_enc = le_fold.fit_transform(d_tr).astype(float)
        # SMOTE (append domain labels to resample consistently)
        X_sm_in = np.column_stack([X_tr_k, d_tr_enc])
        X_sm_out, y_sm = SMOTE(random_state=42).fit_resample(X_sm_in, y_tr)
        X_tr_sm = X_sm_out[:, :-1]
        d_enc   = np.round(X_sm_out[:, -1]).astype(int)
        d_enc   = np.clip(d_enc, 0, len(le_fold.classes_) - 1)
        n_subj  = len(le_fold.classes_)
        d_enc   = d_enc % n_subj  # safety clamp after SMOTE interpolation

        n_feat = X_tr_sm.shape[1]
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        ext = cfg['arch'](n_feat)
        model = DANN(ext, n_subj).to(DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        if cfg.get('cosine'):
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        cw = torch.FloatTensor([1.0, 1.76]).to(DEVICE) if cfg.get('class_weight') else None
        loss_s = nn.CrossEntropyLoss(weight=cw)
        loss_d = nn.CrossEntropyLoss()
        dw = cfg.get('domain_weight', 0.1)

        Xt = torch.FloatTensor(X_tr_sm).to(DEVICE)
        yt = torch.LongTensor(y_sm).to(DEVICE)
        dt = torch.LongTensor(d_enc).to(DEVICE)

        pretrain = cfg.get('pretrain_epochs', 0)
        lam_max  = cfg.get('lam_max', 1.0)

        model.train()
        for epoch in range(EPOCHS):
            if epoch < pretrain:
                lam = 0.0
            else:
                p   = (epoch - pretrain) / (EPOCHS - pretrain)
                lam = lam_max * (2 / (1 + np.exp(-10 * p)) - 1)

            so, do = model(Xt, lam)
            loss = loss_s(so, yt) + dw * loss_d(do, dt)
            opt.zero_grad(); loss.backward()
            if cfg.get('grad_clip'):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if cfg.get('cosine'):
                sched.step()

        model.eval()
        with torch.no_grad():
            so, _ = model(torch.FloatTensor(X_te_k).to(DEVICE), lam=0)
            probs = torch.softmax(so, dim=1)[:, 1].cpu().numpy()
            preds = so.argmax(1).cpu().numpy()

        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te); all_prob.extend(probs)
        print(f'  fold {fold+1:2d}  F1={f1:.4f}', flush=True)

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


# ── Variant definitions ───────────────────────────────────────────────────────
VARIANTS = {
    'V0_Baseline':        {'arch': MLP_128_64,   'lam_max': 1.0, 'domain_weight': 0.1},
    'V1_ScaledLam_CW':    {'arch': MLP_128_64,   'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True},
    'V2_BN_Dropout':      {'arch': MLP_BN_Drop,  'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True},
    'V3_Deeper':          {'arch': MLP_256_128_64,'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True},
    'V4_Residual':        {'arch': MLP_Residual,  'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True},
    'V5_GradClip':        {'arch': MLP_128_64,   'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True, 'grad_clip': True},
    'V6_Cosine':          {'arch': MLP_128_64,   'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True, 'cosine': True},
    'V7_DomW_005':        {'arch': MLP_128_64,   'lam_max': 0.5, 'domain_weight': 0.05,'class_weight': True},
    'V8_Pretrain100':     {'arch': MLP_128_64,   'lam_max': 0.5, 'domain_weight': 0.1, 'class_weight': True, 'pretrain_epochs': 100},
}


if __name__ == '__main__':
    # Table 4.9 (WESAD-Full) and Table 4.10 (WESAD Wrist-Only) "DANN + IG +
    # SMOTE" rows -- V0_Baseline only (the other 8 VARIANTS above are an
    # unrelated architecture ablation, not exercised here).
    CONFIGS = [
        ('all-137 (OLD schema)',          '137'),
        ('wrist-55 (current schema)',     'wrist55'),
    ]

    cfg = VARIANTS['V0_Baseline']
    all_results = []
    for label, mode in CONFIGS:
        print(f'\n{"#"*70}')
        print(f'  DANN + IG + SMOTE (V0_Baseline) -- {label}')
        print(f'{"#"*70}')

        if mode == '137':
            X_df, y, groups, feat_cols = load_wesad_137()
            use_cols = feat_cols
        else:
            X_df, y, groups, feat_cols = load_wesad_tabular()
            use_cols = [c for c in feat_cols if c.startswith(WRIST_PREFIXES_55)]
            assert len(use_cols) == 55, (
                f'Expected 55 wrist features (thesis Section 4.3 canonical definition), '
                f'got {len(use_cols)} -- see DATA.md.'
            )

        print(f'{len(use_cols)} features, {len(np.unique(groups))} subjects\n')
        r = run_variant(X_df, y, groups, use_cols, cfg,
                         label=f'DANN+IG+SMOTE (V0_Baseline) -- {label}')
        all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results',
                                            'dann_tabular_ig_smote_full_wrist.json'))
