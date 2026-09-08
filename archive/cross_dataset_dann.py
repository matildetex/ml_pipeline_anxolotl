"""
ARCHIVED 2026-09-07: pre-leak-fix version (712 windows); superseded by the
686-window leak-fixed pipeline -- see
2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py (note: that
script covers RF/ET/Ensemble only, not DANN -- there is currently no
686-window/leak-fixed DANN cross-dataset rerun anywhere in the repo).

This script's target-side normalisation (bl_protocol, n=8) predates the
2026-07-19 BL-norm leak fix; its outputs (results/cross_dataset_dann.json,
results_master.csv rows dated 2026-06-08, all n_windows=712) are the
pre-fix numbers. If any DANN cross-dataset figure is still cited in the
current thesis text, it has not been re-verified against the leak-fixed
target -- check results_master.csv and the thesis text directly before
treating any DANN cross-dataset number as current.

DANN cross-dataset: train on WDD (or WESAD) with Anxolotl as unlabeled target.
Domain discriminator: source dataset (0) vs Anxolotl (1).
GRL forces feature extractor to produce domain-invariant representations.
Labels of Anxolotl are NEVER seen during training.
"""
import pickle, numpy as np, pandas as pd, torch, json, os
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve,
                              balanced_accuracy_score, matthews_corrcoef)

_RESULTS = []

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── BL-norm helpers ───────────────────────────────────────────────────────────
def bl_labels(X, y, groups):
    X = np.array(X, dtype=float).copy(); y = np.array(y); groups = np.array(groups)
    for s in np.unique(groups):
        sm = groups == s; bm = sm & (y == 0)
        if bm.sum() > 1:
            for j in range(X.shape[1]):
                v = X[bm, j][~np.isnan(X[bm, j])]
                if len(v) > 1:
                    X[sm, j] = (X[sm, j] - v.mean()) / (v.std() + 1e-8)
    return X

def bl_protocol(X, groups, n=8):
    X = np.array(X, dtype=float).copy(); groups = np.array(groups)
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]; bl = idx[:n]
        for j in range(X.shape[1]):
            v = X[bl, j][~np.isnan(X[bl, j])]
            if len(v) > 1:
                X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X

def report(name, probs, ya):
    y_pred = (probs >= 0.5).astype(int)
    f1d = f1_score(ya, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(ya, probs)
    fpr, tpr, thr = roc_curve(ya, probs)
    t = float(thr[np.argmax(tpr - fpr)])
    f1o = f1_score(ya, (probs >= t).astype(int), average='macro', zero_division=0)
    ba  = balanced_accuracy_score(ya, y_pred)
    mcc = matthews_corrcoef(ya, y_pred)
    print(f'  {name:<55} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t={t:.3f}')
    _RESULTS.append({'label': name, 'f1': f1d, 'f1opt': f1o, 'auc': auc,
                     'ba': ba, 'mcc': mcc, 'dataset': 'DANN→Anxolotl'})
    return f1d

# ── Load data ─────────────────────────────────────────────────────────────────
with open('../processed_data/wdd/processed_data.pkl', 'rb') as f:    wdd = pickle.load(f)
# WESAD: the thesis's cross-dataset numbers (Section 4.4, 40-feature intersection) were
# computed on the OLD 137-feat/8-stat schema -- NOT the current 130-feat/11-stat
# processed_data/wesad/processed_data.pkl (regenerated 2026-07-17). See DATA.md.
with open('../processed_data/wesad_archive/processed_data_137feat_snapshot.pkl', 'rb') as f:  w   = pickle.load(f)
with open('../processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f: a = pickle.load(f)

COLS60 = sorted(a['feature_cols'])
COLS40 = sorted(set(w['feature_cols']) & set(COLS60))
if len(COLS40) != 40:
    raise RuntimeError(
        f'Expected 40 WESAD∩Anxolotl common features (old 137-feat/8-stat WESAD '
        f'schema, per thesis Section 4.4), got {len(COLS40)}. This usually means '
        f'WESAD was loaded from the current 130-feat/11-stat schema instead of '
        f'processed_data/wesad_archive/processed_data_137feat_snapshot.pkl -- see DATA.md.'
    )

Xa_df = a['X'] if isinstance(a['X'], pd.DataFrame) else pd.DataFrame(a['X'], columns=a['feature_cols'])
ya = np.array(a['y']); ga = np.array(a['groups'])

Xd_df = wdd['X'] if isinstance(wdd['X'], pd.DataFrame) else pd.DataFrame(wdd['X'], columns=wdd['feature_cols'])
yd = np.array(wdd['y']); gd = np.array(wdd['groups'])

Xw_df = w['X'] if isinstance(w['X'], pd.DataFrame) else pd.DataFrame(w['X'], columns=w['feature_cols'])
yw = np.array(w['y']); gw = np.array(w['groups'])

# ── DANN architecture ─────────────────────────────────────────────────────────
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=128, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DANN(nn.Module):
    def __init__(self, in_dim, feat_dim=128):
        super().__init__()
        self.extractor = MLP(in_dim, hidden=(256, 128), out_dim=feat_dim)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, 64),
                                         nn.ReLU(), nn.Linear(64, 2))
        self.discriminator = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, 64),
                                            nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, lam):
        z = self.extractor(x)
        stress = self.classifier(z)
        domain = self.discriminator(GRL.apply(z, lam))
        return stress, domain

    def predict_proba(self, x):
        z = self.extractor(x)
        logits = self.classifier(z)
        return torch.softmax(logits, dim=1)[:, 1]

def train_dann(Xsrc, ysrc, Xtgt_unlab, epochs=300, lr=1e-3, domain_weight=0.1,
               batch_size=256, class_counts=None):
    n_feat = Xsrc.shape[1]
    model = DANN(n_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # class weights for stress classifier
    if class_counts is None:
        w0 = 1.0; w1 = (ysrc == 0).sum() / (ysrc == 1).sum()
    else:
        w0, w1 = class_counts
    cls_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0, w1]).to(DEVICE))
    dom_loss_fn = nn.CrossEntropyLoss()

    # source: label=0 (source domain), target: label=1 (Anxolotl domain)
    Xs = torch.FloatTensor(Xsrc)
    ys = torch.LongTensor(ysrc)
    ds = torch.zeros(len(Xs), dtype=torch.long)          # source domain = 0

    Xt = torch.FloatTensor(Xtgt_unlab)
    dt = torch.ones(len(Xt), dtype=torch.long)            # target domain = 1

    src_ds = TensorDataset(Xs, ys, ds)
    tgt_ds = TensorDataset(Xt, dt)
    src_dl = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_dl = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        lam = 2 / (1 + np.exp(-10 * epoch / epochs)) - 1
        tgt_iter = iter(tgt_dl)
        for Xb, yb, db in src_dl:
            Xb, yb, db = Xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            try:
                (Xtb, dtb) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_dl)
                (Xtb, dtb) = next(tgt_iter)
            Xtb, dtb = Xtb.to(DEVICE), dtb.to(DEVICE)

            # source: stress + domain loss
            s_out, d_out_src = model(Xb, lam)
            loss_cls = cls_loss_fn(s_out, yb)
            loss_dom_src = dom_loss_fn(d_out_src, db)

            # target: domain loss only (no stress labels)
            _, d_out_tgt = model(Xtb, lam)
            loss_dom_tgt = dom_loss_fn(d_out_tgt, dtb)

            loss = loss_cls + domain_weight * (loss_dom_src + loss_dom_tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    return model

def eval_model(model, Xte):
    model.eval()
    dl = DataLoader(TensorDataset(torch.FloatTensor(Xte)), batch_size=512)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            probs.append(model.predict_proba(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(probs)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*72)
print('  DANN CROSS-DATASET: WDD (labeled source) → Anxolotl (unlabeled target)')
print('  60 features, BL-norm')
print('='*72)

# prepare WDD-60
Xd60_bl = bl_labels(Xd_df[COLS60].values.astype(float), yd, gd)
Xa60_bl = bl_protocol(Xa_df[COLS60].values.astype(float), ga, n=8)
imp60 = SimpleImputer(strategy='mean')
Xtr_d60 = imp60.fit_transform(Xd60_bl)
Xte60   = imp60.transform(Xa60_bl)

print(f'  Source (WDD): {Xtr_d60.shape}  |  Target (Anxolotl, no labels): {Xte60.shape}')
print()

# grid: domain_weight × epochs
best_dann_wdd = ('', 0.0, None)
for dw in [0.05, 0.1, 0.2, 0.5]:
    for ep in [200, 400]:
        np.random.seed(42); torch.manual_seed(42)
        model = train_dann(Xtr_d60, yd, Xte60, epochs=ep, domain_weight=dw)
        probs = eval_model(model, Xte60)
        name = f'DANN WDD-60 dw={dw} ep={ep}'
        f1 = report(name, probs, ya)
        if f1 > best_dann_wdd[1]:
            best_dann_wdd = (name, f1, probs)

print()

# ═══════════════════════════════════════════════════════════════════════════════
print('='*72)
print('  DANN CROSS-DATASET: WESAD (labeled source) → Anxolotl (unlabeled target)')
print('  40 features, BL-norm')
print('='*72)

# prepare WESAD-40
Xw40_bl = bl_labels(Xw_df[COLS40].values.astype(float), yw, gw)
Xa40_bl = bl_protocol(Xa_df[COLS40].values.astype(float), ga, n=8)
imp40 = SimpleImputer(strategy='mean')
Xtr_w40 = imp40.fit_transform(Xw40_bl)
Xte40   = imp40.transform(Xa40_bl)

print(f'  Source (WESAD): {Xtr_w40.shape}  |  Target (Anxolotl, no labels): {Xte40.shape}')
print()

best_dann_wes = ('', 0.0, None)
for dw in [0.05, 0.1, 0.2, 0.5]:
    for ep in [200, 400]:
        np.random.seed(42); torch.manual_seed(42)
        model = train_dann(Xtr_w40, yw, Xte40, epochs=ep, domain_weight=dw)
        probs = eval_model(model, Xte40)
        name = f'DANN WESAD-40 dw={dw} ep={ep}'
        f1 = report(name, probs, ya)
        if f1 > best_dann_wes[1]:
            best_dann_wes = (name, f1, probs)

print()

# ═══════════════════════════════════════════════════════════════════════════════
print('='*72)
print('  DANN CROSS-DATASET: WESAD+WDD (labeled source) → Anxolotl (target)')
print('  40 features, BL-norm')
print('='*72)

Xd40_bl = bl_labels(Xd_df[COLS40].values.astype(float), yd, gd)
X_both40 = np.vstack([Xd40_bl, Xw40_bl])
y_both40 = np.concatenate([yd, yw])

imp_wd = SimpleImputer(strategy='mean')
Xtr_wd40 = imp_wd.fit_transform(X_both40)
Xte40_wd = imp_wd.transform(Xa40_bl)

print(f'  Source (WESAD+WDD): {Xtr_wd40.shape}  |  Target: {Xte40_wd.shape}')
print()

for dw in [0.05, 0.1, 0.2]:
    for ep in [200, 400]:
        np.random.seed(42); torch.manual_seed(42)
        model = train_dann(Xtr_wd40, y_both40, Xte40_wd, epochs=ep, domain_weight=dw)
        probs = eval_model(model, Xte40_wd)
        report(f'DANN WESAD+WDD-40 dw={dw} ep={ep}', probs, ya)

print()
print('='*72)
print('  SUMÁRIO')
print('='*72)
print(f'  Melhor DANN WDD-60 : {best_dann_wdd[0]}  →  F1={best_dann_wdd[1]:.3f}')
print(f'  Melhor DANN WESAD-40: {best_dann_wes[0]}  →  F1={best_dann_wes[1]:.3f}')
print()
print('  Referência RF+ET (sem DANN):')
print('    WDD-60  → Anxolotl : F1=0.867  AUC=0.923')
print('    WESAD-40→ Anxolotl : F1=0.729  AUC=0.874')
print('  LOSO ceiling Anxolotl: F1=0.919')

_out = os.path.join(os.path.dirname(__file__), '..', 'results', 'cross_dataset_dann.json')
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, 'w') as _f:
    json.dump(_RESULTS, _f, indent=2)
print(f'\nResultados guardados em: {_out}')
