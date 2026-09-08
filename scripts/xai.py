"""
XAI — SHAP / feature importance for ALL models on ALL datasets.

Explainers used per model type:
  RF, ET, XGB, LGBM, Ensemble  → shap.TreeExplainer     (exact, fast)
  LogReg                        → shap.LinearExplainer   (exact, fast)
  SVM                           → permutation importance (no SHAP, slow)
  DANN tabular (WDD/Anxolotl)   → shap.DeepExplainer
  DANN sequence (WESAD)         → shap.GradientExplainer (channel-level)
  1D-CNN / CNN-GRU / RNN /
    Transformer / Mamba         → shap.GradientExplainer (channel-level)
  AutoGluon                     → predictor.feature_importance()

For each model × dataset × feature-set:
  • bar.png        — top-20 feature importance
  • beeswarm.png   — direction + distribution  (tabular only)
  • groups.png     — signal-group aggregated importance

Cross-dataset section:
  WESAD-trained models explained on WDD + Anxolotl predictions.

Output: figures/shap/
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sklearn_perm
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from common import (load_wdd, load_anxolotl, load_wesad_seq,
                    wrist_cols, bl_norm_labels)

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'figures', 'shap')
MODELS_DIR = os.path.join(ROOT, 'models_data')
os.makedirs(FIG_DIR, exist_ok=True)

# Thesis Figures 4.7/4.8 (WESAD SHAP, Section 4.3.6) and the WESAD-side of the 40-common
# cross-dataset SHAP below were computed on the OLD 137-feat/8-stat WESAD schema -- NOT
# the current 130-feat processed_data/wesad/processed_data.pkl (regenerated 2026-07-17).
# See DATA.md. Do not point this back at load_wesad_tabular() without re-checking that.
WESAD_SNAPSHOT = os.path.join(ROOT, 'processed_data', 'wesad_archive',
                               'processed_data_137feat_snapshot.pkl')

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Signal groups ─────────────────────────────────────────────────────────────

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'EDA':   lambda c: c.startswith('eda'),
    'TEMP':  lambda c: c.startswith('temp'),
    'HR':    lambda c: c.startswith('hr_') or c.startswith('ecg_hr'),
    'IBI':   lambda c: c.startswith('ibi') or c.startswith('ecg_ibi'),
    'RMSSD': lambda c: c.startswith('rmssd') or c.startswith('ecg_rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn')  or c.startswith('ecg_sdnn'),
    'RESP':  lambda c: c.startswith('resp'),
    'EMG':   lambda c: c.startswith('emg'),
    'DEMO':  lambda c: c in ('age','gender','coffee_today','coffee_last_hour',
                              'sports','smoker','ill','stai_baseline','stai_stress'),
}

def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col): return name
    return 'OTHER'

SEQ_CHANNEL_NAMES = ['ACC_wrist','EDA_wrist','TEMP_wrist',
                     'HR_wrist', 'IBI_wrist','RMSSD_wrist','SDNN_wrist',
                     'ACC_chest','EDA_chest','ECG_HR','ECG_IBI',
                     'ECG_RMSSD','ECG_SDNN','RESP_freq','RESP_iri','EMG_chest']

# ── Preprocessing helpers ─────────────────────────────────────────────────────

def prep_std(X_df, feat_cols):
    X = X_df[feat_cols].values.astype(float)
    imp = SimpleImputer(strategy='mean'); sc = StandardScaler()
    Xn = imp.fit_transform(sc.fit_transform(X))
    return Xn, imp, sc

def prep_bl(X_df, y, groups, feat_cols):
    X = X_df[feat_cols].values.astype(float)
    X = bl_norm_labels(X, y, groups)
    imp = SimpleImputer(strategy='mean')
    return imp.fit_transform(X), imp

def stratified_sample(X, y, n=1200):
    rng = np.random.default_rng(42)
    idx0 = np.where(y==0)[0]; idx1 = np.where(y==1)[0]
    n0 = min(n//2, len(idx0)); n1 = min(n//2, len(idx1))
    idx = np.concatenate([rng.choice(idx0,n0,replace=False),
                          rng.choice(idx1,n1,replace=False)])
    return X[idx], y[idx], idx

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _safe(tag):
    return tag.replace(' ','_').replace('/','_').replace('(','').replace(')','')

def save(fname):
    plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()
    print(f'    → {os.path.basename(fname)}')

def bar_plot(imp_vals, feat_names, title, fname, top=20):
    order = np.argsort(imp_vals)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feat_names)[order][::-1],
            imp_vals[order][::-1], color='steelblue', alpha=0.85)
    ax.set_xlabel('Importance'); ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); save(fname)

def groups_plot(imp_vals, feat_names, title, fname):
    g = {}
    for col, v in zip(feat_names, imp_vals):
        k = assign_group(col); g[k] = g.get(k,0)+v
    g = {k:v for k,v in g.items() if v>0}
    order = sorted(g, key=g.get, reverse=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(order, [g[k] for k in order],
           color=plt.cm.tab10(np.linspace(0,1,len(order))), alpha=0.85)
    ax.set_ylabel('Summed Importance'); ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); save(fname)

def beeswarm_plot(sv, X_s, feat_names, title, fname, max_d=20):
    try:
        expl = shap.Explanation(values=sv, data=X_s,
                                feature_names=list(feat_names))
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(expl, max_display=max_d, show=False)
        plt.title(title, pad=10); plt.tight_layout(); save(fname)
    except Exception as e:
        plt.close(); print(f'    [beeswarm skip] {e}')

def all_plots(imp_vals, sv, X_s, feat_names, tag):
    p = os.path.join(FIG_DIR, _safe(tag))
    bar_plot(imp_vals, feat_names, f'Feature Importance — {tag}', f'{p}_bar.png')
    groups_plot(imp_vals, feat_names, f'Signal Groups — {tag}', f'{p}_groups.png')
    if sv is not None:
        beeswarm_plot(sv, X_s, feat_names, f'SHAP Beeswarm — {tag}', f'{p}_beeswarm.png')

# ── SHAP per model type ────────────────────────────────────────────────────────

def shap_tree(model, X, y, feat_cols, tag):
    """TreeExplainer: RF, ET, XGB, LGBM."""
    Xs, ys, _ = stratified_sample(X, y)
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(Xs)
        sv  = sv[1] if isinstance(sv,list) else (sv[:,:,1] if sv.ndim==3 else sv)
        imp = np.abs(sv).mean(axis=0)
        all_plots(imp, sv, Xs, feat_cols, tag)
    except Exception as e:
        print(f'    [TreeExplainer ERR] {e}')

def shap_ensemble_tree(model, X, y, feat_cols, tag):
    """Average TreeExplainer SHAP across VotingClassifier sub-models."""
    Xs, ys, _ = stratified_sample(X, y)
    svs = []
    for name, est in model.named_estimators_.items():
        try:
            exp = shap.TreeExplainer(est)
            sv  = exp.shap_values(Xs)
            sv  = sv[1] if isinstance(sv,list) else (sv[:,:,1] if sv.ndim==3 else sv)
            svs.append(sv)
        except Exception as e:
            print(f'    [Ensemble sub-model {name} skip] {e}')
    if not svs: return
    sv_avg = np.mean(svs, axis=0)
    imp = np.abs(sv_avg).mean(axis=0)
    all_plots(imp, sv_avg, Xs, feat_cols, tag)

def shap_linear(model, X, y, feat_cols, tag):
    """LinearExplainer for LogReg."""
    Xs, ys, _ = stratified_sample(X, y)
    try:
        exp = shap.LinearExplainer(model, Xs)
        sv  = exp.shap_values(Xs)
        sv  = sv[1] if isinstance(sv,list) else sv
        imp = np.abs(sv).mean(axis=0)
        all_plots(imp, sv, Xs, feat_cols, tag)
    except Exception as e:
        print(f'    [LinearExplainer ERR] {e}')

def shap_perm(model, X, y, feat_cols, tag, n_rep=20):
    """Permutation importance for SVM (no SHAP support)."""
    Xs, ys, _ = stratified_sample(X, y, n=600)
    try:
        r = sklearn_perm(model, Xs, ys, n_repeats=n_rep, random_state=42,
                         scoring='f1_macro', n_jobs=1)
        imp = np.clip(r.importances_mean, 0, None)
        all_plots(imp, None, None, feat_cols, tag)
    except Exception as e:
        print(f'    [Permutation ERR] {e}')

def shap_deep_tabular(model_nn, X, y, feat_cols, tag, n_bg=200, n_test=400):
    """DeepExplainer for tabular MLP/DANN."""
    rng = np.random.default_rng(42)
    idx0 = np.where(y==0)[0]; idx1 = np.where(y==1)[0]
    bg_idx = np.concatenate([rng.choice(idx0,min(n_bg//2,len(idx0)),replace=False),
                              rng.choice(idx1,min(n_bg//2,len(idx1)),replace=False)])
    ts_idx = rng.choice(len(X), min(n_test, len(X)), replace=False)

    Xbg = torch.FloatTensor(X[bg_idx])
    Xts = torch.FloatTensor(X[ts_idx])
    try:
        exp = shap.DeepExplainer(model_nn, Xbg)
        sv  = exp.shap_values(Xts)
        sv  = sv[1] if isinstance(sv,list) else sv
        imp = np.abs(sv).mean(axis=0)
        all_plots(imp, sv, X[ts_idx], feat_cols, tag)
    except Exception as e:
        print(f'    [DeepExplainer ERR] {e}; trying GradientExplainer…')
        try:
            exp2 = shap.GradientExplainer(model_nn, Xbg)
            sv2  = exp2.shap_values(Xts)
            sv2  = sv2[1] if isinstance(sv2,list) else sv2
            imp2 = np.abs(sv2).mean(axis=0)
            all_plots(imp2, sv2, X[ts_idx], feat_cols, tag)
        except Exception as e2:
            print(f'    [GradientExplainer ERR] {e2}')

def shap_seq_gradient(model_nn, X_seq, channel_names, tag, n_bg=200, n_test=400):
    """GradientExplainer for sequence models (WESAD). Returns channel-level importance."""
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_seq), min(n_bg, len(X_seq)), replace=False)
    ts_idx = rng.choice(len(X_seq), min(n_test, len(X_seq)), replace=False)
    Xbg = torch.FloatTensor(X_seq[bg_idx])
    Xts = torch.FloatTensor(X_seq[ts_idx])
    C   = X_seq.shape[1]
    cnames = channel_names[:C] if len(channel_names) >= C else [f'ch_{i}' for i in range(C)]
    try:
        exp = shap.GradientExplainer(model_nn, Xbg)
        sv  = exp.shap_values(Xts)  # (n, C, T) or list
        sv  = sv[1] if isinstance(sv,list) else sv
        # mean over timesteps → (n, C)
        if sv.ndim == 3: sv = sv.mean(axis=2)
        imp = np.abs(sv).mean(axis=0)
        p   = os.path.join(FIG_DIR, _safe(tag))
        bar_plot(imp, cnames, f'Channel Importance — {tag}', f'{p}_bar.png')
        # groups (channels map directly to signal groups)
        groups_plot(imp, cnames, f'Channel Groups — {tag}', f'{p}_groups.png')
    except Exception as e:
        print(f'    [GradientExplainer seq ERR] {e}')

def autogluon_importance(predictor, X_df, y, feat_cols, tag):
    """Use AutoGluon's built-in feature importance (permutation-based)."""
    try:
        eval_df = X_df[feat_cols].copy(); eval_df['label'] = np.array(y)
        fi = predictor.feature_importance(eval_df, num_shuffle_sets=5,
                                           subsample_size=1000)
        imp = fi['importance'].values
        cols = list(fi.index)
        imp_full = np.zeros(len(feat_cols))
        for c, v in zip(cols, imp):
            if c in feat_cols:
                imp_full[feat_cols.index(c)] = max(0, v)
        all_plots(imp_full, None, None, feat_cols, tag)
    except Exception as e:
        print(f'    [AutoGluon FI ERR] {e}')

# ── Build and train final models ──────────────────────────────────────────────

def train_all_models_std(X, y, smote=True):
    """Train all classical models on StandardScaler-normalised data."""
    if smote:
        # subsample before SMOTE to avoid OOM on high-dimensional datasets
        rng = np.random.default_rng(42)
        idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
        n_sub = min(3000, len(idx0), len(idx1))
        sub = np.concatenate([rng.choice(idx0, n_sub, replace=False),
                              rng.choice(idx1, n_sub, replace=False)])
        Xsm, ysm = SMOTE(random_state=42).fit_resample(X[sub], y[sub])
    else:
        Xsm, ysm = X, y

    models = {}
    print('    Training RF…');      m=RandomForestClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1); m.fit(X,y); models['RF']=m
    print('    Training ET…');      m=ExtraTreesClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1);  m.fit(X,y); models['ET']=m
    print('    Training LogReg…');  m=LogisticRegression(C=0.001,penalty='l1',solver='saga',max_iter=1000,tol=1e-4,random_state=42); m.fit(Xsm,ysm); models['LogReg']=m
    print('    Training SVM…');     m=SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,cache_size=500); m.fit(Xsm[:5000],ysm[:5000]); models['SVM']=m
    print('    Training XGBoost…'); m=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,n_jobs=-1,random_state=42,eval_metric='mlogloss',verbosity=0); m.fit(Xsm,ysm); models['XGBoost']=m
    print('    Training LightGBM…');m=LGBMClassifier(n_estimators=305,max_depth=4,learning_rate=0.195,num_leaves=37,min_child_samples=30,subsample=0.654,colsample_bytree=0.852,n_jobs=-1,random_state=42,verbose=-1); m.fit(Xsm,ysm); models['LightGBM']=m
    print('    Training Ensemble XGB+RF+LGBM…');
    ens=VotingClassifier(voting='soft',estimators=[
        ('xgb',XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,n_jobs=-1,random_state=42,eval_metric='mlogloss',verbosity=0)),
        ('rf', RandomForestClassifier(n_estimators=200,max_features='sqrt',max_depth=10,n_jobs=-1,random_state=42)),
        ('lgbm',LGBMClassifier(n_estimators=305,max_depth=4,learning_rate=0.195,num_leaves=37,min_child_samples=30,subsample=0.654,colsample_bytree=0.852,n_jobs=-1,random_state=42,verbose=-1)),
    ]); ens.fit(Xsm,ysm); models['Ensemble_XGB_RF_LGBM']=ens
    print('    Training Ensemble RF+ET…');
    ens2=VotingClassifier(voting='soft',estimators=[
        ('rf', RandomForestClassifier(n_estimators=1000,max_features=0.5,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=1000,max_features=0.5,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1)),
    ]); ens2.fit(X,y); models['Ensemble_RF_ET']=ens2
    print('    Training Ensemble ET+RF+HGB…');
    ens3=VotingClassifier(voting='hard',estimators=[
        ('et',  ExtraTreesClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1)),
        ('rf',  RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200,max_depth=4,learning_rate=0.05,random_state=42)),
    ]); ens3.fit(X,y); models['Ensemble_ET_RF_HGB']=ens3
    return models

# ── DANN tabular for WDD/Anxolotl ─────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256,128), out_dim=128, dr=0.3):
        super().__init__()
        ls=[]; prev=in_dim
        for h in hidden:
            ls+=[nn.Linear(prev,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(dr)]; prev=h
        ls.append(nn.Linear(prev,out_dim)); self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x)

class _DANNTabular(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.ext = _MLP(in_dim)
        self.cls = nn.Sequential(nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self, x): return self.cls(self.ext(x))

def train_dann_tabular(X, y, epochs=200):
    from torch.utils.data import DataLoader, TensorDataset
    n_feat = X.shape[1]
    w1 = float((y==0).sum())/max(float((y==1).sum()),1)
    model = _DANNTabular(n_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,w1]).to(DEVICE))
    dl = DataLoader(TensorDataset(torch.FloatTensor(X),torch.LongTensor(y)),
                    batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
    model.eval(); return model

# ── DL models for WESAD sequences ─────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv1d(C,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,256,3,padding=1),nn.BatchNorm1d(256),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.cls=nn.Sequential(nn.Flatten(),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,2))
    def forward(self,x): return self.cls(self.enc(x))

class RNN1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.rnn=nn.GRU(C,128,2,batch_first=True,dropout=0.3,bidirectional=True)
        self.cls=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,2))
    def forward(self,x): x=x.permute(0,2,1); out,_=self.rnn(x); return self.cls(out[:,-1,:])

class CNNGRUHybrid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv1d(C,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU())
        self.gru=nn.GRU(128,256,2,batch_first=True,dropout=0.2,bidirectional=True)
        self.cls=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,2))
    def forward(self,x):
        x=self.cnn(x); x=x.permute(0,2,1); out,_=self.gru(x); return self.cls(out[:,-1,:])

class TransformerClassifier(nn.Module):
    def __init__(self, C, seq_len=60, d_model=128, nhead=4):
        super().__init__()
        self.proj=nn.Linear(C,d_model)
        self.pos=nn.Parameter(torch.randn(1,seq_len,d_model))
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=256,dropout=0.2,batch_first=True)
        self.tf=nn.TransformerEncoder(enc,2)
        self.cls=nn.Sequential(nn.Linear(d_model,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,2))
    def forward(self,x):
        x=x.permute(0,2,1); x=self.proj(x)+self.pos[:,:x.size(1),:]; return self.cls(self.tf(x).mean(1))

class SSMBlock(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.A=nn.Parameter(torch.randn(d)); self.B=nn.Linear(d,d)
        self.C=nn.Linear(d,d); self.D=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
    def forward(self,x):
        A=torch.exp(-torch.exp(self.A)); h=torch.zeros(x.size(0),x.size(2),device=x.device)
        outs=[]
        for t in range(x.size(1)):
            h=A*h+self.B(x[:,t,:]); outs.append(self.C(h))
        return self.norm(torch.stack(outs,1)+self.D(x))

class SSMClassifier(nn.Module):
    def __init__(self, C, d_model=128):
        super().__init__()
        self.proj=nn.Linear(C,d_model)
        self.ssm=nn.ModuleList([SSMBlock(d_model) for _ in range(2)])
        self.drop=nn.Dropout(0.2)
        self.cls=nn.Sequential(nn.Linear(d_model,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,2))
    def forward(self,x):
        x=x.permute(0,2,1); x=self.proj(x)
        for l in self.ssm: x=self.drop(l(x))
        return self.cls(x.mean(1))

class DANNSeq(nn.Module):
    """Sequence-based DANN — FeatureExtractor (Transformer) + StressClassifier only."""
    def __init__(self, C):
        super().__init__()
        enc=nn.TransformerEncoderLayer(d_model=C,nhead=max(1,C//4) if C%4==0 else 1,
                                        dim_feedforward=64,dropout=0.1,batch_first=True)
        self.tf=nn.TransformerEncoder(enc,2)
        self.proj=nn.Linear(C,128)
        self.cls=nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        x=x.permute(0,2,1); x=self.tf(x).mean(1); return self.cls(self.proj(x))

def train_seq_model(model_cls, X_seq, y_seq, epochs=30):
    from torch.utils.data import DataLoader, TensorDataset
    from imblearn.over_sampling import SMOTE
    C,T = X_seq.shape[1], X_seq.shape[2]
    flat = X_seq.reshape(len(X_seq), C*T)
    Xr,yr = SMOTE(random_state=42).fit_resample(flat, y_seq)
    Xr = Xr.reshape(-1,C,T)
    model = model_cls(C).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    dl = DataLoader(TensorDataset(torch.FloatTensor(Xr),torch.LongTensor(yr)),
                    batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(xb),yb).backward(); opt.step()
    model.eval(); return model

# ── Dataset runners ───────────────────────────────────────────────────────────

def run_dataset_tabular(X_df, y_np, groups, feat_cols, ds_label, use_blnorm):
    """Run all tabular models + SHAP for one dataset × feature set."""
    print(f'\n  Preprocessing {ds_label} ({len(feat_cols)} features)…')
    if use_blnorm:
        Xn, imp = prep_bl(X_df, y_np, groups, feat_cols)
    else:
        Xn, imp, _ = prep_std(X_df, feat_cols)

    print(f'  Training classical models…')
    clfs = train_all_models_std(Xn, y_np)

    for mname, clf in clfs.items():
        print(f'  SHAP — {mname}')
        tag = f'{ds_label}_{mname}'
        if mname in ('RF', 'ET', 'XGBoost', 'LightGBM'):
            shap_tree(clf, Xn, y_np, feat_cols, tag)
        elif mname == 'LogReg':
            shap_linear(clf, Xn, y_np, feat_cols, tag)
        elif mname == 'SVM':
            shap_perm(clf, Xn, y_np, feat_cols, tag)
        elif mname in ('Ensemble_XGB_RF_LGBM', 'Ensemble_RF_ET', 'Ensemble_ET_RF_HGB'):
            shap_ensemble_tree(clf, Xn, y_np, feat_cols, tag)

    # DANN tabular
    print(f'  Training DANN tabular…')
    dann = train_dann_tabular(Xn, y_np)
    print(f'  SHAP — DANN')
    shap_deep_tabular(dann, Xn, y_np, feat_cols, f'{ds_label}_DANN')

    return Xn, imp


def run_wesad_seq(X_seq, y_seq, sel_idx, label):
    """Run all DL models on WESAD sequences, GradientExplainer at channel level."""
    X_sel = X_seq[:, sel_idx, :]
    C = X_sel.shape[1]
    ch_names = [SEQ_CHANNEL_NAMES[i] if i < len(SEQ_CHANNEL_NAMES) else f'ch_{i}' for i in sel_idx]

    for mname, cls in [('1D-CNN', CNN1D), ('CNN-GRU', CNNGRUHybrid),
                        ('RNN', RNN1D), ('Transformer', TransformerClassifier),
                        ('Mamba', SSMClassifier), ('DANN', DANNSeq)]:
        print(f'  Training {mname} on WESAD sequences ({C} channels)…')
        try:
            model = train_seq_model(cls, X_sel, y_seq, epochs=20)
            print(f'  SHAP (gradient) — {mname}')
            shap_seq_gradient(model, X_sel, ch_names, f'WESAD_{label}_{mname}')
        except Exception as e:
            print(f'    [{mname} ERR] {e}')

    # Load saved RNN (trained on all WESAD data) if available
    rnn_pt = os.path.join(MODELS_DIR, 'rnn_wrist_final.pt')
    if os.path.exists(rnn_pt):
        print(f'  Loading saved RNN…')
        try:
            ckpt = torch.load(rnn_pt, map_location='cpu', weights_only=False)
            saved_idx = ckpt.get('selected_idx', sel_idx)
            X_saved = X_seq[:, saved_idx, :]
            saved_ch = [SEQ_CHANNEL_NAMES[i] if i < len(SEQ_CHANNEL_NAMES)
                        else f'ch_{i}' for i in saved_idx]
            rnn_saved = RNN1D(len(saved_idx)).to(DEVICE)
            rnn_saved.load_state_dict(ckpt['model_state_dict'])
            rnn_saved.eval()
            shap_seq_gradient(rnn_saved, X_saved, saved_ch, 'WESAD_wrist_RNN_saved')
        except Exception as e:
            print(f'    [saved RNN ERR] {e}')

    # Load saved DANN if available
    dann_pt = os.path.join(MODELS_DIR, 'dann_wrist_final.pt')
    if os.path.exists(dann_pt):
        print(f'  Loading saved DANN…')
        try:
            ckpt = torch.load(dann_pt, map_location='cpu', weights_only=False)
            # saved DANN uses 5-channel wrist sequences
            wrist_idx = [0, 5, 6, 7, 8]
            X_wrist = X_seq[:, wrist_idx, :]
            wrist_ch = [SEQ_CHANNEL_NAMES[i] for i in wrist_idx]
            dann_saved = DANNSeq(len(wrist_idx)).to(DEVICE)
            # try loading — architecture may differ, so wrap in try
            dann_saved.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            dann_saved.eval()
            shap_seq_gradient(dann_saved, X_wrist, wrist_ch, 'WESAD_wrist_DANN_saved')
        except Exception as e:
            print(f'    [saved DANN ERR] {e}')


def run_autogluon_wesad(X_df, y, feat_cols, tag):
    """AutoGluon feature importance (WESAD-trained predictor)."""
    ag_path = os.path.join(MODELS_DIR, 'autogluon_wrist_final')
    if not os.path.exists(ag_path):
        print(f'  [SKIP] AutoGluon model not found at {ag_path}'); return
    try:
        from autogluon.tabular import TabularPredictor
        pred = TabularPredictor.load(ag_path)
        print(f'  AutoGluon feature importance…')
        autogluon_importance(pred, X_df, y, feat_cols, tag)
    except Exception as e:
        print(f'  [AutoGluon ERR] {e}')


def run_autogluon_train(X_arr, y, feat_cols, ds_label, use_blnorm,
                         X_df_raw=None, groups=None, time_limit=120):
    """Train AutoGluon on full dataset and compute feature importance."""
    import pandas as pd
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print('  [SKIP] autogluon not available'); return

    print(f'  Training AutoGluon for {ds_label}…')
    try:
        train_df = pd.DataFrame(X_arr, columns=feat_cols)
        train_df['label'] = y
        pred = TabularPredictor(label='label', eval_metric='f1_macro',
                                verbosity=0).fit(train_df, time_limit=time_limit,
                                                  presets='good_quality')
        print(f'  AutoGluon feature importance — {ds_label}')
        autogluon_importance(pred, pd.DataFrame(X_arr, columns=feat_cols),
                             y, feat_cols, f'{ds_label}_AutoGluon')
    except Exception as e:
        print(f'  [AutoGluon train ERR] {e}')


# ── Cross-dataset signal-group comparison ─────────────────────────────────────

def cross_dataset_groups(wesad_Xn, wesad_y, wdd_Xn, wdd_y, anx_Xn, anx_y,
                          feat_cols, label):
    """
    Fit RF+ET on WESAD, compute signal-group importance on all 3 datasets.
    One overlaid bar chart per model.
    """
    for mname, mfn in [('RF', lambda: RandomForestClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1)),
                        ('ET', lambda: ExtraTreesClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1))]:
        print(f'  Cross-dataset groups — {mname}')
        clf = mfn(); clf.fit(wesad_Xn, wesad_y)
        exp = shap.TreeExplainer(clf)

        def group_imp(X_samp):
            sv = exp.shap_values(X_samp)
            sv = sv[1] if isinstance(sv,list) else (sv[:,:,1] if sv.ndim==3 else sv)
            ma = np.abs(sv).mean(axis=0)
            g = {}
            for c,v in zip(feat_cols, ma): k=assign_group(c); g[k]=g.get(k,0)+v
            return g

        Xw,_,_ = stratified_sample(wesad_Xn, wesad_y, 800)
        Xd,_,_ = stratified_sample(wdd_Xn,   wdd_y,   800)
        Xa,_,_ = stratified_sample(anx_Xn,   anx_y,   800)

        try:
            gw=group_imp(Xw); gd=group_imp(Xd); ga=group_imp(Xa)
        except Exception as e:
            print(f'    [cross ERR] {e}'); continue

        all_g = sorted(set(gw)|set(gd)|set(ga), key=lambda k: gw.get(k,0), reverse=True)
        x=np.arange(len(all_g)); bw=0.25
        fig,ax=plt.subplots(figsize=(10,5))
        ax.bar(x-bw,[gw.get(k,0) for k in all_g],bw,label='WESAD',color='steelblue',alpha=0.85)
        ax.bar(x,   [gd.get(k,0) for k in all_g],bw,label='WDD',  color='darkorange',alpha=0.85)
        ax.bar(x+bw,[ga.get(k,0) for k in all_g],bw,label='Anxolotl',color='green',alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(all_g)
        ax.set_ylabel('Sum |SHAP|'); ax.legend(framealpha=0.8)
        ax.set_title(f'Signal Group Importance Across Datasets — {mname} ({label})')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fname = os.path.join(FIG_DIR, f'cross_{_safe(label)}_{mname}_groups.png')
        save(fname)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading datasets…')
    print(f'  WESAD <- {WESAD_SNAPSHOT} (old 137-feat schema)')
    with open(WESAD_SNAPSHOT, 'rb') as f:
        _d = pickle.load(f)
    X_wesad = _d['X'] if isinstance(_d['X'], pd.DataFrame) else pd.DataFrame(_d['X'], columns=_d['feature_cols'])
    y_wesad = _d['y']; g_wesad = _d['groups']; fc_wesad = list(_d['feature_cols'])
    assert len(fc_wesad) == 137, (
        f'Expected 137 features from the old-schema WESAD snapshot (thesis Figs 4.7/4.8), '
        f'got {len(fc_wesad)} -- see DATA.md.'
    )
    X_wdd,   y_wdd,   g_wdd,   fc_wdd   = load_wdd()
    X_anx,   y_anx,   g_anx,   fc_anx   = load_anxolotl()

    w_cols = wrist_cols(fc_wesad)
    assert len(w_cols) == 56, (
        f'Expected 56 wrist features from the old-schema WESAD snapshot, got {len(w_cols)} -- see DATA.md.'
    )
    fc_60  = sorted(set(fc_wdd)   & set(fc_anx))
    fc_40  = sorted(set(fc_wesad) & set(fc_anx))
    assert len(fc_40) == 40, (
        f'Expected 40 WESAD∩Anxolotl common features (old schema), got {len(fc_40)} -- see DATA.md.'
    )
    y_w = np.array(y_wesad); y_d = np.array(y_wdd); y_a = np.array(y_anx)

    # ── WESAD tabular ──────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  WESAD — wrist features (tabular)')
    print('='*60)
    Xw_wrist, _ = run_dataset_tabular(X_wesad, y_w, g_wesad, w_cols,
                                        'WESAD_wrist', use_blnorm=False)[:2]

    print('\n' + '='*60)
    print('  WESAD — all features (tabular)')
    print('='*60)
    Xw_all, _ = run_dataset_tabular(X_wesad, y_w, g_wesad, fc_wesad,
                                     'WESAD_all', use_blnorm=False)[:2]

    run_autogluon_wesad(X_wesad, y_w, w_cols, 'WESAD_wrist_AutoGluon')
    run_autogluon_wesad(X_wesad, y_w, fc_wesad, 'WESAD_all_AutoGluon')

    # ── WESAD sequences (DL: todos os modelos) ────────────────────────────────
    print('\n' + '='*60)
    print('  WESAD — sequences (1D-CNN, CNN-GRU, RNN, Transformer, Mamba, DANN)')
    print('='*60)
    try:
        X_seq, y_seq, _ = load_wesad_seq()
        SEQ_WRIST_IDX = [0, 5, 6, 7, 8]
        run_wesad_seq(X_seq, y_seq, SEQ_WRIST_IDX, 'wrist5ch')
        run_wesad_seq(X_seq, y_seq, list(range(16)), 'all16ch')
    except Exception as e:
        print(f'  [seq load ERR] {e}')

    # ── WDD ────────────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  WDD — 60 common features')
    print('='*60)
    Xd_60, _ = run_dataset_tabular(X_wdd, y_d, g_wdd, fc_60,
                                    'WDD_60common', use_blnorm=True)[:2]

    print('\n' + '='*60)
    print('  WDD — 90 all features')
    print('='*60)
    Xd_90, _ = run_dataset_tabular(X_wdd, y_d, g_wdd, fc_wdd, 'WDD_90all', use_blnorm=True)
    run_autogluon_train(Xd_60, y_d, fc_60, 'WDD_60common', use_blnorm=True)
    run_autogluon_train(Xd_90, y_d, fc_wdd, 'WDD_90all', use_blnorm=True)

    # ── Anxolotl ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  Anxolotl — 40 common features')
    print('='*60)
    Xa_40, _ = run_dataset_tabular(X_anx, y_a, g_anx, fc_40,
                                    'Anxolotl_40common', use_blnorm=True)[:2]

    print('\n' + '='*60)
    print('  Anxolotl — 60 all features')
    print('='*60)
    Xa_60, _ = run_dataset_tabular(X_anx, y_a, g_anx, fc_anx, 'Anxolotl_60all', use_blnorm=True)
    run_autogluon_train(Xa_40, y_a, fc_40, 'Anxolotl_40common', use_blnorm=True)
    run_autogluon_train(Xa_60, y_a, fc_anx, 'Anxolotl_60all', use_blnorm=True)

    # ── Cross-dataset signal groups ────────────────────────────────────────────
    print('\n' + '='*60)
    print('  Cross-dataset comparison (WESAD-trained → WDD + Anxolotl)')
    print('='*60)

    # 40 common features (WESAD × Anxolotl intersection)
    Xd_40, _ = prep_bl(X_wdd, y_d, g_wdd, fc_40)
    Xw_40, _ = prep_std(X_wesad, fc_40)
    print('  40 common features…')
    cross_dataset_groups(Xw_40, y_w, Xd_40, y_d, Xa_40, y_a,
                          fc_40, '40common')

    print('\nDone. All SHAP figures in:', FIG_DIR)
