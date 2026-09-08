"""
Estima o tempo total do run_all.sh.
Mede 1 fold de cada modelo com dados reais e extrapola.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from common import (load_wesad_tabular, load_wdd, load_anxolotl, load_wesad_seq,
                    wrist_cols, bl_norm_labels)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── helpers ───────────────────────────────────────────────────────────────────

def first_fold(X_arr, y, groups):
    tr, te = next(LeaveOneGroupOut().split(X_arr, y, groups))
    return tr, te

def std_prep(Xtr, Xte):
    imp = SimpleImputer(strategy='mean'); sc = StandardScaler()
    return imp.fit_transform(sc.fit_transform(Xtr)), imp.transform(sc.transform(Xte))

def bl_prep(Xtr_raw, ytr, gtr):
    X = bl_norm_labels(Xtr_raw, ytr, gtr)
    imp = SimpleImputer(strategy='mean')
    return imp.fit_transform(X)

def smote(X, y):
    try:    return SMOTE(random_state=42).fit_resample(X, y)
    except: return X, y

def t(fn):
    t0 = time.time(); fn(); return time.time() - t0

def make_ensemble():
    return VotingClassifier(voting='soft', estimators=[
        ('xgb',  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                n_jobs=-1, random_state=42, eval_metric='mlogloss', verbosity=0)),
        ('rf',   RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                         max_depth=10, n_jobs=-1, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=305, max_depth=4, num_leaves=37,
                                  n_jobs=-1, random_state=42, verbose=-1)),
    ])

def train_dl(model, X_seq_tr, y_tr, epochs=20, bs=64):
    C, T = X_seq_tr.shape[1], X_seq_tr.shape[2]
    flat = X_seq_tr.reshape(len(X_seq_tr), C*T)
    Xr, yr = smote(flat, y_tr)
    Xr = Xr.reshape(-1, C, T)
    dl = DataLoader(TensorDataset(torch.FloatTensor(Xr), torch.LongTensor(yr)),
                    batch_size=bs, shuffle=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(xb), yb).backward(); opt.step()

def train_dann_tab(X, y, epochs=300):
    w1 = float((y==0).sum()) / max(float((y==1).sum()), 1)
    net = nn.Sequential(
        nn.Linear(X.shape[1],256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128,2))
    net = net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1., w1]).to(DEVICE))
    dl = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                    batch_size=256, shuffle=True)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()

# DL model classes (inline, same as scripts)
class CNN1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.e=nn.Sequential(nn.Conv1d(C,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
                              nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.c=nn.Sequential(nn.Flatten(),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,2))
    def forward(self,x): return self.c(self.e(x))

class RNN1D(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.r=nn.GRU(C,64,2,batch_first=True,dropout=0.3,bidirectional=True)
        self.c=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,2))
    def forward(self,x): x=x.permute(0,2,1); o,_=self.r(x); return self.c(o[:,-1,:])

class CNNGRU(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv1d(C,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
                                nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU())
        self.gru=nn.GRU(64,128,2,batch_first=True,dropout=0.2,bidirectional=True)
        self.c=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,2))
    def forward(self,x): x=self.cnn(x).permute(0,2,1); o,_=self.gru(x); return self.c(o[:,-1,:])

class Transformer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.p=nn.Linear(C,64); self.pos=nn.Parameter(torch.randn(1,60,64))
        enc=nn.TransformerEncoderLayer(64,4,dim_feedforward=128,dropout=0.2,batch_first=True)
        self.tf=nn.TransformerEncoder(enc,2)
        self.c=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,2))
    def forward(self,x): x=x.permute(0,2,1); x=self.p(x)+self.pos[:,:x.size(1),:]; return self.c(self.tf(x).mean(1))

class Mamba(nn.Module):
    def __init__(self, C, d=64):
        super().__init__()
        self.p=nn.Linear(C,d); self.A=nn.Parameter(torch.randn(d))
        self.B=nn.Linear(d,d); self.C=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
        self.cls=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):
        x=x.permute(0,2,1); x=self.p(x); A=torch.exp(-torch.exp(self.A))
        h=torch.zeros(x.size(0),x.size(2),device=x.device); outs=[]
        for t in range(x.size(1)):
            h=A*h+self.B(x[:,t,:]); outs.append(self.C(h))
        return self.cls(self.norm(torch.stack(outs,1)).mean(1))

# ── Load data ─────────────────────────────────────────────────────────────────
print('Carregando dados…')
X_w, y_w, g_w, fc_w = load_wesad_tabular()
X_d, y_d, g_d, fc_d = load_wdd()
X_a, y_a, g_a, fc_a = load_anxolotl()
y_w=np.array(y_w); y_d=np.array(y_d); y_a=np.array(y_a)
w_cols=wrist_cols(fc_w); fc_60=sorted(set(fc_d)&set(fc_a)); fc_40=sorted(set(fc_w)&set(fc_a))

try:
    X_seq, y_seq, g_seq = load_wesad_seq(); has_seq=True
except: has_seq=False

print(f'  WESAD {X_w.shape}  WDD {X_d.shape}  Anxolotl {X_a.shape}  device={DEVICE}\n')

rows = []   # (label, n_folds, n_feat_sets, t_per_fold_s)

# ── WESAD tabular ─────────────────────────────────────────────────────────────
trw,tew=first_fold(X_w[w_cols].values, y_w, g_w)
Xwn,_=std_prep(X_w[w_cols].values[trw].astype(float), X_w[w_cols].values[tew].astype(float))
yw=y_w[trw]; Xwsm,ywsm=smote(Xwn,yw)

for nm,fn in [
    ('LogReg',   lambda:LogisticRegression(C=0.001,penalty='l1',solver='saga',max_iter=1000,tol=1e-4,random_state=42).fit(Xwsm,ywsm)),
    ('SVM',      lambda:SVC(C=0.1,kernel='poly',gamma='auto',degree=3,class_weight='balanced',probability=True).fit(Xwsm[:4000],ywsm[:4000])),
    ('RF',       lambda:RandomForestClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xwn,yw)),
    ('ET',       lambda:ExtraTreesClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xwn,yw)),
    ('Ensemble', lambda:make_ensemble().fit(Xwsm,ywsm)),
]:
    tf=t(fn); rows.append((f'WESAD {nm}',15,2,tf))
    print(f'  WESAD {nm:<12} {tf:5.1f}s/fold  → est. {tf*15*2/60:.0f}min total')

# WESAD DANN tabular
tf=t(lambda:train_dann_tab(Xwn,yw,epochs=300))
rows.append(('WESAD DANN tabular',15,2,tf))
print(f'  WESAD DANN tab   {tf:5.1f}s/fold  → est. {tf*15*2/60:.0f}min total')

# WESAD AutoGluon (use time_limit as proxy since 1 fold is capped)
rows.append(('WESAD AutoGluon (600s/fold)',15,2,600))
print(f'  WESAD AutoGluon  600.0s/fold  → est. {600*15*2/60:.0f}min total (time_limit)')

# ── WESAD sequences DL ────────────────────────────────────────────────────────
if has_seq:
    trs,tes=first_fold(X_seq, y_seq, g_seq)
    Xs5=X_seq[trs][:, [0,5,6,7,8], :]  # wrist 5ch
    ys=y_seq[trs]
    for nm,cls in [('1D-CNN',CNN1D),('CNN-GRU',CNNGRU),('RNN',RNN1D),
                    ('Transformer',Transformer),('Mamba',Mamba)]:
        m=cls(5).to(DEVICE)
        tf=t(lambda m=m,X=Xs5,y=ys:train_dl(m,X,y,epochs=20))
        # scale to 50 epochs (max with early stopping)
        tf50=tf*50/20
        rows.append((f'WESAD {nm} (seq)',15,2,tf50))
        print(f'  WESAD {nm:<12} {tf:.1f}s/20ep → {tf50:.0f}s/50ep  → est. {tf50*15*2/3600:.1f}h')

# ── WDD ───────────────────────────────────────────────────────────────────────
trd,ted=first_fold(X_d[fc_60].values, y_d, g_d)
Xdn=bl_prep(X_d[fc_60].values[trd].astype(float), y_d[trd], g_d[trd])
yd=y_d[trd]; Xdsm,ydsm=smote(Xdn,yd)

for nm,fn in [
    ('RF',       lambda:RandomForestClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xdn,yd)),
    ('ET',       lambda:ExtraTreesClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xdn,yd)),
    ('LogReg',   lambda:LogisticRegression(C=0.001,penalty='l1',solver='saga',max_iter=1000,tol=1e-4,random_state=42).fit(Xdsm,ydsm)),
    ('Ensemble', lambda:make_ensemble().fit(Xdsm,ydsm)),
]:
    tf=t(fn); rows.append((f'WDD {nm}',16,2,tf))
    print(f'  WDD  {nm:<12} {tf:5.1f}s/fold  → est. {tf*16*2/60:.0f}min total')

tf=t(lambda:train_dann_tab(Xdn,yd,epochs=300))
rows.append(('WDD DANN',16,2,tf))
print(f'  WDD  DANN        {tf:5.1f}s/fold  → est. {tf*16*2/60:.0f}min total')
rows.append(('WDD AutoGluon (120s/fold)',16,2,120))
print(f'  WDD  AutoGluon   120.0s/fold  → est. {120*16*2/60:.0f}min total (time_limit)')

# ── Anxolotl ──────────────────────────────────────────────────────────────────
tra,tea=first_fold(X_a[fc_40].values, y_a, g_a)
Xan=bl_prep(X_a[fc_40].values[tra].astype(float), y_a[tra], g_a[tra])
ya=y_a[tra]; Xasm,yasm=smote(Xan,ya)

for nm,fn in [
    ('RF',       lambda:RandomForestClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xan,ya)),
    ('ET',       lambda:ExtraTreesClassifier(n_estimators=1000,max_features=0.5,class_weight='balanced',random_state=42,n_jobs=-1).fit(Xan,ya)),
    ('LogReg',   lambda:LogisticRegression(C=0.001,penalty='l1',solver='saga',max_iter=1000,tol=1e-4,random_state=42).fit(Xasm,yasm)),
    ('Ensemble', lambda:make_ensemble().fit(Xasm,yasm)),
]:
    tf=t(fn); rows.append((f'Anxolotl {nm}',37,2,tf))
    print(f'  Anx  {nm:<12} {tf:5.1f}s/fold  → est. {tf*37*2/60:.0f}min total')

tf=t(lambda:train_dann_tab(Xan,ya,epochs=300))
rows.append(('Anxolotl DANN',37,2,tf))
print(f'  Anx  DANN        {tf:5.1f}s/fold  → est. {tf*37*2/60:.0f}min total')
rows.append(('Anxolotl AutoGluon (120s)',37,2,120))
print(f'  Anx  AutoGluon   120.0s/fold  → est. {120*37*2/60:.0f}min total (time_limit)')

# ── Sumário ───────────────────────────────────────────────────────────────────
print('\n')
print('='*72)
print('  ESTIMATIVA TOTAL — run_all.sh')
print('='*72)
print(f'  {"Fase / Modelo":<42} {"t/fold":>7}  {"folds×feat":>10}  {"Total":>9}')
print('-'*72)

total_s = 0
for label, n_folds, n_feat, tf in rows:
    tot = tf * n_folds * n_feat; total_s += tot
    hh=int(tot)//3600; mm=(int(tot)%3600)//60
    print(f'  {label:<42} {tf:>6.0f}s  {n_folds}×{n_feat:>2}={n_folds*n_feat:>3}  '
          f'  {hh:>2}h{mm:02d}m')

extras = [('Cross-dataset (v4 + blnorm + DANN)', 3600),
          ('XAI / SHAP (todos os modelos)',       7200),
          ('Gráficos',                             300)]
print('-'*72)
for lbl, s in extras:
    total_s += s
    print(f'  {lbl:<42}        —        —  '
          f'  {s//3600:>2}h{(s%3600)//60:02d}m')

print('='*72)
hh=int(total_s)//3600; mm=(int(total_s)%3600)//60
print(f'  TOTAL ESTIMADO: {hh}h {mm:02d}min   (device: {DEVICE})')
if str(DEVICE) == 'cpu':
    print(f'  Com GPU (CUDA): ~{max(1,hh//4)}-{max(2,hh//3)}h  (DL 4-6× mais rápido)')
print('='*72)
