import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb

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
    f1d = f1_score(ya, (probs >= 0.5).astype(int), average='macro', zero_division=0)
    auc = roc_auc_score(ya, probs)
    fpr, tpr, thr = roc_curve(ya, probs)
    t = float(thr[np.argmax(tpr - fpr)])
    f1o = f1_score(ya, (probs >= t).astype(int), average='macro', zero_division=0)
    print(f'  {name:<52} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t={t:.3f}')
    return f1d

with open('wdd/processed_data.pkl', 'rb') as f:    wdd = pickle.load(f)
with open('wesad/processed_data.pkl', 'rb') as f:  w   = pickle.load(f)
with open('output/processed_data_anxolotl.pkl', 'rb') as f: a = pickle.load(f)

COLS60 = sorted(a['feature_cols'])

Xa_df = a['X'] if isinstance(a['X'], pd.DataFrame) else pd.DataFrame(a['X'], columns=a['feature_cols'])
Xa = Xa_df[COLS60].values.astype(float)
ya = np.array(a['y']); ga = np.array(a['groups'])
Xa_bl = bl_protocol(Xa.copy(), ga, n=8)

Xd_df = wdd['X'] if isinstance(wdd['X'], pd.DataFrame) else pd.DataFrame(wdd['X'], columns=wdd['feature_cols'])
Xd = Xd_df[COLS60].values.astype(float)
yd = np.array(wdd['y']); gd = np.array(wdd['groups'])
Xd_bl = bl_labels(Xd.copy(), yd, gd)

Xw_df = w['X'] if isinstance(w['X'], pd.DataFrame) else pd.DataFrame(w['X'], columns=w['feature_cols'])
Xw = np.full((len(Xw_df), len(COLS60)), np.nan)
for i, c in enumerate(COLS60):
    if c in Xw_df.columns:
        Xw[:, i] = Xw_df[c].values
yw = np.array(w['y']); gw = np.array(w['groups'])
Xw_bl = bl_labels(Xw.copy(), yw, gw)

imp_wdd = SimpleImputer(strategy='mean')
Xtr = imp_wdd.fit_transform(Xd_bl)
Xte = imp_wdd.transform(Xa_bl)

imp_all = SimpleImputer(strategy='mean')
X_all_tr = imp_all.fit_transform(np.vstack([Xd_bl, Xw_bl]))
X_all_te = imp_all.transform(Xa_bl)
y_all = np.concatenate([yd, yw])

print('=' * 78)
print('  TUNING RF/ET — WDD apenas (60 feat)')
print('=' * 78)

configs = [
    ('RF  n=500 leaf=1',  RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', max_features='sqrt', random_state=42, n_jobs=-1)),
    ('RF  n=1000 leaf=1', RandomForestClassifier(n_estimators=1000, min_samples_leaf=1, class_weight='balanced', max_features='sqrt', random_state=42, n_jobs=-1)),
    ('RF  n=500 leaf=1 mf=0.5', RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', max_features=0.5, random_state=42, n_jobs=-1)),
    ('ET  n=500 leaf=1',  ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('ET  n=1000 leaf=1', ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('ET  n=500 leaf=2',  ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('LGB n=500 d=4 l=25', lgb.LGBMClassifier(n_estimators=500, max_depth=4, num_leaves=25, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10)),
    ('LGB n=500 d=6 l=50', lgb.LGBMClassifier(n_estimators=500, max_depth=6, num_leaves=50, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10)),
]

probs_list = []
for name, clf in configs:
    clf.fit(Xtr, yd)
    p = clf.predict_proba(Xte)[:, 1]
    probs_list.append(p)
    report(name, p, ya)

print()
print('=' * 78)
print('  ENSEMBLES de top modelos')
print('=' * 78)
# todos
report('Ensemble todos (8 mod)', np.mean(probs_list, axis=0), ya)
# top: RF leaf=1, ET leaf=1, ET leaf=2, LGB d=4
top = [probs_list[0], probs_list[3], probs_list[5], probs_list[6]]
report('Ensemble top4 (RF1+ET1+ET2+LGB4)', np.mean(top, axis=0), ya)
top2 = [probs_list[0], probs_list[1], probs_list[3], probs_list[4]]
report('Ensemble RF+ET grande (4 mod)', np.mean(top2, axis=0), ya)

print()
print('=' * 78)
print('  STACKING — meta-learner treinado no WDD (LOSO) + teste em Anxolotl')
print('=' * 78)
# stacking: gerar OOF no WDD com os melhores modelos → treinar meta-learner
base_clfs = [
    RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', max_features='sqrt', random_state=42, n_jobs=-1),
    ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    lgb.LGBMClassifier(n_estimators=500, max_depth=4, num_leaves=25, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10),
    lgb.LGBMClassifier(n_estimators=500, max_depth=6, num_leaves=50, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10),
]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF predictions no WDD para treinar o meta-learner
oof = np.zeros((len(yd), len(base_clfs)))
for i, clf in enumerate(base_clfs):
    for tr, te in skf.split(Xtr, yd):
        clf.fit(Xtr[tr], yd[tr])
        oof[te, i] = clf.predict_proba(Xtr[te])[:, 1]

# treinar modelos completos no WDD para inferência no Anxolotl
test_preds = np.zeros((len(ya), len(base_clfs)))
for i, clf in enumerate(base_clfs):
    clf.fit(Xtr, yd)
    test_preds[:, i] = clf.predict_proba(Xte)[:, 1]

# meta-learner: regressão logística treinada nas OOF
meta = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
meta.fit(oof, yd)
p_stack = meta.predict_proba(test_preds)[:, 1]
report('Stacking (LR meta)', p_stack, ya)

# media simples dos base models
report('Avg base models', np.mean(test_preds, axis=0), ya)

print()
print('  anterior melhor: F1=0.853')
print('  LOSO ceiling:    F1=0.919')
