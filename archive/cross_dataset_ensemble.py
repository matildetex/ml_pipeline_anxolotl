import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
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
    print(f'  {name:<45} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t={t:.3f}')

with open('wdd/processed_data.pkl', 'rb') as f:
    wdd = pickle.load(f)
with open('wesad/processed_data.pkl', 'rb') as f:
    w = pickle.load(f)
with open('output/processed_data_anxolotl.pkl', 'rb') as f:
    a = pickle.load(f)

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

X_train = np.vstack([Xd_bl, Xw_bl])
y_train = np.concatenate([yd, yw])

imp = SimpleImputer(strategy='mean')
Xtr = imp.fit_transform(X_train)
Xte = imp.transform(Xa_bl)

# 7 features com correlacao consistente entre WDD e WESAD
CONSISTENT = ['acc_wrist_kurtosis', 'acc_wrist_skew', 'hr_skew',
              'ibi_mean', 'ibi_p25', 'ibi_p75', 'rmssd_delta_mean']
sel = [COLS60.index(c) for c in CONSISTENT]
Xtr_s = Xtr[:, sel]
Xte_s = Xte[:, sel]

print('Treino: WESAD + WDD')
print(f'  60 features: {Xtr.shape}  |  7 consistentes: {Xtr_s.shape}')

et  = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr, y_train)
rf  = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr, y_train)
lg1 = lgb.LGBMClassifier(n_estimators=500, max_depth=5, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1).fit(Xtr, y_train)
lg2 = lgb.LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=15, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=30).fit(Xtr, y_train)
et_s = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr_s, y_train)
lg_s = lgb.LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=15, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=30).fit(Xtr_s, y_train)

p_et  = et.predict_proba(Xte)[:, 1]
p_rf  = rf.predict_proba(Xte)[:, 1]
p_lg1 = lg1.predict_proba(Xte)[:, 1]
p_lg2 = lg2.predict_proba(Xte)[:, 1]
p_ets = et_s.predict_proba(Xte_s)[:, 1]
p_lgs = lg_s.predict_proba(Xte_s)[:, 1]

print()
print('=' * 72)
report('ET (60 feat)', p_et, ya)
report('RF (60 feat)', p_rf, ya)
report('LG1 (60 feat)', p_lg1, ya)
report('LG2 (60 feat)', p_lg2, ya)
report('ET (7 feat consistentes)', p_ets, ya)
report('LG (7 feat consistentes)', p_lgs, ya)
print()
report('Ensemble 4 (ET+RF+LG1+LG2)', np.mean([p_et, p_rf, p_lg1, p_lg2], axis=0), ya)
report('Ensemble 6 (all)', np.mean([p_et, p_rf, p_lg1, p_lg2, p_ets, p_lgs], axis=0), ya)
report('Ensemble 60+7 (ET+LG2+ETs+LGs)', np.mean([p_et, p_lg2, p_ets, p_lgs], axis=0), ya)
print()
print('  LOSO ceiling: F1=0.919  AUC=0.973')
