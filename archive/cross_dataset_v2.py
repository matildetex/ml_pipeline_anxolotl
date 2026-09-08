import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

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
    print(f'  {name:<48} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t={t:.3f}')
    return f1d, auc

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

imp_all  = SimpleImputer(strategy='mean')
imp_wdd  = SimpleImputer(strategy='mean')

# Datasets preparados
X_wdd_only = imp_wdd.fit_transform(Xd_bl)
X_anx_wdd  = imp_wdd.transform(Xa_bl)

X_all = np.vstack([Xd_bl, Xw_bl])
y_all = np.concatenate([yd, yw])
X_all_i = imp_all.fit_transform(X_all)
X_anx_all = imp_all.transform(Xa_bl)

# WESAD subsampled to match WDD size (balance treino)
rng = np.random.default_rng(42)
w_idx = rng.choice(len(yw), size=len(yd), replace=False)
X_bal = np.vstack([Xd_bl, Xw_bl[w_idx]])
y_bal = np.concatenate([yd, yw[w_idx]])
imp_bal = SimpleImputer(strategy='mean')
X_bal_i = imp_bal.fit_transform(X_bal)
X_anx_bal = imp_bal.transform(Xa_bl)

print('=' * 75)
print('  1. WDD APENAS (60 feat) — fonte mais parecida com Anxolotl')
print('=' * 75)
et_w  = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_wdd_only, yd)
rf_w  = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_wdd_only, yd)
lg1_w = lgb.LGBMClassifier(n_estimators=500, max_depth=5, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1).fit(X_wdd_only, yd)
lg2_w = lgb.LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=15, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=20).fit(X_wdd_only, yd)
xg_w  = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=sum(yd==0)/sum(yd==1), random_state=42, verbosity=0, eval_metric='logloss').fit(X_wdd_only, yd)

p_et_w  = et_w.predict_proba(X_anx_wdd)[:, 1]
p_rf_w  = rf_w.predict_proba(X_anx_wdd)[:, 1]
p_lg1_w = lg1_w.predict_proba(X_anx_wdd)[:, 1]
p_lg2_w = lg2_w.predict_proba(X_anx_wdd)[:, 1]
p_xg_w  = xg_w.predict_proba(X_anx_wdd)[:, 1]

report('ET', p_et_w, ya)
report('RF', p_rf_w, ya)
report('LG1', p_lg1_w, ya)
report('LG2', p_lg2_w, ya)
report('XGBoost', p_xg_w, ya)
print()
report('Ensemble 4 (ET+RF+LG1+LG2)', np.mean([p_et_w, p_rf_w, p_lg1_w, p_lg2_w], axis=0), ya)
report('Ensemble 5 (+XGB)', np.mean([p_et_w, p_rf_w, p_lg1_w, p_lg2_w, p_xg_w], axis=0), ya)

print()
print('=' * 75)
print('  2. WESAD+WDD balanceado (subsample WESAD para tamanho WDD)')
print('=' * 75)
et_b  = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_bal_i, y_bal)
rf_b  = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1).fit(X_bal_i, y_bal)
lg1_b = lgb.LGBMClassifier(n_estimators=500, max_depth=5, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1).fit(X_bal_i, y_bal)
lg2_b = lgb.LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=15, class_weight='balanced', random_state=42, verbose=-1, min_child_samples=20).fit(X_bal_i, y_bal)
xg_b  = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=sum(y_bal==0)/sum(y_bal==1), random_state=42, verbosity=0, eval_metric='logloss').fit(X_bal_i, y_bal)

p_et_b  = et_b.predict_proba(X_anx_bal)[:, 1]
p_rf_b  = rf_b.predict_proba(X_anx_bal)[:, 1]
p_lg1_b = lg1_b.predict_proba(X_anx_bal)[:, 1]
p_lg2_b = lg2_b.predict_proba(X_anx_bal)[:, 1]
p_xg_b  = xg_b.predict_proba(X_anx_bal)[:, 1]

report('ET', p_et_b, ya)
report('RF', p_rf_b, ya)
report('LG1', p_lg1_b, ya)
report('LG2', p_lg2_b, ya)
report('XGBoost', p_xg_b, ya)
print()
report('Ensemble 4 (ET+RF+LG1+LG2)', np.mean([p_et_b, p_rf_b, p_lg1_b, p_lg2_b], axis=0), ya)
report('Ensemble 5 (+XGB)', np.mean([p_et_b, p_rf_b, p_lg1_b, p_lg2_b, p_xg_b], axis=0), ya)

print()
print('=' * 75)
print('  3. MEGA-ENSEMBLE: WDD-only + WESAD+WDD balanceado')
print('=' * 75)
all_probs = [p_et_w, p_rf_w, p_lg1_w, p_lg2_w, p_xg_w,
             p_et_b, p_rf_b, p_lg1_b, p_lg2_b, p_xg_b]
report('Mega-ensemble 10 modelos', np.mean(all_probs, axis=0), ya)
report('WDD 5 + bal 5 (avg)', np.mean(all_probs, axis=0), ya)

# Weighted: dar mais peso ao WDD (mais parecido com Anxolotl)
w_wdd = 0.7; w_bal = 0.3
p_wdd_avg = np.mean([p_et_w, p_rf_w, p_lg1_w, p_lg2_w, p_xg_w], axis=0)
p_bal_avg = np.mean([p_et_b, p_rf_b, p_lg1_b, p_lg2_b, p_xg_b], axis=0)
report('Weighted (WDD×0.7 + bal×0.3)', w_wdd * p_wdd_avg + w_bal * p_bal_avg, ya)
report('Weighted (WDD×0.8 + bal×0.2)', 0.8 * p_wdd_avg + 0.2 * p_bal_avg, ya)
report('WDD ensemble apenas (5 mod)', p_wdd_avg, ya)

print()
print('  anterior melhor: F1=0.853  (Ensemble 4, WESAD+WDD completo)')
print('  LOSO ceiling:    F1=0.919  AUC=0.973')
