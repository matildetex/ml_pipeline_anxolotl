import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

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

def report(name, probs, ya, threshold=0.5):
    f1d = f1_score(ya, (probs >= threshold).astype(int), average='macro', zero_division=0)
    auc = roc_auc_score(ya, probs)
    fpr, tpr, thr = roc_curve(ya, probs)
    t = float(thr[np.argmax(tpr - fpr)])
    f1o = f1_score(ya, (probs >= t).astype(int), average='macro', zero_division=0)
    print(f'  {name:<62} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t_used={threshold:.3f}')
    return f1d, t

def rank_norm(p):
    """Convert probabilities to uniform [0,1] via rank normalization."""
    r = np.argsort(np.argsort(p))
    return r / (len(r) - 1)

with open('wdd/processed_data.pkl', 'rb') as f:    wdd = pickle.load(f)
with open('wesad/processed_data.pkl', 'rb') as f:  w   = pickle.load(f)
with open('output/processed_data_anxolotl.pkl', 'rb') as f: a = pickle.load(f)

COLS60 = sorted(a['feature_cols'])

Xa_df = a['X'] if isinstance(a['X'], pd.DataFrame) else pd.DataFrame(a['X'], columns=a['feature_cols'])
ya = np.array(a['y']); ga = np.array(a['groups'])
Xa60 = bl_protocol(Xa_df[COLS60].values.astype(float), ga, n=8)

Xd_df = wdd['X'] if isinstance(wdd['X'], pd.DataFrame) else pd.DataFrame(wdd['X'], columns=wdd['feature_cols'])
yd = np.array(wdd['y']); gd = np.array(wdd['groups'])
Xd60_bl = bl_labels(Xd_df[COLS60].values.astype(float), yd, gd)

imp60 = SimpleImputer(strategy='mean')
Xtr = imp60.fit_transform(Xd60_bl)
Xte = imp60.transform(Xa60)

# ═══════════════════════════════════════════════════════════════════════════════
print('='*78)
print('  MULTI-SEED ENSEMBLE: variance reduction across 8 random seeds')
print('='*78)
SEEDS = [0, 7, 42, 99, 123, 456, 789, 1234]
seed_probs = []
for seed in SEEDS:
    rf = RandomForestClassifier(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                                 class_weight='balanced', random_state=seed, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(Xtr, yd); et.fit(Xtr, yd)
    p = np.mean([rf.predict_proba(Xte)[:, 1], et.predict_proba(Xte)[:, 1]], axis=0)
    seed_probs.append(p)
    report(f'RF+ET mf=0.5 seed={seed}', p, ya)

print()
p_multiseed = np.mean(seed_probs, axis=0)
report('Multi-seed avg (8 seeds × RF+ET = 16 models)', p_multiseed, ya)

# rank-normalized version
p_rank_avg = np.mean([rank_norm(p) for p in seed_probs], axis=0)
report('Multi-seed RANK-avg (8 seeds)', p_rank_avg, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print()
print('='*78)
print('  WDD-LOSO THRESHOLD: calibrate threshold on WDD without Anxolotl labels')
print('='*78)
# LOSO on WDD to get OOF predictions from the same model family
logo = LeaveOneGroupOut()
oof_probs = np.zeros(len(yd))
for tr, te in logo.split(Xtr, yd, groups=gd):
    rf_l = RandomForestClassifier(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    et_l = ExtraTreesClassifier(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf_l.fit(Xtr[tr], yd[tr]); et_l.fit(Xtr[tr], yd[tr])
    oof_probs[te] = np.mean([rf_l.predict_proba(Xtr[te])[:, 1],
                              et_l.predict_proba(Xtr[te])[:, 1]], axis=0)

# find optimal threshold from WDD LOSO OOF (no Anxolotl info used)
fpr_w, tpr_w, thr_w = roc_curve(yd, oof_probs)
t_wdd_opt = float(thr_w[np.argmax(tpr_w - fpr_w)])
f1_wdd_oof = f1_score(yd, (oof_probs >= t_wdd_opt).astype(int), average='macro', zero_division=0)
print(f'  WDD LOSO OOF: AUC={roc_auc_score(yd, oof_probs):.3f}  F1@opt={f1_wdd_oof:.3f}  t_opt={t_wdd_opt:.3f}')
print()

# apply WDD-calibrated threshold to Anxolotl predictions (multi-seed ensemble)
print(f'  Applying WDD-optimal threshold t={t_wdd_opt:.3f} to Anxolotl predictions:')
report('Multi-seed avg @ WDD-LOSO threshold', p_multiseed, ya, threshold=t_wdd_opt)
report('Baseline RF+ET42 @ WDD-LOSO threshold', seed_probs[SEEDS.index(42)], ya, threshold=t_wdd_opt)

# also try a range of fixed thresholds near t_wdd_opt
print()
print('  Threshold sweep on Anxolotl (using WDD-LOSO as guide, not Anxolotl labels):')
for t in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]:
    f1 = f1_score(ya, (p_multiseed >= t).astype(int), average='macro', zero_division=0)
    print(f'    t={t:.2f}  F1={f1:.3f}')

# ═══════════════════════════════════════════════════════════════════════════════
print()
print('='*78)
print('  RANK-NORMALIZED ENSEMBLE: rank each model before averaging')
print('='*78)
# train 4-model ensemble (2 seeds × RF+ET) and rank-normalize
rf42 = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=42, n_jobs=-1)
et42 = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                              class_weight='balanced', random_state=42, n_jobs=-1)
rf0  = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=0, n_jobs=-1)
et0  = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                              class_weight='balanced', random_state=0, n_jobs=-1)
for m in [rf42, et42, rf0, et0]: m.fit(Xtr, yd)
p_rf42 = rf42.predict_proba(Xte)[:, 1]
p_et42 = et42.predict_proba(Xte)[:, 1]
p_rf0  = rf0.predict_proba(Xte)[:, 1]
p_et0  = et0.predict_proba(Xte)[:, 1]

raw4 = [p_rf42, p_et42, p_rf0, p_et0]
report('Avg-4 raw probs (2seeds×RF+ET)', np.mean(raw4, axis=0), ya)
report('Avg-4 rank-norm probs', np.mean([rank_norm(p) for p in raw4], axis=0), ya)

# ═══════════════════════════════════════════════════════════════════════════════
print()
print('='*78)
print('  ISOTONIC CALIBRATION: calibrate ET probs on WDD LOSO OOF')
print('='*78)
# train final model on all WDD
rf_final = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
et_final = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
rf_final.fit(Xtr, yd); et_final.fit(Xtr, yd)
p_rf_te = rf_final.predict_proba(Xte)[:, 1]
p_et_te = et_final.predict_proba(Xte)[:, 1]

# OOF for calibration (separate LOSO)
oof_rf = np.zeros(len(yd)); oof_et = np.zeros(len(yd))
for tr, te in logo.split(Xtr, yd, groups=gd):
    rf_c = RandomForestClassifier(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    et_c = ExtraTreesClassifier(n_estimators=300, max_features=0.5, min_samples_leaf=1,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf_c.fit(Xtr[tr], yd[tr]); et_c.fit(Xtr[tr], yd[tr])
    oof_rf[te] = rf_c.predict_proba(Xtr[te])[:, 1]
    oof_et[te] = et_c.predict_proba(Xtr[te])[:, 1]

# isotonic calibration
iso_rf = IsotonicRegression(out_of_bounds='clip').fit(oof_rf, yd)
iso_et = IsotonicRegression(out_of_bounds='clip').fit(oof_et, yd)
p_rf_cal = iso_rf.predict(p_rf_te)
p_et_cal = iso_et.predict(p_et_te)

report('RF calibrated (isotonic, WDD-LOSO)', p_rf_cal, ya)
report('ET calibrated (isotonic, WDD-LOSO)', p_et_cal, ya)
report('Avg RF+ET calibrated', np.mean([p_rf_cal, p_et_cal], axis=0), ya)
# combine calibrated + uncalibrated
report('Avg RF+ET raw+cal (4 models)', np.mean([p_rf_te, p_et_te, p_rf_cal, p_et_cal], axis=0), ya)

print()
print('  anterior melhor: F1=0.867  (RF+ET mf=0.5 n=1000, WDD-60)')
print('  LOSO ceiling:    F1=0.919')
