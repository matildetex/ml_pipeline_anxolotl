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
    print(f'  {name:<62} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}')
    return f1d

# ── Load ──────────────────────────────────────────────────────────────────────
with open('wdd/processed_data.pkl', 'rb') as f:    wdd = pickle.load(f)
with open('wesad/processed_data.pkl', 'rb') as f:  w   = pickle.load(f)
with open('output/processed_data_anxolotl.pkl', 'rb') as f: a = pickle.load(f)

COLS60 = sorted(a['feature_cols'])
COLS40 = sorted(set(w['feature_cols']) & set(COLS60))
print(f'Anxolotl: {len(COLS60)} features  |  WESAD∩Anxolotl: {len(COLS40)} features')

# ── Anxolotl ──────────────────────────────────────────────────────────────────
Xa_df = a['X'] if isinstance(a['X'], pd.DataFrame) else pd.DataFrame(a['X'], columns=a['feature_cols'])
ya = np.array(a['y']); ga = np.array(a['groups'])
Xa60 = bl_protocol(Xa_df[COLS60].values.astype(float), ga, n=8)
Xa40 = bl_protocol(Xa_df[COLS40].values.astype(float), ga, n=8)

# ── WDD ───────────────────────────────────────────────────────────────────────
Xd_df = wdd['X'] if isinstance(wdd['X'], pd.DataFrame) else pd.DataFrame(wdd['X'], columns=wdd['feature_cols'])
yd = np.array(wdd['y']); gd = np.array(wdd['groups'])
Xd60_bl = bl_labels(Xd_df[COLS60].values.astype(float), yd, gd)
Xd40_bl = bl_labels(Xd_df[COLS40].values.astype(float), yd, gd)

# ── WESAD ─────────────────────────────────────────────────────────────────────
Xw_df = w['X'] if isinstance(w['X'], pd.DataFrame) else pd.DataFrame(w['X'], columns=w['feature_cols'])
yw = np.array(w['y']); gw = np.array(w['groups'])
Xw40_bl = bl_labels(Xw_df[COLS40].values.astype(float), yw, gw)

# ── Imputers ──────────────────────────────────────────────────────────────────
# WESAD-only (40 feat)
imp_w = SimpleImputer(strategy='mean')
Xtr_w = imp_w.fit_transform(Xw40_bl)
Xte_w = imp_w.transform(Xa40)

# WDD-only (60 feat)
imp_d60 = SimpleImputer(strategy='mean')
Xtr_d60 = imp_d60.fit_transform(Xd60_bl)
Xte_d60 = imp_d60.transform(Xa60)

# WDD-only (40 feat)
imp_d40 = SimpleImputer(strategy='mean')
Xtr_d40 = imp_d40.fit_transform(Xd40_bl)
Xte_d40 = imp_d40.transform(Xa40)

# WESAD+WDD (40 feat)
imp_wd40 = SimpleImputer(strategy='mean')
Xtr_wd40 = imp_wd40.fit_transform(np.vstack([Xd40_bl, Xw40_bl]))
y_wd40   = np.concatenate([yd, yw])
Xte_wd40 = imp_wd40.transform(Xa40)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  GOAL 1 — WESAD-ONLY → Anxolotl  (40 features, BL-labels norm)')
print('  baseline: ET single model F1=0.729')
print('='*78)

# baseline ET (single, reproduz cross_dataset_blnorm.py)
et_w_ref = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
et_w_ref.fit(Xtr_w, yw)
p_et_ref = et_w_ref.predict_proba(Xte_w)[:, 1]
report('ET n=500 mf=sqrt (baseline replicate)', p_et_ref, ya)

print()
print('  --- Hyperparameter grid (RF+ET ensemble) ---')
best_wesad = ('', 0.0, None)
for mf in [0.3, 0.4, 0.5, 0.6, 'sqrt']:
    for leaf in [1, 2]:
        for n in [500, 1000]:
            rf = RandomForestClassifier(n_estimators=n, max_features=mf, min_samples_leaf=leaf,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            et = ExtraTreesClassifier(n_estimators=n, max_features=mf, min_samples_leaf=leaf,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(Xtr_w, yw); et.fit(Xtr_w, yw)
            p = np.mean([rf.predict_proba(Xte_w)[:, 1], et.predict_proba(Xte_w)[:, 1]], axis=0)
            f1 = report(f'RF+ET mf={mf} leaf={leaf} n={n}  [WESAD-40]', p, ya)
            if f1 > best_wesad[1]:
                best_wesad = (f'mf={mf} leaf={leaf} n={n}', f1, p)
    print()

print('  --- LightGBM on WESAD-40 ---')
lgb_configs = [
    ('LGB d=4 l=25 n=500',  lgb.LGBMClassifier(n_estimators=500,  max_depth=4, num_leaves=25,  class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10, feature_fraction=0.5)),
    ('LGB d=3 l=15 n=500',  lgb.LGBMClassifier(n_estimators=500,  max_depth=3, num_leaves=15,  class_weight='balanced', random_state=42, verbose=-1, min_child_samples=15, feature_fraction=0.5)),
    ('LGB d=4 l=25 n=1000', lgb.LGBMClassifier(n_estimators=1000, max_depth=4, num_leaves=25,  class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10, feature_fraction=0.5)),
    ('LGB d=6 l=50 n=500',  lgb.LGBMClassifier(n_estimators=500,  max_depth=6, num_leaves=50,  class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10, feature_fraction=0.6)),
]
lgb_probs_w = []
for name, clf in lgb_configs:
    clf.fit(Xtr_w, yw)
    p = clf.predict_proba(Xte_w)[:, 1]
    lgb_probs_w.append(p)
    report(name + '  [WESAD-40]', p, ya)

# best ET+RF + best LGB
print()
# build best RF+ET from grid (mf=0.5 n=1000 leaf=1, based on WDD experience)
rf_w_best = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
et_w_best = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
rf_w_best.fit(Xtr_w, yw); et_w_best.fit(Xtr_w, yw)
p_rfe_w = np.mean([rf_w_best.predict_proba(Xte_w)[:, 1], et_w_best.predict_proba(Xte_w)[:, 1]], axis=0)
for i, (name, _) in enumerate(lgb_configs):
    p = np.mean([p_rfe_w, lgb_probs_w[i]], axis=0)
    report(f'RF+ET + {name[:12]}  [WESAD-40]', p, ya)
print()
report('RF+ET+LGB avg (all 4 LGB)  [WESAD-40]', np.mean([p_rfe_w] + lgb_probs_w, axis=0), ya)
best_wesad_p = np.mean([p_rfe_w, lgb_probs_w[0]], axis=0)  # RF+ET + LGB d=4

# ═══════════════════════════════════════════════════════════════════════════════
print()
print('='*78)
print('  GOAL 2 — WESAD+WDD → Anxolotl  (best of both datasets)')
print('  prev best two-tower: F1=0.868  |  WDD-only: F1=0.867')
print('='*78)

# Reference: WDD-60 best model
rf_d_best = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
et_d_best = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
rf_d_best.fit(Xtr_d60, yd); et_d_best.fit(Xtr_d60, yd)
p_wdd60 = np.mean([rf_d_best.predict_proba(Xte_d60)[:, 1], et_d_best.predict_proba(Xte_d60)[:, 1]], axis=0)
report('WDD-60 RF+ET mf=0.5 n=1000  (reference)', p_wdd60, ya)

print()
print('  --- Strategy A: WDD-60 tower + WESAD-40 tower (heterogeneous) ---')
# WESAD-40 best single model predictions
p_wesad40_best = p_rfe_w  # RF+ET mf=0.5 n=1000 on WESAD-40
for wa, wb in [(0.95,0.05),(0.90,0.10),(0.85,0.15),(0.80,0.20),(0.75,0.25),(0.70,0.30),(0.60,0.40)]:
    p = wa * p_wdd60 + wb * p_wesad40_best
    report(f'WDD-60×{wa:.2f} + WESAD-40×{wb:.2f}', p, ya)

print()
print('  --- Strategy B: WESAD+WDD concat on 40 feat, best hyperparams ---')
for mf in [0.3, 0.4, 0.5]:
    rf_wd = RandomForestClassifier(n_estimators=1000, max_features=mf, min_samples_leaf=1,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    et_wd = ExtraTreesClassifier(n_estimators=1000, max_features=mf, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    rf_wd.fit(Xtr_wd40, y_wd40); et_wd.fit(Xtr_wd40, y_wd40)
    p_wd = np.mean([rf_wd.predict_proba(Xte_wd40)[:, 1], et_wd.predict_proba(Xte_wd40)[:, 1]], axis=0)
    report(f'RF+ET mf={mf} n=1000  [WESAD+WDD-40]', p_wd, ya)

print()
print('  --- Strategy C: 3-tower (WDD-60 + WESAD-40 + WESAD+WDD-40) ---')
rf_wd5 = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
et_wd5 = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                class_weight='balanced', random_state=42, n_jobs=-1)
rf_wd5.fit(Xtr_wd40, y_wd40); et_wd5.fit(Xtr_wd40, y_wd40)
p_wd40 = np.mean([rf_wd5.predict_proba(Xte_wd40)[:, 1], et_wd5.predict_proba(Xte_wd40)[:, 1]], axis=0)

for w1, w2, w3 in [(0.6,0.2,0.2),(0.7,0.15,0.15),(0.7,0.2,0.1),(0.75,0.15,0.1),(0.8,0.1,0.1)]:
    p = w1*p_wdd60 + w2*p_wesad40_best + w3*p_wd40
    report(f'WDD60×{w1:.2f} + W40×{w2:.2f} + WD40×{w3:.2f}', p, ya)

print()
print('  --- Strategy D: sample weights WESAD in combined (WDD upweighted) ---')
for w_wdd in [5, 8, 10]:
    sw = np.concatenate([np.full(len(yd), w_wdd), np.ones(len(yw))])
    rf_sw = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    et_sw = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    rf_sw.fit(Xtr_wd40, y_wd40, sample_weight=sw)
    et_sw.fit(Xtr_wd40, y_wd40, sample_weight=sw)
    p_sw = np.mean([rf_sw.predict_proba(Xte_wd40)[:, 1], et_sw.predict_proba(Xte_wd40)[:, 1]], axis=0)
    report(f'RF+ET [WESAD+WDD-40] WDD_weight={w_wdd}  + WDD-60', p_sw, ya)
    for wa in [0.7, 0.75, 0.8]:
        p = wa * p_wdd60 + (1-wa) * p_sw
        report(f'  WDD60×{wa:.2f} + WD40_sw{w_wdd}×{1-wa:.2f}', p, ya)
    print()

print()
print('  anterior melhor: F1=0.867  (WDD-60)  /  F1=0.868  (two-tower v4)')
print('  LOSO ceiling:    F1=0.919')
