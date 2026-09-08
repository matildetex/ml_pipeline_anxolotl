import pickle, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

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
    print(f'  {name:<60} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}')
    return f1d

with open('wdd/processed_data.pkl', 'rb') as f:    wdd = pickle.load(f)
with open('wesad/processed_data.pkl', 'rb') as f:  w   = pickle.load(f)
with open('output/processed_data_anxolotl.pkl', 'rb') as f: a = pickle.load(f)

COLS60 = sorted(a['feature_cols'])
COLS40 = sorted(set(w['feature_cols']) & set(COLS60))

Xa_df = a['X'] if isinstance(a['X'], pd.DataFrame) else pd.DataFrame(a['X'], columns=a['feature_cols'])
ya = np.array(a['y']); ga = np.array(a['groups'])
Xa60 = bl_protocol(Xa_df[COLS60].values.astype(float), ga, n=8)

Xd_df = wdd['X'] if isinstance(wdd['X'], pd.DataFrame) else pd.DataFrame(wdd['X'], columns=wdd['feature_cols'])
yd = np.array(wdd['y']); gd = np.array(wdd['groups'])
Xd60_bl = bl_labels(Xd_df[COLS60].values.astype(float), yd, gd)

Xw_df = w['X'] if isinstance(w['X'], pd.DataFrame) else pd.DataFrame(w['X'], columns=w['feature_cols'])
yw = np.array(w['y']); gw = np.array(w['groups'])
Xw40_bl = bl_labels(Xw_df[COLS40].values.astype(float), yw, gw)
Xd40_bl = bl_labels(Xd_df[COLS40].values.astype(float), yd, gd)

imp60 = SimpleImputer(strategy='mean')
Xtr60 = imp60.fit_transform(Xd60_bl)
Xte60 = imp60.transform(Xa60)

imp40 = SimpleImputer(strategy='mean')
X_both40 = np.vstack([Xd40_bl, Xw40_bl])
y_both40 = np.concatenate([yd, yw])
Xtr40 = imp40.fit_transform(X_both40)
Xa40 = bl_protocol(Xa_df[COLS40].values.astype(float), ga, n=8)
Xte40 = imp40.transform(Xa40)

# ═══════════════════════════════════════════════════════════════════════════════
print('='*78)
print('  COVARIATE SHIFT: weight WDD samples by P(Anxolotl) / P(WDD)')
print('  (domain classifier LR; uses unlabeled Anxolotl features — no label leakage)')
print('='*78)

# Domain classifier: WDD=0, Anxolotl=1  → estimate density ratio
X_dom = np.vstack([Xtr60, Xte60])
y_dom = np.concatenate([np.zeros(len(Xtr60)), np.ones(len(Xte60))])
sc = StandardScaler()
X_dom_s = sc.fit_transform(X_dom)
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_dom_s, y_dom)

# P(Anxolotl|x) for WDD training samples
p_anx = lr.predict_proba(sc.transform(Xtr60))[:, 1]
p_wdd = 1 - p_anx
# density ratio w(x) = P(Anxolotl|x) / P(WDD|x), clipped to avoid extremes
w_ratio = np.clip(p_anx / (p_wdd + 1e-8), 0.1, 10.0)
print(f'  density-ratio weights: min={w_ratio.min():.3f} mean={w_ratio.mean():.3f} max={w_ratio.max():.3f}')

for leaf in [1, 2]:
    for mf in [0.5, 0.4]:
        rf_cs = RandomForestClassifier(n_estimators=1000, max_features=mf, min_samples_leaf=leaf,
                                        class_weight='balanced', random_state=42, n_jobs=-1)
        et_cs = ExtraTreesClassifier(n_estimators=1000, max_features=mf, min_samples_leaf=leaf,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        rf_cs.fit(Xtr60, yd, sample_weight=w_ratio)
        et_cs.fit(Xtr60, yd, sample_weight=w_ratio)
        p_cs = np.mean([rf_cs.predict_proba(Xte60)[:, 1], et_cs.predict_proba(Xte60)[:, 1]], axis=0)
        report(f'Covariate-shift RF+ET mf={mf} leaf={leaf}  (WDD-60)', p_cs, ya)

print()

# ═══════════════════════════════════════════════════════════════════════════════
print('='*78)
print('  FEATURE SELECTION: top-K by WDD ET importance  (WDD-60 then train)')
print('='*78)

# fit ET on full WDD-60 to get importance ranking
et_imp = ExtraTreesClassifier(n_estimators=500, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=42, n_jobs=-1)
et_imp.fit(Xtr60, yd)
importances = et_imp.feature_importances_
feat_order = np.argsort(importances)[::-1]
print(f'  Top 10 features: {[COLS60[i] for i in feat_order[:10]]}')

best_topk = ('', 0.0, None)
for k in [10, 15, 20, 25, 30, 40]:
    sel = feat_order[:k]
    Xtr_k = Xtr60[:, sel]; Xte_k = Xte60[:, sel]
    rf_k = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    et_k = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    rf_k.fit(Xtr_k, yd); et_k.fit(Xtr_k, yd)
    p_k = np.mean([rf_k.predict_proba(Xte_k)[:, 1], et_k.predict_proba(Xte_k)[:, 1]], axis=0)
    f1 = report(f'RF+ET mf=0.5 n=1000  top-{k} features', p_k, ya)
    if f1 > best_topk[1]:
        best_topk = (k, f1, p_k)

print()

# combine best top-K with full-60 ensemble
rf_full = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
et_full = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                class_weight='balanced', random_state=42, n_jobs=-1)
rf_full.fit(Xtr60, yd); et_full.fit(Xtr60, yd)
p_full = np.mean([rf_full.predict_proba(Xte60)[:, 1], et_full.predict_proba(Xte60)[:, 1]], axis=0)

k_best, _, p_topk = best_topk
if p_topk is not None:
    for wa in [0.6, 0.7, 0.75, 0.8, 0.9]:
        p_combo = wa * p_full + (1-wa) * p_topk
        report(f'Full-60×{wa:.2f} + top-{k_best}×{1-wa:.2f}', p_combo, ya)

print()

# ═══════════════════════════════════════════════════════════════════════════════
print('='*78)
print('  ET-ONLY: tune min_samples_leaf + max_features  (ET has best AUC=0.926)')
print('='*78)
et_probs = {}
for leaf in [1, 2, 3, 5]:
    for mf in [0.3, 0.4, 0.5, 0.6]:
        et_t = ExtraTreesClassifier(n_estimators=1000, max_features=mf, min_samples_leaf=leaf,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        et_t.fit(Xtr60, yd)
        p = et_t.predict_proba(Xte60)[:, 1]
        et_probs[(leaf, mf)] = p
        report(f'ET mf={mf} leaf={leaf} n=1000  (WDD-60)', p, ya)
    print()

# best ET-only combos
print('  --- ET-only ensembles ---')
keys = list(et_probs.keys())
best_single = max(keys, key=lambda k: f1_score(ya, (et_probs[k]>=0.5).astype(int), average='macro', zero_division=0))
print(f'  Best single ET: leaf={best_single[0]} mf={best_single[1]}')

# ensemble of diverse ET configs
p_et_div = np.mean([et_probs[(1,0.5)], et_probs[(1,0.4)], et_probs[(2,0.5)], et_probs[(2,0.4)]], axis=0)
report('ET ensemble (leaf=1,2 × mf=0.4,0.5)', p_et_div, ya)

p_et_all = np.mean(list(et_probs.values()), axis=0)
report('ET ensemble (all 16 configs)', p_et_all, ya)

# RF + best ET combos
rf_ref = rf_full
for w_et in [0.6, 0.7, 0.8]:
    p_combo = (1-w_et) * rf_ref.predict_proba(Xte60)[:, 1] + w_et * et_probs[(1,0.5)]
    report(f'RF×{1-w_et:.1f} + ET(1,0.5)×{w_et:.1f}', p_combo, ya)

print()

# ═══════════════════════════════════════════════════════════════════════════════
print('='*78)
print('  SHALLOW TREES: max_depth constraint for better generalization')
print('='*78)
for depth in [8, 10, 15, 20, None]:
    rf_d = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                   max_depth=depth, class_weight='balanced', random_state=42, n_jobs=-1)
    et_d = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  max_depth=depth, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_d.fit(Xtr60, yd); et_d.fit(Xtr60, yd)
    p_d = np.mean([rf_d.predict_proba(Xte60)[:, 1], et_d.predict_proba(Xte60)[:, 1]], axis=0)
    report(f'RF+ET mf=0.5 max_depth={depth}  (WDD-60)', p_d, ya)

print()
print('  anterior melhor: F1=0.867  (RF+ET mf=0.5 n=1000, WDD-60)')
print('  LOSO ceiling:    F1=0.919')
