"""
ARCHIVED 2026-09-07: pre-leak-fix version (712 windows); superseded by the
686-window leak-fixed pipeline -- see
2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py.

This script's normalisation (bl_protocol, target-side) and raw (non-
canonical, non-phase3-excluded) Anxolotl windowing predate the 2026-07-19
BL-norm leak fix. Its own baseline comparison print ("LOSO ceiling: F1=0.919")
is itself a stale hardcoded constant, not a live computation -- see
results_master.csv row for 'Anxolotl,60all-feat-STALE'. Its real outputs
(results/cross_dataset_wdd.json, results_master.csv rows dated 2026-06-08)
are themselves superseded: the '0.868' two-tower number they produced was
briefly (incorrectly) believed to be Table 4.12's source, until the thesis
text was read directly and confirmed to report 686 windows / Macro-F1=0.87,
which traces to the canonical script named above, not this one.
"""
import pickle, numpy as np, pandas as pd, json, os
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve,
                              balanced_accuracy_score, matthews_corrcoef)
import lightgbm as lgb

_RESULTS = []   # accumulates all results for JSON output

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
    print(f'  {name:<56} F1={f1d:.3f}  F1@opt={f1o:.3f}  AUC={auc:.3f}  t={t:.3f}')
    _RESULTS.append({'label': name, 'f1': f1d, 'f1opt': f1o, 'auc': auc,
                     'ba': ba, 'mcc': mcc, 'dataset': 'WDD→Anxolotl'})
    return f1d

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
print(f'60-feat cols (WDD→Anxolotl): {len(COLS60)}')
print(f'40-feat cols (WESAD∩Anxolotl): {len(COLS40)}')

# ── Anxolotl ─────────────────────────────────────────────────────────────────
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
imp60 = SimpleImputer(strategy='mean')
Xtr60 = imp60.fit_transform(Xd60_bl)
Xte60 = imp60.transform(Xa60)

imp40 = SimpleImputer(strategy='mean')
X_both40 = np.vstack([Xd40_bl, Xw40_bl])
y_both40  = np.concatenate([yd, yw])
Xtr40 = imp40.fit_transform(X_both40)
Xte40 = imp40.transform(Xa40)

# sample weights: WDD rows get weight=w_wdd, WESAD rows get weight=1
def make_weights(n_wdd, n_wesad, w_wdd):
    return np.concatenate([np.full(n_wdd, w_wdd), np.ones(n_wesad)])

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  BASELINE: WDD-only 60 feat, RF+ET mf=0.5 n=1000  (prev best F1=0.867)')
print('='*78)
rf_b = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=42, n_jobs=-1)
et_b = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                              class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(Xtr60, yd)
et_b.fit(Xtr60, yd)
p_rf_b = rf_b.predict_proba(Xte60)[:, 1]
p_et_b = et_b.predict_proba(Xte60)[:, 1]
report('RF mf=0.5 n=1000  (WDD-60)', p_rf_b, ya)
report('ET mf=0.5 n=1000  (WDD-60)', p_et_b, ya)
p_base = np.mean([p_rf_b, p_et_b], axis=0)
report('Ensemble RF+ET mf=0.5 n=1000  (WDD-60)', p_base, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  TOWER B: WESAD+WDD 40 feat  (add WESAD signal on shared features)')
print('='*78)
rf_40 = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                class_weight='balanced', random_state=42, n_jobs=-1)
et_40 = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=42, n_jobs=-1)
rf_40.fit(Xtr40, y_both40)
et_40.fit(Xtr40, y_both40)
p_rf_40 = rf_40.predict_proba(Xte40)[:, 1]
p_et_40 = et_40.predict_proba(Xte40)[:, 1]
p_tower_b = np.mean([p_rf_40, p_et_40], axis=0)
report('RF mf=0.5 n=1000  (WESAD+WDD-40)', p_rf_40, ya)
report('ET mf=0.5 n=1000  (WESAD+WDD-40)', p_et_40, ya)
report('Ensemble RF+ET  (WESAD+WDD-40)', p_tower_b, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  TWO-TOWER: WDD-60 + WESAD+WDD-40  (weighted combinations)')
print('='*78)
for wa, wb in [(0.9,0.1),(0.8,0.2),(0.75,0.25),(0.7,0.3),(0.6,0.4),(0.5,0.5)]:
    p = wa * p_base + wb * p_tower_b
    report(f'WDD-60×{wa:.2f} + WESAD+WDD-40×{wb:.2f}', p, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  SAMPLE WEIGHTS: WDD samples upweighted vs WESAD (train on 40 feat)')
print('='*78)
for w_wdd in [3, 5, 8, 10, 15]:
    sw = make_weights(len(yd), len(yw), w_wdd)
    rf_sw = RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    et_sw = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    rf_sw.fit(Xtr40, y_both40, sample_weight=sw)
    et_sw.fit(Xtr40, y_both40, sample_weight=sw)
    p_sw = np.mean([rf_sw.predict_proba(Xte40)[:, 1], et_sw.predict_proba(Xte40)[:, 1]], axis=0)
    report(f'RF+ET WESAD+WDD-40  WDD_weight={w_wdd}', p_sw, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  LARGER ENSEMBLES: n=2000 on WDD-60')
print('='*78)
rf2k = RandomForestClassifier(n_estimators=2000, max_features=0.5, min_samples_leaf=1,
                               class_weight='balanced', random_state=42, n_jobs=-1)
et2k = ExtraTreesClassifier(n_estimators=2000, max_features=0.5, min_samples_leaf=1,
                              class_weight='balanced', random_state=42, n_jobs=-1)
rf2k.fit(Xtr60, yd)
et2k.fit(Xtr60, yd)
p_rf2k = rf2k.predict_proba(Xte60)[:, 1]
p_et2k = et2k.predict_proba(Xte60)[:, 1]
report('RF mf=0.5 n=2000  (WDD-60)', p_rf2k, ya)
report('ET mf=0.5 n=2000  (WDD-60)', p_et2k, ya)
p_2k = np.mean([p_rf2k, p_et2k], axis=0)
report('Ensemble RF+ET mf=0.5 n=2000  (WDD-60)', p_2k, ya)

# combine 2k WDD-60 + WESAD+WDD-40 tower
for wa, wb in [(0.8,0.2),(0.75,0.25),(0.7,0.3)]:
    p = wa * p_2k + wb * p_tower_b
    report(f'WDD-60-2k×{wa:.2f} + WESAD+WDD-40×{wb:.2f}', p, ya)

# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*78)
print('  LGBM: tune on WDD-60 (add to ensemble)')
print('='*78)
lgbm_configs = [
    ('LGB d=4 l=25 n=1000', lgb.LGBMClassifier(n_estimators=1000, max_depth=4, num_leaves=25,
        class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1)),
    ('LGB d=3 l=15 n=1000', lgb.LGBMClassifier(n_estimators=1000, max_depth=3, num_leaves=15,
        class_weight='balanced', random_state=42, verbose=-1, min_child_samples=15,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1)),
    ('LGB d=5 l=31 n=500', lgb.LGBMClassifier(n_estimators=500, max_depth=5, num_leaves=31,
        class_weight='balanced', random_state=42, verbose=-1, min_child_samples=10,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1)),
]
lgbm_probs = []
for name, clf in lgbm_configs:
    clf.fit(Xtr60, yd)
    p = clf.predict_proba(Xte60)[:, 1]
    lgbm_probs.append(p)
    report(name + '  (WDD-60)', p, ya)

print()
# RF+ET best + LGBM combos
for i, (name, _) in enumerate(lgbm_configs):
    p_combo = np.mean([p_rf_b, p_et_b, lgbm_probs[i]], axis=0)
    report(f'RF+ET+{name[:10]}  (WDD-60)', p_combo, ya)

# big ensemble: RF+ET n=1000 + RF+ET n=2000 + top LGBM
p_mega = np.mean([p_rf_b, p_et_b, p_rf2k, p_et2k, lgbm_probs[0]], axis=0)
report('Mega: RF1k+ET1k+RF2k+ET2k+LGB (WDD-60)', p_mega, ya)

print()
print('  anterior melhor: F1=0.867  (RF+ET mf=0.5 n=1000, WDD-60)')
print('  LOSO ceiling:    F1=0.919')

_out = os.path.join(os.path.dirname(__file__), '..', 'results', 'cross_dataset_wdd.json')
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, 'w') as _f:
    json.dump(_RESULTS, _f, indent=2)
print(f'\nResultados guardados em: {_out}')
