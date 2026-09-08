"""
Reproduces RECONCILIATION.md SS E3 (WDD->Anxolotl, VARIANT B) with per-subject
granularity, to test whether the Ensemble RF+ET macro-F1 advantage over ET-alone
(0.8702 vs 0.8455, both already on disk in results_master.csv) is statistically
significant via a paired Wilcoxon signed-rank test across the 37 Anxolotl subjects.

Methodology matches E3 exactly:
  - Source (WDD): RandomForest/ExtraTrees(n_estimators=1000, max_features=0.5,
    min_samples_leaf=1, class_weight='balanced', random_state=42), trained once
    on all of WDD (55 canonical columns, source-side bl_labels normalisation,
    using WDD's own true labels -- see archive/cross_dataset_v4.py:10-19, the
    pre-leak-fix predecessor this script's methodology was lifted from).
  - Target (Anxolotl): canonical 686-window/55-feature/37-subject set, leak-fixed
    baseline_relative_scale (demographics_analysis.py:157-190) applied once
    (no LOSO fold on the Anxolotl side -- Anxolotl is pure held-out target,
    never used for training).
  - Ensemble = simple mean of RF/ET predict_proba (soft-vote average), exactly
    as in archive/cross_dataset_v4.py:100 (p_base = np.mean([p_rf_b, p_et_b],
    axis=0)). This is a 2-model ensemble (RF+ET), NOT a 3-model RF+ET+HGB one.

RELOCATED 2026-09-07: moved from 3_anxolotl/ to 2_cross_dataset/ -- this is
Ch. 4.4 (Cross-Dataset)'s canonical source, not Ch. 4.5's (Anxolotl-only).
It still reuses Anxolotl's OWN preprocessing/BL-norm code from 3_anxolotl/
(anxolotl_preprocessing.py, demographics_analysis.py) rather than duplicating
either module here -- see the explicit sys.path addition below.

Run from the repo root (not from inside 2_cross_dataset/): demographics_
analysis.py creates ./results and ./figures relative to the CURRENT WORKING
DIRECTORY at import time (not relative to its own file location) -- running
from repo root correctly no-ops against the existing top-level results/ and
figures/; running from inside 2_cross_dataset/ creates stray empty copies
there instead (harmless, but pointless clutter -- clean them up if you see
them).
"""
import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3_anxolotl'))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, roc_curve
from scipy.stats import wilcoxon

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale

ROOT = os.path.join(os.path.dirname(__file__), '..')

# ── Source: WDD ──────────────────────────────────────────────────────────────
with open(os.path.join(ROOT, 'processed_data', 'wdd', 'processed_data.pkl'), 'rb') as f:
    wdd = pickle.load(f)

# ── Target: Anxolotl canonical (686 windows, 55 feat, 37 subjects) ───────────
CACHE_PATH = os.path.join(ROOT, 'results', 'df_all_1hz_cache.pkl')
with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)
df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=60, stride_sec=30)
df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
print(f'Anxolotl target: {len(df_feat)} windows, {df_feat["subject"].nunique()} subjects, '
      f'{len(feat_cols)} features')

assert set(feat_cols) <= set(wdd['feature_cols']), "column mismatch vs WDD schema"

X_wdd = wdd['X'][feat_cols].values.astype(float)
y_wdd = np.array(wdd['y'])
groups_wdd = np.array(wdd['groups'])

X_anx = df_feat[feat_cols].values.astype(float)
y_anx = df_feat['label'].values.astype(int)
groups_anx = df_feat['subject'].values
subjects = sorted(np.unique(groups_anx), key=lambda s: int(s[1:]))
print(f'Subjects: {len(subjects)}')


def bl_labels(X, y, groups):
    """Source-side normalisation using WDD's own true labels (archive/cross_dataset_v4.py:10-19)."""
    X = np.array(X, dtype=float).copy(); y = np.array(y); groups = np.array(groups)
    for s in np.unique(groups):
        sm = groups == s; bm = sm & (y == 0)
        if bm.sum() > 1:
            for j in range(X.shape[1]):
                v = X[bm, j][~np.isnan(X[bm, j])]
                if len(v) > 1:
                    X[sm, j] = (X[sm, j] - v.mean()) / (v.std() + 1e-8)
    return X


# ── Normalise ──────────────────────────────────────────────────────────────
X_wdd_n = bl_labels(X_wdd, y_wdd, groups_wdd)
X_anx_n = baseline_relative_scale(X_anx, y_anx, groups_anx)

imp = SimpleImputer(strategy='median')
X_wdd_n = imp.fit_transform(X_wdd_n)
X_anx_n = imp.transform(np.nan_to_num(X_anx_n, nan=np.nan))

# ── Train once on all of WDD (source), predict on Anxolotl (target) ─────────
PARAMS = dict(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
              class_weight='balanced', random_state=42, n_jobs=-1)
rf = RandomForestClassifier(**PARAMS).fit(X_wdd_n, y_wdd)
et = ExtraTreesClassifier(**PARAMS).fit(X_wdd_n, y_wdd)

p_rf = rf.predict_proba(X_anx_n)[:, 1]
p_et = et.predict_proba(X_anx_n)[:, 1]
p_ens = np.mean([p_rf, p_et], axis=0)

pred_rf  = (p_rf  >= 0.5).astype(int)
pred_et  = (p_et  >= 0.5).astype(int)
pred_ens = (p_ens >= 0.5).astype(int)

# ── Aggregate sanity check against results_master.csv / RECONCILIATION.md SS E3 ─
f1_rf_agg  = f1_score(y_anx, pred_rf,  average='macro')
f1_et_agg  = f1_score(y_anx, pred_et,  average='macro')
f1_ens_agg = f1_score(y_anx, pred_ens, average='macro')
auc_rf  = roc_auc_score(y_anx, p_rf)
auc_et  = roc_auc_score(y_anx, p_et)
auc_ens = roc_auc_score(y_anx, p_ens)

print('\n=== Aggregate (sanity check vs documented 0.8659 / 0.8455 / 0.8702) ===')
print(f'RF        MacroF1={f1_rf_agg:.4f}  AUC={auc_rf:.4f}   (documented: 0.8659 / 0.9168)')
print(f'ET        MacroF1={f1_et_agg:.4f}  AUC={auc_et:.4f}   (documented: 0.8455 / 0.9244)')
print(f'Ensemble  MacroF1={f1_ens_agg:.4f}  AUC={auc_ens:.4f}   (documented: 0.8702 / 0.9281)')

# ── Per-subject F1 (this is what results_master.csv/RECONCILIATION.md do NOT have) ─
rows = []
for s in subjects:
    m = groups_anx == s
    if m.sum() == 0:
        continue
    f1_et_s  = f1_score(y_anx[m], pred_et[m],  average='macro', zero_division=0)
    f1_ens_s = f1_score(y_anx[m], pred_ens[m], average='macro', zero_division=0)
    f1_rf_s  = f1_score(y_anx[m], pred_rf[m],  average='macro', zero_division=0)
    rows.append((s, int(m.sum()), f1_rf_s, f1_et_s, f1_ens_s))

print(f'\n{"subject":8s}{"n_win":>6s}{"RF":>8s}{"ET":>8s}{"Ensemble":>10s}{"Ens-ET":>9s}')
for s, n, rf_s, et_s, ens_s in rows:
    print(f'{s:8s}{n:6d}{rf_s:8.3f}{et_s:8.3f}{ens_s:10.3f}{ens_s-et_s:+9.3f}')

et_arr  = np.array([r[3] for r in rows])
ens_arr = np.array([r[4] for r in rows])
diffs = ens_arr - et_arr
print(f'\nMean per-subject F1:  ET={et_arr.mean():.4f}  Ensemble={ens_arr.mean():.4f}  '
      f'diff={diffs.mean():+.4f}')
print(f'Subjects where Ensemble > ET: {(diffs>1e-9).sum()}  '
      f'= ET: {(np.abs(diffs)<=1e-9).sum()}  < ET: {(diffs<-1e-9).sum()}')

stat, p = wilcoxon(ens_arr, et_arr)
print(f'\nWilcoxon signed-rank (Ensemble vs ET, n={len(rows)} paired subjects): '
      f'statistic={stat:.2f}  p={p:.4f}  '
      f'({"SIGNIFICANT" if p < 0.05 else "NOT significant"} at alpha=0.05)')

# ── Calibration: Brier score, 10-bin ECE, F1@opt (Youden's J) vs F1@default ─
def ece_10bin(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return ece


def f1_at_opt(y_true, p):
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    t_opt = thr[np.argmax(j)]
    pred = (p >= t_opt).astype(int)
    return f1_score(y_true, pred, average='macro', zero_division=0), t_opt


print('\n=== Calibration (Brier score, 10-bin ECE) + F1@opt vs F1@default gap ===')
for name, p, pred, f1_def in [('RF', p_rf, pred_rf, f1_rf_agg),
                               ('ET', p_et, pred_et, f1_et_agg),
                               ('Ensemble', p_ens, pred_ens, f1_ens_agg)]:
    brier = brier_score_loss(y_anx, p)
    ece = ece_10bin(y_anx, p)
    f1opt, topt = f1_at_opt(y_anx, p)
    gap = f1opt - f1_def
    print(f'{name:10s} Brier={brier:.4f}  ECE={ece:.4f}  '
          f'F1@0.5={f1_def:.4f}  F1@opt(t={topt:.3f})={f1opt:.4f}  gap={gap:+.4f}')

OUT = os.path.join(ROOT, 'results', 'wdd_anxolotl_et_vs_ensemble_per_subject.csv')
with open(OUT, 'w') as f:
    f.write('subject,n_windows,rf_f1,et_f1,ensemble_f1,ensemble_minus_et\n')
    for s, n, rf_s, et_s, ens_s in rows:
        f.write(f'{s},{n},{rf_s:.4f},{et_s:.4f},{ens_s:.4f},{ens_s-et_s:.4f}\n')
print(f'\nSaved: {os.path.relpath(OUT, ROOT)}')
