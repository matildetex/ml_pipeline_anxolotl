"""
HRV-channel ablation -- canonical Anxolotl ET-1000 configuration.

Motivation: in the Anxolotl data-collection app, ibi_ms is not a real
beat-to-beat interval -- it is algebraically derived from the aggregated
HR reading (ibi_ms = 60000 / hr_bpm, confirmed empirically on raw session
CSVs, correlation ~0.99999998). rmssd and sdnn are in turn computed from
that same ibi_ms series (anxolotl_preprocessing.py:preprocess_hr_ibi).
So 33 of the canonical 55 features (ibi_*, rmssd_*, sdnn_*) are, in
principle, redundant with the 11 hr_* features. This script tests that
empirically: does dropping ibi/rmssd/sdnn cost the canonical model any
LOSO performance?

Reuses, unmodified:
  - anxolotl_preprocessing.build_features_tabular  (via _canonical_sensitivity_common)
  - demographics_analysis.baseline_relative_scale  (leak-fixed BL-norm, 2026-07-19)
  - _canonical_sensitivity_common.et1000_factory   (ExtraTrees, n_estimators=1000,
    max_features=0.5, class_weight=balanced, random_state=42, n_jobs=-1)
  - _canonical_sensitivity_common.build_canonical_xy (60s/30s window/stride,
    boundary-fixed labelling, phase3 excluded -> 686 windows, 37 subjects, 55 feat)

The 55-feature matrix is extracted ONCE via build_canonical_xy(). Every
subset below is a column selection on that same matrix -- no re-extraction,
no change to windowing/features/CV. LOSO-CV (Leave-One-Subject-Out) is run
independently per subset because BL-norm (baseline_relative_scale) and the
median imputer must be re-fit per fold per subset, exactly as every other
canonical sensitivity script in 3_anxolotl/ already does.

Writes only inside 3_anxolotl/hrv_ablation/. Does not modify or overwrite
anything in 3_anxolotl/ or results/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from scipy.stats import wilcoxon

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, et1000_factory, baseline_relative_scale
)

OUT_DIR = os.path.dirname(__file__)
OUT_CSV = os.path.join(OUT_DIR, 'results.csv')
OUT_PER_FOLD_CSV = os.path.join(OUT_DIR, 'per_subject_f1.csv')
OUT_CORR_CSV = os.path.join(OUT_DIR, 'channel_mean_correlation.csv')

WINDOW_SEC, STRIDE_SEC = 60, 30

# ── 1. Extract the 55-feature matrix ONCE ─────────────────────────────────────

df_all = load_cache()
X55, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)

assert len(feat_cols) == 55, f'expected 55 features, got {len(feat_cols)}'
assert n_after == 686, f'expected 686 windows after phase3 exclusion, got {n_after}'
assert len(subjects) == 37, f'expected 37 subjects, got {len(subjects)}'

print(f'Canonical matrix: {n_after} windows (from {n_before}), {len(subjects)} subjects, '
      f'{len(feat_cols)} features. Extracted once; all subsets below are column selections.')

# ── 2. Define channel subsets by column-name filtering ────────────────────────

def cols_with_prefix(prefix):
    return [c for c in feat_cols if c.startswith(prefix)]

acc_cols   = cols_with_prefix('acc_wrist_')
hr_cols    = cols_with_prefix('hr_')
ibi_cols   = cols_with_prefix('ibi_')
rmssd_cols = cols_with_prefix('rmssd_')
sdnn_cols  = cols_with_prefix('sdnn_')

for name, cols in [('acc_wrist', acc_cols), ('hr', hr_cols), ('ibi', ibi_cols),
                    ('rmssd', rmssd_cols), ('sdnn', sdnn_cols)]:
    assert len(cols) == 11, f'expected 11 {name}_* columns, got {len(cols)}: {cols}'
assert len(acc_cols) + len(hr_cols) + len(ibi_cols) + len(rmssd_cols) + len(sdnn_cols) == 55

SUBSETS = {
    'FULL':      feat_cols,
    'HR+ACC':    hr_cols + acc_cols,
    'ACC_only':  acc_cols,
    'HR_only':   hr_cols,
    'HRV_only':  ibi_cols + rmssd_cols + sdnn_cols,
    'no_HRV':    hr_cols + acc_cols,  # duplicate of HR+ACC -- sanity check on column names
}

# ── 3. LOSO-CV per subset (same norm_fn, same model factory, same imputer) ───

def run_loso(X, y, groups, subjects, norm_fn):
    """Same contract/logic as _canonical_sensitivity_common.run_loso_et1000,
    extended to also track balanced accuracy per fold."""
    imp = SimpleImputer(strategy='median')
    fold_f1, fold_ba = {}, {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(norm_fn(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(norm_fn(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = et1000_factory()
        clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y[te], pred, average='macro', zero_division=0)
        fold_ba[subj] = balanced_accuracy_score(y[te], pred)
        proba = clf.predict_proba(X_te_n)
        pos_idx = list(clf.classes_).index(1)
        y_true_all.append(y[te])
        y_prob_all.append(proba[:, pos_idx])
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    try:
        auroc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auroc = float('nan')
    return fold_f1, fold_ba, auroc


results = {}
per_fold_f1 = {'subject': list(subjects)}

print('\nRunning LOSO-CV per subset (37 folds each, ET-1000, leak-fixed BL-norm)...')
for name, cols in SUBSETS.items():
    idx = [feat_cols.index(c) for c in cols]
    X_sub = X55[:, idx]
    fold_f1, fold_ba, auroc = run_loso(X_sub, y, groups, subjects, baseline_relative_scale)
    f1_vals = np.array([fold_f1[s] for s in subjects])
    ba_vals = np.array([fold_ba[s] for s in subjects])
    results[name] = {
        'n_features': len(cols),
        'macro_f1_mean': f1_vals.mean(), 'macro_f1_std': f1_vals.std(),
        'balanced_acc_mean': ba_vals.mean(), 'balanced_acc_std': ba_vals.std(),
        'auroc': auroc,
        'fold_f1': f1_vals,
    }
    per_fold_f1[f'{name}_f1'] = f1_vals
    print(f'  {name:10s} ({len(cols):2d} feat): '
          f'MacroF1={f1_vals.mean():.4f}±{f1_vals.std():.4f}  '
          f'BalAcc={ba_vals.mean():.4f}±{ba_vals.std():.4f}  AUC={auroc:.4f}')

# ── 4. Results table ───────────────────────────────────────────────────────────

df_results = pd.DataFrame([
    {'subset': name, 'n_features': r['n_features'],
     'macro_f1_mean': r['macro_f1_mean'], 'macro_f1_std': r['macro_f1_std'],
     'balanced_acc_mean': r['balanced_acc_mean'], 'balanced_acc_std': r['balanced_acc_std'],
     'auroc': r['auroc']}
    for name, r in results.items()
])
df_results.to_csv(OUT_CSV, index=False)
pd.DataFrame(per_fold_f1).to_csv(OUT_PER_FOLD_CSV, index=False)

print('\n' + '=' * 100)
print('RESULTS TABLE (LOSO-CV, 37 subjects, ET-1000, leak-fixed BL-norm)')
print('=' * 100)
print(f'{"Subset":<10} {"nFeat":>6} {"MacroF1":>10} {"±SD":>8} {"BalAcc":>10} {"±SD":>8} {"AUC":>8}')
for name, r in results.items():
    print(f'{name:<10} {r["n_features"]:>6} {r["macro_f1_mean"]:>10.4f} {r["macro_f1_std"]:>8.4f} '
          f'{r["balanced_acc_mean"]:>10.4f} {r["balanced_acc_std"]:>8.4f} {r["auroc"]:>8.4f}')

# ── 5. Wilcoxon: FULL vs HR+ACC, paired by test subject ───────────────────────

f1_full = results['FULL']['fold_f1']
f1_hracc = results['HR+ACC']['fold_f1']
stat, p_value = wilcoxon(f1_full, f1_hracc, alternative='two-sided')
sig = 'SIGNIFICANT' if p_value < 0.05 else 'not significant'

print('\n' + '-' * 100)
print('Paired Wilcoxon signed-rank (two-sided), 37 subjects, FULL vs HR+ACC macro-F1 per fold:')
print(f'  statistic={stat:.2f}  p-value={p_value:.6f}  ({sig} at alpha=0.05)')

# ── 6. Pearson correlation between the 5 channels at *_mean level ─────────────
# Computed on the raw (pre-BL-norm) canonical 686-row feature matrix, pooled
# across all subjects/windows -- documents the raw hr/ibi collinearity, not a
# per-subject or post-normalisation artifact.

mean_cols = ['acc_wrist_mean', 'hr_mean', 'ibi_mean', 'rmssd_mean', 'sdnn_mean']
mean_idx = [feat_cols.index(c) for c in mean_cols]
df_means = pd.DataFrame(X55[:, mean_idx], columns=mean_cols)
corr = df_means.corr(method='pearson')
corr.to_csv(OUT_CORR_CSV)

print('\n' + '-' * 100)
print('Pearson correlation, *_mean features, raw (pre-BL-norm), 686 windows pooled:')
print('-' * 100)
print(corr.round(4).to_string())

# ── 7. Direct answers ──────────────────────────────────────────────────────────

r_full = results['FULL']
r_hracc = results['HR+ACC']
r_hrv = results['HRV_only']
r_hr = results['HR_only']
r_acc = results['ACC_only']
r_nohrv = results['no_HRV']

print('\n' + '=' * 100)
print('DIRECT ANSWERS')
print('=' * 100)

diff = r_full['macro_f1_mean'] - r_hracc['macro_f1_mean']
print(f'(a) HR+ACC iguala FULL?')
print(f'    FULL (55 feat)   Macro-F1 = {r_full["macro_f1_mean"]:.4f} ± {r_full["macro_f1_std"]:.4f}')
print(f'    HR+ACC (22 feat) Macro-F1 = {r_hracc["macro_f1_mean"]:.4f} ± {r_hracc["macro_f1_std"]:.4f}')
print(f'    Diferença (FULL - HR+ACC) = {diff:+.4f}')
print(f'    no_HRV (sanity check, deve ser == HR+ACC): Macro-F1 = {r_nohrv["macro_f1_mean"]:.4f} '
      f'({"OK, nomes de colunas consistentes" if np.allclose(r_nohrv["fold_f1"], r_hracc["fold_f1"]) else "MISMATCH -- rever nomes de colunas"})')

print(f'\n(b) A diferença é estatisticamente significativa?')
print(f'    Wilcoxon signed-rank, FULL vs HR+ACC, n=37 sujeitos: '
      f'statistic={stat:.2f}, p={p_value:.6f} -> {sig} a alpha=0.05')

print(f'\n(c) HRV_only sozinho tem desempenho acima do acaso? Como se compara a HR_only?')
print(f'    HRV_only (33 feat: ibi+rmssd+sdnn) Macro-F1 = {r_hrv["macro_f1_mean"]:.4f} ± {r_hrv["macro_f1_std"]:.4f}  '
      f'BalAcc={r_hrv["balanced_acc_mean"]:.4f}  AUC={r_hrv["auroc"]:.4f}')
print(f'    HR_only  (11 feat: hr)             Macro-F1 = {r_hr["macro_f1_mean"]:.4f} ± {r_hr["macro_f1_std"]:.4f}  '
      f'BalAcc={r_hr["balanced_acc_mean"]:.4f}  AUC={r_hr["auroc"]:.4f}')
print(f'    ACC_only (11 feat: acc_wrist)      Macro-F1 = {r_acc["macro_f1_mean"]:.4f} ± {r_acc["macro_f1_std"]:.4f}  '
      f'BalAcc={r_acc["balanced_acc_mean"]:.4f}  AUC={r_acc["auroc"]:.4f}')
print(f'    (acaso, binário balanceado: Macro-F1≈0.50, BalAcc≈0.50, AUC≈0.50)')
print(f'    HRV_only vs HR_only, diferença de Macro-F1 = {r_hrv["macro_f1_mean"] - r_hr["macro_f1_mean"]:+.4f}')

print(f'\nCorrelação de Pearson hr_mean vs ibi_mean (raw): {corr.loc["hr_mean", "ibi_mean"]:.6f}')

print(f'\nSaved: {os.path.relpath(OUT_CSV, OUT_DIR)}')
print(f'Saved: {os.path.relpath(OUT_PER_FOLD_CSV, OUT_DIR)}')
print(f'Saved: {os.path.relpath(OUT_CORR_CSV, OUT_DIR)}')
