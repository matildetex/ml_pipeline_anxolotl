"""
Normalisation-scheme ablation -- canonical ET-1000 configuration.

Replaces baseline_relative_scale with each of three schemes, on the
IDENTICAL canonical LOSO pipeline (same 55 features, same phase-3
exclusion, same model, same metric, same evaluation loop structure as
3_anxolotl/friedman_nemenyi_anxolotl.py / _canonical_sensitivity_common.py):

  1. no_norm          -- no normalisation at all (median imputation only)
  2. standard_scaler  -- sklearn StandardScaler, fit on the training fold
                         only, applied to both train and test
  3. blnorm_canonical -- baseline_relative_scale, LEAK-FIXED 2026-07-19
                         (reference row -- should reproduce 0.9169)

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : varies per row (see above); canonical row uses the
                LEAK-FIXED baseline_relative_scale (demographics_analysis.py)
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) -- 686 windows
Features      : 55 (5 wrist channels: acc_wrist, hr, ibi, rmssd, sdnn x 11 stats)
Metric        : Macro F1 (mean across 37 subjects), SD, pooled out-of-fold AUROC

Statistical test: paired two-sided Wilcoxon signed-rank test on the 37
per-subject macro-F1 values, BL-norm vs no-norm and BL-norm vs StandardScaler.

Does NOT overwrite any existing result file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, et1000_factory, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'normalization_ablation_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'normalization_ablation_et1000_leakfixed_per_subject.csv')

WINDOW_SEC, STRIDE_SEC = 60, 30

df_all = load_cache()
X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)


def run_loso_paired(transform_pair_fn):
    """
    transform_pair_fn(X_tr, y_tr, g_tr, X_te, y_te, g_te) -> (X_tr_out, X_te_out)
    Generalises the canonical loop so the normalisation step can see both
    sides of the fold jointly (needed for StandardScaler, which must fit
    on train only, unlike baseline_relative_scale which is applied
    independently to each side in the original canonical loop).
    """
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        Xtr_t, Xte_t = transform_pair_fn(X[tr], y[tr], groups[tr], X[te], y[te], groups[te])
        X_tr_n = np.nan_to_num(Xtr_t)
        X_te_n = np.nan_to_num(Xte_t)
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = et1000_factory()
        clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y[te], pred, average='macro', zero_division=0)
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
    return fold_f1, auroc


def transform_no_norm(Xtr, ytr, gtr, Xte, yte, gte):
    return Xtr, Xte


def transform_standard(Xtr, ytr, gtr, Xte, yte, gte):
    Xtr0 = np.nan_to_num(Xtr)
    Xte0 = np.nan_to_num(Xte)
    sc = StandardScaler().fit(Xtr0)
    return sc.transform(Xtr0), sc.transform(Xte0)


def transform_blnorm(Xtr, ytr, gtr, Xte, yte, gte):
    return baseline_relative_scale(Xtr, ytr, gtr), baseline_relative_scale(Xte, yte, gte)


CONFIGS = [
    ('no_norm',          transform_no_norm),
    ('standard_scaler',  transform_standard),
    ('blnorm_canonical', transform_blnorm),
]

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('NORMALISATION-SCHEME ABLATION -- canonical ET-1000 configuration')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : varies per row -- no_norm (impute only) / StandardScaler (fit-train-only) /')
w('                blnorm_canonical (baseline_relative_scale, LEAK-FIXED 2026-07-19, reference row)')
w(f'CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded. Windows: {n_after} '
  f'(of {n_before} before exclusion).')
w('Features      : 55 (5 wrist channels x 11 stats). Metric: Macro F1.')
w()
w(f'{"Scheme":<20}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)

per_subject_cols = {}
results = {}

for name, fn in CONFIGS:
    fold_f1, auroc = run_loso_paired(fn)
    vals = np.array([fold_f1[s] for s in subjects])
    mean_f1, sd_f1 = vals.mean(), vals.std()
    per_subject_cols[name] = pd.Series(fold_f1)
    results[name] = vals
    w(f'{name:<20}  {mean_f1:>8.4f}  {sd_f1:>7.4f}  {auroc:>7.4f}'
      + ('   <- reference, should match canonical ET-1000 (0.9169)' if name == 'blnorm_canonical' else ''))

w()
w('-'*100)
w('Paired Wilcoxon signed-rank test (two-sided), 37 per-subject Macro F1 values, BL-norm vs each alternative')
w('-'*100)
bl_vals = results['blnorm_canonical']
for name in ['no_norm', 'standard_scaler']:
    alt_vals = results[name]
    stat, p = wilcoxon(bl_vals, alt_vals, alternative='two-sided')
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    w(f'BL-norm vs {name:<20}: statistic={stat:.2f}  p-value={p:.6f}  ({sig} at alpha=0.05)')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

df_out = pd.DataFrame(per_subject_cols)
df_out.to_csv(OUT_CSV)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
