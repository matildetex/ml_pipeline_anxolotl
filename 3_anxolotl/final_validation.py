"""
Robustness validation for the final model (dissertation credibility checks):

  1. Seed-robustness sweep: re-run the final config (ET-1000 tuned,
     watch+ecg+demo, calibration=0.5, no threshold) across 10 random seeds
     to see if F1=0.9541 is stable or a lucky draw from random_state=42.

  2. Paired statistical significance test: Wilcoxon signed-rank test on the
     37 per-subject F1 scores, baseline (watch-only, original ET-1000 params,
     no ECG/demo/calibration) vs. final config — tests whether the improvement
     is significant, not just numerically bigger.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale

os.makedirs('results', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    meta = pickle.load(f)
X_watch    = np.array(meta['X'], dtype=float)
y_all      = np.array(meta['y'])
groups_all = np.array(meta['groups'])

with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
    ecg_data = pickle.load(f)
X_ecg_raw = ecg_data['X_ecg'].astype(float)
ecg_cols  = ecg_data['ecg_cols']
X_ecg = X_ecg_raw.copy()
rmssd_idx = ecg_cols.index('ecg_rmssd')
std_idx   = ecg_cols.index('ecg_rr_std')
lfhf_idx  = ecg_cols.index('ecg_lf_hf')
X_ecg[X_ecg[:, rmssd_idx] > 300, :] = np.nan
X_ecg[X_ecg[:, std_idx]   > 300, :] = np.nan
X_ecg[X_ecg[:, lfhf_idx]  > 50, lfhf_idx] = np.nan

df_demo = load_demographics('Data_tese.xlsx')
demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
             'respiratory', 'medication', 'exercises', 'exercise_freq',
             'athlete', 'stimulant']

def expand(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_demo = expand(demo_cols)
X_WIN  = np.hstack([X_watch, X_ecg, X_demo])   # winning feature set, 88 feat
X_WD   = np.hstack([X_watch, X_demo])           # watch + demo (no ECG), 71 feat

subjects = np.unique(groups_all)

# ECG quality fallback: subjects with >50% NaN ECG windows use watch+demo
_ecg_nan_rows = np.all(np.isnan(X_ecg), axis=1)
ECG_QUALITY_THRESH = 0.50
fallback_subjects = {s for s in subjects
                     if _ecg_nan_rows[groups_all == s].mean() > ECG_QUALITY_THRESH}
print(f'ECG fallback subjects: {sorted(fallback_subjects)}')

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx  = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx])


def per_subject_f1(X, use_calibration, clf_params, seed):
    imp = SimpleImputer(strategy='median')
    f1s = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_tr, y_tr = X[train_mask], y_all[train_mask]
        X_te, y_te = X[test_mask],  y_all[test_mask]
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        if use_calibration:
            X_tr_n, y_tr, X_te_n, y_te = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)
        clf = ExtraTreesClassifier(random_state=seed, n_jobs=-1, **clf_params)
        clf.fit(X_tr_n, y_tr)
        y_pred = clf.predict(X_te_n)
        f1s.append(f1_score(y_te, y_pred, average='binary', zero_division=0))
    return np.array(f1s)

def per_subject_f1_adaptive(seed):
    """Final config with ECG-quality fallback per test subject."""
    imp = SimpleImputer(strategy='median')
    f1s = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_feat  = X_WD  if subj in fallback_subjects else X_WIN
        params  = FINAL_PARAMS_WD if subj in fallback_subjects else FINAL_PARAMS
        X_tr, y_tr = X_feat[train_mask], y_all[train_mask]
        X_te, y_te = X_feat[test_mask],  y_all[test_mask]
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        X_tr_n, y_tr, X_te_n, y_te = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)
        clf = ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params)
        clf.fit(X_tr_n, y_tr)
        f1s.append(f1_score(y_te, clf.predict(X_te_n), average='binary', zero_division=0))
    return np.array(f1s)


FINAL_PARAMS    = dict(n_estimators=1000, max_features=0.3, min_samples_leaf=1, class_weight='balanced')
FINAL_PARAMS_WD = dict(n_estimators=1000, max_features=0.6, min_samples_leaf=1, class_weight='balanced')
BASELINE_PARAMS = dict(n_estimators=1000, max_features=0.5, min_samples_leaf=1, class_weight='balanced')

# ═══════════════════════════════════════════════════════════════════════════
# 1. SEED-ROBUSTNESS SWEEP (final config)
# ═══════════════════════════════════════════════════════════════════════════

print('='*60)
print('1. SEED-ROBUSTNESS SWEEP — final config across 10 seeds')
print('='*60)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
seed_f1s = []
for seed in seeds:
    f1s = per_subject_f1_adaptive(seed=seed)
    mf1 = f1s.mean()
    seed_f1s.append(mf1)
    print(f'  seed={seed:<4} F1={mf1:.4f}')

seed_f1s = np.array(seed_f1s)
print(f'\nAcross {len(seeds)} seeds: mean={seed_f1s.mean():.4f}  std={seed_f1s.std():.4f}  '
      f'min={seed_f1s.min():.4f}  max={seed_f1s.max():.4f}')
pd.DataFrame({'seed': seeds, 'f1': seed_f1s}).to_csv('results/seed_robustness.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 2. PAIRED SIGNIFICANCE TEST (baseline vs. final, seed=42)
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*60)
print('2. PAIRED WILCOXON SIGNED-RANK TEST — baseline vs. final')
print('='*60)

f1_baseline = per_subject_f1(X_watch, use_calibration=False, clf_params=BASELINE_PARAMS, seed=42)
f1_final    = per_subject_f1_adaptive(seed=42)

print(f'Baseline (watch-only, ET-1000):       mean F1={f1_baseline.mean():.4f}')
print(f'Final (watch+ecg+demo+cal, tuned):    mean F1={f1_final.mean():.4f}')
print(f'Mean per-subject improvement:         {(f1_final - f1_baseline).mean():+.4f}')
print(f'Subjects improved / same / worse:     '
      f'{(f1_final > f1_baseline).sum()} / {(f1_final == f1_baseline).sum()} / {(f1_final < f1_baseline).sum()} '
      f'(of {len(subjects)})')

stat, pval = wilcoxon(f1_final, f1_baseline)
print(f'\nWilcoxon signed-rank: statistic={stat:.2f}, p-value={pval:.6f}')
print('=> ' + ('SIGNIFICANT' if pval < 0.05 else 'NOT significant') + ' at alpha=0.05')

pd.DataFrame({
    'subject': subjects, 'f1_baseline': f1_baseline, 'f1_final': f1_final,
    'delta': f1_final - f1_baseline,
}).to_csv('results/paired_significance.csv', index=False)

print('\nDONE')
