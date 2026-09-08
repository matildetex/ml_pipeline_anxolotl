"""
Non-contributing-component ablation -- canonical ET-1000 configuration.

Two isolated toggles, each run on the canonical pipeline with everything
else held fixed (same 55 wrist features unless noted, same phase-3
exclusion, same model, same metric, same LOSO loop as
3_anxolotl/friedman_nemenyi_anxolotl.py / _canonical_sensitivity_common.py):

  PART 1 -- Adaptive calibration
    The calibration_split procedure from 3_anxolotl/final_summary.py:84-93,
    reused verbatim: for the held-out test subject, half (cal_frac=0.5) of
    that subject's own baseline-labelled (label=0) test windows are moved
    into the training set before fitting, and excluded from evaluation.
    Applied AFTER BL-norm + imputation, exactly where final_summary.py
    applies it. "Without" = standard canonical loop (no calibration_split),
    evaluated on the full 686-window test folds -- this is also the
    canonical reference row and should reproduce 0.9169.
    NOTE: because calibration_split removes windows from the evaluation
    set, the "with calibration" row is scored on fewer windows per subject
    than the "without" row -- this is inherent to the method, not a bug;
    flagged again in the output.

  PART 2 -- Demographic features
    11 demographic/questionnaire fields (age, gender, height, weight, bmi,
    respiratory, medication, exercises, exercise_freq, athlete, stimulant)
    from Data_tese.xlsx (3_anxolotl/demographics_analysis.py:load_demographics),
    appended as extra columns (55 -> 66 features). "Without" = the 55-feature
    canonical reference row (should reproduce 0.9169). BL-norm is applied to
    ALL columns, including the demographic ones, exactly as in
    final_validation.py / all_models_demographics.py -- see the diagnostic
    printed below for what this does to per-subject-constant columns.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale, LEAK-FIXED 2026-07-19 (demographics_analysis.py)
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) -- 686 windows
Metric        : Macro F1 (mean across 37 subjects), SD, pooled out-of-fold AUROC

Statistical test: paired two-sided Wilcoxon signed-rank test on the 37
per-subject macro-F1 values, with vs without, for each component.

Does NOT overwrite any existing result file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, et1000_factory, baseline_relative_scale
)
from demographics_analysis import load_demographics

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'component_ablation_et1000_leakfixed.txt')
OUT_CSV_CAL = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'component_ablation_et1000_leakfixed_calibration_per_subject.csv')
OUT_CSV_DEMO = os.path.join(os.path.dirname(__file__), '..', 'results',
                             'component_ablation_et1000_leakfixed_demographics_per_subject.csv')
XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data_tese.xlsx')

WINDOW_SEC, STRIDE_SEC = 60, 30

df_all = load_cache()
X55, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('NON-CONTRIBUTING-COMPONENT ABLATION -- canonical ET-1000 configuration')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py)')
w(f'CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded. Windows: {n_after} '
  f'(of {n_before} before exclusion).')
w('Metric        : Macro F1.')
w()

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 -- ADAPTIVE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    """Verbatim from 3_anxolotl/final_summary.py:84-93."""
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx])


def run_loso_calibration(use_calibration):
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    n_eval_windows = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X55[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X55[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        y_tr_f, y_te_f = y[tr], y[te]
        if use_calibration:
            X_tr_n, y_tr_f, X_te_n, y_te_f = calibration_split(X_tr_n, y_tr_f, X_te_n, y_te_f, cal_frac=0.5)
        clf = et1000_factory()
        clf.fit(X_tr_n, y_tr_f)
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y_te_f, pred, average='macro', zero_division=0)
        n_eval_windows[subj] = len(y_te_f)
        proba = clf.predict_proba(X_te_n)
        pos_idx = list(clf.classes_).index(1)
        y_true_all.append(y_te_f)
        y_prob_all.append(proba[:, pos_idx])
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    try:
        auroc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auroc = float('nan')
    return fold_f1, auroc, n_eval_windows


w('-'*100)
w('PART 1 -- ADAPTIVE CALIBRATION (calibration_split from final_summary.py:84-93, cal_frac=0.5)')
w('-'*100)

fold_f1_nocal, auroc_nocal, nwin_nocal = run_loso_calibration(use_calibration=False)
fold_f1_cal, auroc_cal, nwin_cal = run_loso_calibration(use_calibration=True)

vals_nocal = np.array([fold_f1_nocal[s] for s in subjects])
vals_cal = np.array([fold_f1_cal[s] for s in subjects])
total_eval_nocal = sum(nwin_nocal.values())
total_eval_cal = sum(nwin_cal.values())

w(f'{"Config":<25}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}  {"TotalEvalWindows":>17}')
w('-'*100)
w(f'{"without calibration":<25}  {vals_nocal.mean():>8.4f}  {vals_nocal.std():>7.4f}  {auroc_nocal:>7.4f}  '
  f'{total_eval_nocal:>17}   <- reference, should match canonical ET-1000 (0.9169)')
w(f'{"with calibration":<25}  {vals_cal.mean():>8.4f}  {vals_cal.std():>7.4f}  {auroc_cal:>7.4f}  '
  f'{total_eval_cal:>17}')
w()
w(f'NOTE: eval-set size differs by design ({total_eval_nocal} vs {total_eval_cal} windows total across 37 '
  f'subjects) -- calibration_split removes the windows it moves into training from evaluation. The two rows')
w('are not scored on the identical set of windows; this is inherent to the calibration method, not an error.')
w()
stat_cal, p_cal = wilcoxon(vals_cal, vals_nocal, alternative='two-sided')
sig_cal = 'SIGNIFICANT' if p_cal < 0.05 else 'not significant'
w(f'Paired Wilcoxon signed-rank (two-sided), 37 subjects, with vs without calibration:')
w(f'  statistic={stat_cal:.2f}  p-value={p_cal:.6f}  ({sig_cal} at alpha=0.05)')
w()

pd.DataFrame({
    'subject': subjects,
    'macro_f1_without_calibration': [fold_f1_nocal[s] for s in subjects],
    'macro_f1_with_calibration': [fold_f1_cal[s] for s in subjects],
    'n_eval_windows_without': [nwin_nocal[s] for s in subjects],
    'n_eval_windows_with': [nwin_cal[s] for s in subjects],
}).to_csv(OUT_CSV_CAL, index=False)

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 -- DEMOGRAPHIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════

w('-'*100)
w('PART 2 -- DEMOGRAPHIC FEATURES (11 fields from Data_tese.xlsx, appended: 55 -> 66 features)')
w('-'*100)

df_demo = load_demographics(XLSX_PATH)
demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
             'respiratory', 'medication', 'exercises', 'exercise_freq',
             'athlete', 'stimulant']

def expand_demo(groups_arr):
    rows = []
    for g in groups_arr:
        rows.append(df_demo.loc[g, demo_cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(demo_cols)))
    return np.array(rows)

X_demo11 = expand_demo(groups)
missing_subjects = [s for s in subjects if s not in df_demo.index]
X66 = np.hstack([X55, X_demo11])


def run_loso_generic(X_mat):
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_mat[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_mat[te], y[te], groups[te]))
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


fold_f1_nodemo, auroc_nodemo = run_loso_generic(X55)
fold_f1_demo, auroc_demo = run_loso_generic(X66)

vals_nodemo = np.array([fold_f1_nodemo[s] for s in subjects])
vals_demo = np.array([fold_f1_demo[s] for s in subjects])

w(f'Subjects missing from Data_tese.xlsx (zero-filled demo row): {missing_subjects if missing_subjects else "none"}')
w()
w(f'{"Config":<25}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)
w(f'{"without demographics (55)":<25}  {vals_nodemo.mean():>8.4f}  {vals_nodemo.std():>7.4f}  {auroc_nodemo:>7.4f}'
  '   <- reference, should match canonical ET-1000 (0.9169)')
w(f'{"with demographics (66)":<25}  {vals_demo.mean():>8.4f}  {vals_demo.std():>7.4f}  {auroc_demo:>7.4f}')
w()

# Diagnostic: does per-subject BL-norm collapse the (per-subject-constant) demographic
# columns to zero, since they are applied the same z-score-vs-own-baseline treatment
# as the time-varying physiological columns?
X66_bl_full = baseline_relative_scale(X66, y, groups)  # single whole-dataset pass, diagnostic only
demo_col_start = X55.shape[1]
demo_stds_after_bl = X66_bl_full[:, demo_col_start:].std(axis=0)
demo_stds_before_bl = X_demo11.std(axis=0)
w('Diagnostic -- effect of per-subject BL-norm on the (per-subject-constant) demographic columns:')
w(f'{"Demo field":<15}  {"raw std (all windows)":>22}  {"std AFTER BL-norm":>18}')
for name, s_before, s_after in zip(demo_cols, demo_stds_before_bl, demo_stds_after_bl):
    w(f'{name:<15}  {s_before:>22.4f}  {s_after:>18.6f}')
w('(BL-norm z-scores each column per-subject against that subject\'s own baseline windows; a')
w('demographic value is identical across all of a subject\'s own windows, so its per-subject mean')
w('equals every value in that subject\'s column and its per-subject std is exactly 0 -- BL-norm')
w('therefore maps it to ~0 for every subject, regardless of the field\'s true cross-subject variance')
w('shown in the raw-std column above. This numeric collapse, not a lack of true relevance, is the')
w('most likely reason demographics contribute ~nothing once BL-norm is applied on top of them.')
w()

stat_demo, p_demo = wilcoxon(vals_demo, vals_nodemo, alternative='two-sided')
sig_demo = 'SIGNIFICANT' if p_demo < 0.05 else 'not significant'
w(f'Paired Wilcoxon signed-rank (two-sided), 37 subjects, with vs without demographics:')
w(f'  statistic={stat_demo:.2f}  p-value={p_demo:.6f}  ({sig_demo} at alpha=0.05)')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

pd.DataFrame({
    'subject': subjects,
    'macro_f1_without_demographics': [fold_f1_nodemo[s] for s in subjects],
    'macro_f1_with_demographics': [fold_f1_demo[s] for s in subjects],
}).to_csv(OUT_CSV_DEMO, index=False)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV_CAL}')
print(f'Saved: {OUT_CSV_DEMO}')
