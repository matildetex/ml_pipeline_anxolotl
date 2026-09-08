"""
CANONICAL COPY (2026-09-06) -- this is the active source for thesis Table 4.10
(Section 4.3, WESAD Wrist-Only), confirmed to match all 7 individual model
rows exactly. Copied from
results/section1_2_wesad_wrist55_blnorm_2026-07-26/run_wesad_wrist55_blnorm.py
(the original 2026-07-26 run, kept there untouched as the historical record
-- do not edit that copy). See README.md / DATA.md for the full provenance
note on why this file lives in 1_wesad/wrist_only/ and not the results/
snapshot.

Section 1.2 -- WESAD wrist-only, 55-feature set, BL-norm variant.
Companion to results/section1_2_wesad_wrist55_2026-07-26/ (StandardScaler
run). Same 55 features, same 7 models, same LOSO-CV, same hyperparameters --
the ONLY thing that changes is the normalization: BL-norm (baseline-relative
per-subject z-score) instead of StandardScaler. Self-contained: everything
written by this script lives under this same directory.

BL-NORM IMPLEMENTATION -- provenance (printed again at runtime, see STEP 1):
PATH NOTE (2026-09-07): item (4) below, cross_dataset_blnorm.py, lived at
2_cross_dataset/ when this search was originally done (2026-07-26/09-06);
archived as pre-leak-fix code during the 2026-09-07 cross-dataset reorg
(see DATA.md/README.md) -- paths below updated to archive/cross_dataset_
blnorm.py accordingly. This does not change the reasoning: the function
itself is untouched, still byte-for-byte identical to (1)-(3).
Searched the whole repo for BL-norm implementations. Found FIVE:
  1. scripts/common.py :: bl_norm_labels(X, y, groups)
  2. 3_anxolotl/demographics_analysis.py :: baseline_relative_scale(X, y, groups)
  3. 3_anxolotl/window_size_ablation_no_phase3.py :: baseline_relative_scale(X, y, groups)
  4. archive/cross_dataset_blnorm.py :: baseline_relative_scale(X, y, groups)
  5. 3_anxolotl/_canonical_sensitivity_common.py :: baseline_relative_scale_capped(X, y, groups, cap=None)
(1)-(4) are byte-for-byte the same logic (verified by reading all four): for
each subject, take the LEADING RUN of y==0 windows (stops at the first
non-zero label -- the "fixed 2026-07-19" leak fix, see demographics_analysis.py's
docstring: the old version used every y==0 row anywhere in the timeline,
which is self-referential/leaky when a fold's own ground truth picks its
own normalisation reference). Compute per-feature mean/std over that
leading run, z-score the WHOLE subject's data (baseline + everything else)
against it. (5) is the same rule with an added optional cap on how many of
the leading-run windows are used -- cap=None (its default) reproduces (1)-(4)
exactly; it exists for a calibration-window-length ablation, not used here.

There is a SIXTH, DIFFERENT function that is NOT used: bl_norm_protocol /
baseline_relative_scale_protocol(X, groups, n_baseline=8) (in scripts/common.py
and archive/cross_dataset_blnorm.py). This ignores labels entirely and
always uses the first 8 windows per subject as reference -- valid ONLY for
Anxolotl, whose protocol has a fixed 5-min/8-window baseline regardless of
label. WESAD's baseline runs are label-defined and variable-length per subject
(~1080-1140 windows here, not 8) -- using n_baseline=8 on WESAD would silently
truncate the baseline reference to the first 8 seconds-ish of an ~18-19min
rest period. Confirmed via the label-derived leading-run count for this exact
55-feature dataset -- S10=1121, S15=1116, S2=1085, ... -- WHICH EXACTLY MATCHES
the "BL=1121"/"BL=1116"/"BL=1085"/... counts printed in the pre-existing
results/wesad_55plus_ecg_loso.txt (the file this run's ExtraTrees is meant to
reproduce). That match is strong evidence the leading-run/label-based function
is the one that actually produced that reference number.

Import source used in THIS script: archive/cross_dataset_blnorm.py
:: baseline_relative_scale -- chosen because its own docstring explicitly
says "Only used here for WESAD's training side" (i.e. it has direct WESAD
provenance in this repo, not just Anxolotl), and it is functionally identical
to the other three per-subject/leading-run implementations, so the choice
among (1)-(4) does not change any number below.

Per-subject BL-norm needs no per-fold refitting: it only ever uses a
subject's OWN leading-baseline-run statistics, and LOSO never splits a
single subject's rows across train/test, so computing it once up front for
the full dataset (as demographics_analysis.py / shap_canonical_et1000_leakfixed.py
do) is equivalent to computing it separately inside each fold. SimpleImputer
(mean), fit on the train fold only, still runs per-fold after BL-norm to
handle residual NaNs (BL-norm passes NaNs through unchanged) -- there is no
StandardScaler step in this run; BL-norm replaces it entirely, matching how
ExtraTrees+BL-norm was described in the reference file (BL-norm only, no
separate scaler mentioned).
"""
import pickle, os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

T0 = time.time()
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Output goes to a fresh local subfolder -- NOT the 2026-07-26 snapshot dir,
# which stays untouched as the historical record of the original run.
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wrist55_blnorm_rerun')
STD_DIR   = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_2026-07-26')
os.makedirs(OUT_DIR, exist_ok=True)

# baseline_relative_scale, copied VERBATIM from archive/cross_dataset_blnorm.py
# (that module executes top-level data-loading code with a relative path at import
# time, so `from cross_dataset_blnorm import baseline_relative_scale` fails outside
# its own directory -- confirmed by trying it: FileNotFoundError on
# '../processed_data/wesad/processed_data.pkl'. Inlining the exact function body
# avoids that side effect while using the identical, unmodified logic -- not a
# reimplementation.)
def baseline_relative_scale(X, y, groups):
    """Uses the leading run of label=0 windows per subject as baseline
    reference (for WESAD training). Fixed 2026-07-19: previously used every
    y==0 row anywhere in the timeline, which is a self-referential leak when
    this function is ever applied to a fold using its own true labels — see
    3_anxolotl/demographics_analysis.py for the full writeup. Only used here
    for WESAD's training side, but fixed for consistency / to stop this
    pattern being copy-pasted into a leaky context again."""
    X = np.array(X, dtype=float).copy()
    y = np.array(y)
    groups = np.array(groups)
    for subj in np.unique(groups):
        subj_idx = np.where(groups == subj)[0]
        n_baseline = 0
        for v in y[subj_idx]:
            if v == 0:
                n_baseline += 1
            else:
                break
        baseline_idx = subj_idx[:n_baseline]
        if len(baseline_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[baseline_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

# ===========================================================================
# STEP 0 -- validate data source + 55-column feature set (same as StdScaler run)
# ===========================================================================
w('='*78)
w('STEP 0 -- DATA SOURCE + FEATURE SET VALIDATION (must match the StandardScaler run)')
w('='*78)

PKL_PATH = os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl')
mtime = pd.Timestamp(os.path.getmtime(PKL_PATH), unit='s')
w(f'File: {os.path.relpath(PKL_PATH, ROOT)}  (mtime {mtime})')

with open(PKL_PATH, 'rb') as f:
    d = pickle.load(f)

X_all = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
y_all = pd.Series(np.asarray(d['y']))
groups_all = pd.Series(np.asarray(d['groups']))
feat_cols_all = list(d['feature_cols'])
w(f'  shape: {X_all.shape[0]} rows x {X_all.shape[1]} cols, n_subjects={len(np.unique(groups_all))}')

WRIST_PREFIXES = ('acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_')
reduced_cols = [c for c in feat_cols_all if c.startswith(WRIST_PREFIXES)]
assert len(reduced_cols) == 55, f'STOP: expected 55 columns, got {len(reduced_cols)}: {reduced_cols}'
w(f'Reduced column selection (prefixes {WRIST_PREFIXES}): {len(reduced_cols)} columns.')
w('ASSERTION PASSED: exactly 55 columns (identical set to the StandardScaler run).')
w()
w('Full list of the 55 selected feature names:')
for i, c in enumerate(reduced_cols, 1):
    w(f'  {i:2d}. {c}')
w()

X_reduced = X_all[reduced_cols].reset_index(drop=True)
y = y_all.reset_index(drop=True)
groups = groups_all.reset_index(drop=True)

# ===========================================================================
# STEP 1 -- BL-norm provenance + baseline-window sanity check
# ===========================================================================
w('='*78)
w('STEP 1 -- BL-NORM IMPLEMENTATION: PROVENANCE + BASELINE-WINDOW DEFINITION')
w('='*78)
w('Implementations found in the repo (full reasoning in this file\'s module docstring):')
w('  1. scripts/common.py :: bl_norm_labels(X, y, groups)')
w('  2. 3_anxolotl/demographics_analysis.py :: baseline_relative_scale(X, y, groups)')
w('  3. 3_anxolotl/window_size_ablation_no_phase3.py :: baseline_relative_scale(X, y, groups)')
w('  4. archive/cross_dataset_blnorm.py :: baseline_relative_scale(X, y, groups)   <-- USED HERE')
w('  5. 3_anxolotl/_canonical_sensitivity_common.py :: baseline_relative_scale_capped(X, y, groups, cap=None)')
w('  (rejected) scripts/common.py :: bl_norm_protocol / archive/cross_dataset_blnorm.py :: '
  'baseline_relative_scale_protocol(X, groups, n_baseline=8) -- fixed-8-window, '
  'Anxolotl-protocol-only, label-agnostic. Wrong for WESAD (see below).')
w()
w('(1)-(4) are the SAME leak-fixed (2026-07-19) logic: per subject, leading run of '
  'y==0 windows (stops at first non-zero label), used as the mean/std reference for '
  'a per-feature z-score applied to that subject\'s entire timeline. (5) with cap=None '
  'reproduces this exactly.')
w()
w('CHOSEN: archive/cross_dataset_blnorm.py :: baseline_relative_scale')
w('  Reason: its own docstring states it is "used here for WESAD\'s training side" -- '
  'the only implementation with direct, documented WESAD provenance in this repo '
  '(the others were written for/used on Anxolotl). Functionally identical to (1)-(3), '
  'so this choice changes nothing numerically vs. picking one of those instead.')
w()

w('Baseline-window definition for WESAD (from 1_wesad/wesad_preprocessing.py line 14: '
  '"WESAD labels: 1->baseline(0), 2->stress(1), others->skip"): y==0 IS EXACTLY the '
  'WESAD protocol\'s Baseline condition -- amusement/meditation (other WESAD labels) are '
  'never included in this dataset at all, so there is no ambiguity about what counts as '
  '"baseline" here. Because baseline is a single continuous block at the START of every '
  'subject\'s recording (by protocol design) and no window is ever re-labelled 0 later, '
  'the "leading run of y==0" equals ALL of that subject\'s y==0 windows -- verified below.')
w()

y_np = y.values.astype(int)
groups_np = groups.values
w(f'{"Subject":10s} {"total_windows":>13s} {"leading_run_BL":>15s} {"all_y0_windows":>15s} {"match":>6s}')
all_match = True
for s in np.unique(groups_np):
    idx = np.where(groups_np == s)[0]
    n_bl = 0
    for v in y_np[idx]:
        if v == 0: n_bl += 1
        else: break
    all_y0 = int((y_np[idx] == 0).sum())
    match = (n_bl == all_y0)
    all_match = all_match and match
    w(f'{str(s):10s} {len(idx):13d} {n_bl:15d} {all_y0:15d} {str(match):>6s}')
w()
w(f'ALL SUBJECTS leading-run == all-y==0 windows: {all_match} (confirms no ambiguity: '
  f'baseline is one contiguous block per subject, not scattered).')
w()
w('Cross-check against results/wesad_55plus_ecg_loso.txt (the file this run\'s ExtraTrees '
  'is meant to reproduce): that file prints BL=1121 (S10), BL=1116 (S15), BL=1085 (S2), '
  'etc. -- IDENTICAL to the leading_run_BL counts above. This is strong evidence the '
  'label-based/leading-run function (not the fixed-n=8 protocol variant) is what produced '
  'that reference number.')
w()

# ===========================================================================
# STEP 2 -- apply BL-norm (once, globally -- safe under LOSO, see module docstring)
# ===========================================================================
w('='*78)
w('STEP 2 -- APPLYING BL-NORM')
w('='*78)
X_bl = baseline_relative_scale(X_reduced.values.astype(float), y_np, groups_np)
n_nan_before = int(np.isnan(X_reduced.values.astype(float)).sum())
n_nan_after = int(np.isnan(X_bl).sum())
w(f'X_bl shape: {X_bl.shape}. NaN count before={n_nan_before}, after BL-norm={n_nan_after} '
  f'(BL-norm passes NaNs through unchanged; SimpleImputer(mean) still runs per-fold below).')
w('Computed ONCE for the full dataset (not per-fold): per-subject reference stats only '
  'depend on that subject\'s own leading-baseline-run, and LOSO never splits a subject\'s '
  'rows across train/test, so this is equivalent to computing it inside each fold.')
w()
X_bl_df = pd.DataFrame(X_bl, columns=reduced_cols)

# ===========================================================================
# STEP 3 -- 7 standalone models, LOSO-CV, BL-norm (no StandardScaler, no SMOTE)
# ===========================================================================
w('='*78)
w('STEP 3 -- 7 STANDALONE MODELS, LOSO-CV (15 folds), 55-feature wrist set, BL-norm')
w('='*78)
w('Hyperparameters: IDENTICAL to results/section1_2_wesad_wrist55_2026-07-26/ -- only '
  'the normalization changes (BL-norm here vs StandardScaler there). Per-fold '
  'SimpleImputer(mean) fit on train only (BL-norm itself is not fold-dependent, see '
  'above); no SMOTE.')
w()

def make_logreg():
    return LogisticRegression(penalty='l1', C=0.001, solver='saga', max_iter=1000, random_state=42)

def make_svm():
    return SVC(kernel='poly', degree=3, C=0.1, gamma='scale', class_weight='balanced',
               probability=True, random_state=42)

def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=10, max_features='sqrt',
                                   random_state=42, n_jobs=-1)

def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          eval_metric='logloss', random_state=42, n_jobs=-1)

def make_lgbm():
    return LGBMClassifier(n_estimators=305, max_depth=4, learning_rate=0.195, num_leaves=37,
                           random_state=42, verbose=-1, n_jobs=-1)

def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.1,
                                           random_state=42)

def make_et():
    return ExtraTreesClassifier(n_estimators=1000, max_features=0.5, class_weight='balanced',
                                 random_state=42, n_jobs=-1)

MODEL_BUILDERS = {
    'LogisticRegression': make_logreg,
    'SVM': make_svm,
    'RandomForest': make_rf,
    'XGBoost': make_xgb,
    'LightGBM': make_lgbm,
    'HistGradientBoosting': make_hgb,
    'ExtraTrees': make_et,
}

logo = LeaveOneGroupOut()

def stratified_subsample(X, y, groups, frac=0.4, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y.values == 0)[0]
    idx1 = np.where(y.values == 1)[0]
    n0, n1 = int(len(idx0) * frac), int(len(idx1) * frac)
    sel = np.sort(np.concatenate([rng.choice(idx0, n0, replace=False),
                                   rng.choice(idx1, n1, replace=False)]))
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True), groups.iloc[sel].reset_index(drop=True)

X_svm, y_svm, groups_svm = stratified_subsample(X_bl_df, y, groups, frac=0.4)
w(f'SVM deviation (same flag as the StandardScaler run): 40% stratified subsample '
  f'({X_svm.shape[0]} of {X_bl_df.shape[0]} windows). All other 6 models use the full data.')
w()

def run_loso(model_builder, X, y, groups):
    all_true, all_pred, all_prob = [], [], []
    fold_f1s = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(X_train)
        X_test_p = imputer.transform(X_test)
        model = model_builder()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_true.extend(y_test); all_pred.extend(y_pred); all_prob.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    return dict(
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        fold_f1s=[float(v) for v in fold_f1s],
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        accuracy=float(accuracy_score(all_true, all_pred)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        n_folds=len(fold_f1s),
    )

results = {}
for mname, builder in MODEL_BUILDERS.items():
    Xd, yd, gd = (X_svm, y_svm, groups_svm) if mname == 'SVM' else (X_bl_df, y, groups)
    w(f'-- {mname} --')
    t0 = time.time()
    try:
        res = run_loso(builder, Xd, yd, gd)
        dt = time.time() - t0
        res['runtime_sec'] = dt
        results[mname] = res
        w(f'   MacroF1={res["macro_f1"]:.4f}±{res["macro_f1_std"]:.4f}  '
          f'BalAcc={res["balanced_accuracy"]:.4f}  AUROC={res["roc_auc"]:.4f}  '
          f'({dt:.1f}s, {res["n_folds"]} folds)')
    except Exception as e:
        dt = time.time() - t0
        w(f'   ERROR after {dt:.1f}s: {type(e).__name__}: {e}')
        results[mname] = {'error': f'{type(e).__name__}: {e}'}
w()

with open(os.path.join(OUT_DIR, 'metrics_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ===========================================================================
# STEP 4 -- sanity check vs results/wesad_55plus_ecg_loso.txt
# ===========================================================================
w('='*78)
w('STEP 4 -- SANITY CHECK: ExtraTrees vs results/wesad_55plus_ecg_loso.txt')
w('='*78)
REF_F1, REF_AUC = 0.6902, 0.8536
et = results.get('ExtraTrees', {})
if 'error' in et:
    w(f'ExtraTrees ERRORED -- cannot sanity-check. Error: {et["error"]}')
    sanity_ok = False
else:
    d_f1 = abs(et['macro_f1'] - REF_F1)
    d_auc = abs(et['roc_auc'] - REF_AUC)
    sanity_ok = d_f1 <= 0.02
    w(f'  Reference (wesad_55plus_ecg_loso.txt): MacroF1={REF_F1:.4f}  AUROC={REF_AUC:.4f}')
    w(f'  This run                             : MacroF1={et["macro_f1"]:.4f}  AUROC={et["roc_auc"]:.4f}')
    w(f'  |delta MacroF1| = {d_f1:.4f}  |delta AUROC| = {d_auc:.4f}')
    if sanity_ok:
        w('  SANITY CHECK PASSED (MacroF1 within 0.02 of the reference).')
    else:
        w('  *** SANITY CHECK FAILED *** (MacroF1 differs by more than 0.02 from the '
          'reference) -- per instructions, treat these numbers as NOT YET CONFIRMED and '
          'flag to the user before using them; the BL-norm applied here may not be the '
          'same one that produced the reference file.')
w()

# ===========================================================================
# STEP 5 -- summary table, sorted by Macro F1
# ===========================================================================
w('='*78)
w('STEP 5 -- SUMMARY TABLE (sorted by Macro F1 desc), BL-norm')
w('='*78)
ok_rows = [(m, r) for m, r in results.items() if 'error' not in r]
ok_rows.sort(key=lambda kv: kv[1]['macro_f1'], reverse=True)
err_rows = [(m, r) for m, r in results.items() if 'error' in r]

header = f'{"Model":24s} {"Macro F1":>16s} {"BalAcc":>8s} {"AUROC":>8s}'
w(header); w('-' * len(header))
table_lines = [header, '-' * len(header)]
for m, r in ok_rows:
    line = (f'{m:24s} {r["macro_f1"]:.4f} ± {r["macro_f1_std"]:.4f}  '
            f'{r["balanced_accuracy"]:.4f} {r["roc_auc"]:.4f}')
    w(line); table_lines.append(line)
for m, r in err_rows:
    line = f'{m:24s} ERROR: {r["error"]}'
    w(line); table_lines.append(line)
w()

with open(os.path.join(OUT_DIR, 'table_1_7_draft_blnorm.txt'), 'w') as f:
    f.write('\n'.join(table_lines) + '\n')

# ===========================================================================
# STEP 6 -- comparison table, StandardScaler vs BL-norm
# ===========================================================================
w('='*78)
w('STEP 6 -- COMPARISON: StandardScaler (previous run) vs BL-norm (this run)')
w('='*78)
std_path = os.path.join(STD_DIR, 'metrics_results.json')
if os.path.exists(std_path):
    with open(std_path) as f:
        std_results = json.load(f)
    header2 = f'{"Model":24s} {"F1 (StdScaler)":>15s} {"F1 (BL-norm)":>13s} {"Delta":>8s} {"AUROC (Std)":>12s} {"AUROC (BL)":>11s} {"Delta":>8s}'
    w(header2); w('-'*len(header2))
    cmp_lines = [header2, '-'*len(header2)]
    for m in MODEL_BUILDERS:
        std_r = std_results.get(m, {})
        bl_r = results.get(m, {})
        if 'error' in std_r or 'error' in bl_r:
            line = f'{m:24s} ERROR in one or both runs'
        else:
            df1 = bl_r['macro_f1'] - std_r['macro_f1']
            dauc = bl_r['roc_auc'] - std_r['roc_auc']
            line = (f'{m:24s} {std_r["macro_f1"]:15.4f} {bl_r["macro_f1"]:13.4f} {df1:+8.4f} '
                     f'{std_r["roc_auc"]:12.4f} {bl_r["roc_auc"]:11.4f} {dauc:+8.4f}')
        w(line); cmp_lines.append(line)
    w()
    with open(os.path.join(OUT_DIR, 'comparison_stdscaler_vs_blnorm.txt'), 'w') as f:
        f.write('\n'.join(cmp_lines) + '\n')
else:
    w(f'  WARNING: {std_path} not found -- cannot build comparison table.')

w()
w('='*78)
w(f'DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
w('SHAP NOT regenerated in this job, per instructions (1.2\'s SHAP already exists in the '
  'StandardScaler run directory).')
w('='*78)

with open(os.path.join(OUT_DIR, 'full_log.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nAll output under: {os.path.relpath(OUT_DIR, ROOT)}/')
