"""
Section 1.2 -- WESAD wrist-only, 55-feature set, BL-norm: the missing
Ensemble(ET+RF+HGB) row for Table 1.7, to sit next to Table 1.11's Anxolotl
Ensemble(ET+RF+HGB) row.

Reuses, VERBATIM, three things already in the repo -- nothing here is a new
model definition:
  1. The 55-feature wrist column selection + WESAD pkl loading, copied from
     this same directory's run_wesad_wrist55_blnorm.py (STEP 0 below).
  2. baseline_relative_scale, copied from 2_cross_dataset/cross_dataset_blnorm.py
     (same reasoning as run_wesad_wrist55_blnorm.py for inlining rather than
     importing: that module runs relative-path data loading at import time).
  3. The Ensemble(ET+RF+HGB) VotingClassifier definition, copied VERBATIM
     from 3_anxolotl/friedman_nemenyi_anxolotl.py line 70-74 (identical copy
     also exists at 3_anxolotl/recompute_macro_f1.py line 173-176) -- this is
     the exact object behind Table 1.11's "Ensemble(ET+RF+HGB)" row.

*** FLAG #1 -- voting mode mismatch ***
The task described this as a "soft-voting" ensemble. The actual canonical
definition in the repo uses voting='hard':
    VotingClassifier(voting='hard', estimators=[
        ('et',  ExtraTreesClassifier(n_estimators=300, class_weight='balanced',
                                      random_state=42, n_jobs=-1)),
        ('rf',  RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                        random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                learning_rate=0.05, random_state=42)),
    ])
Per instructions ("localiza a definicao... reutiliza-a, nao reimplementes"),
this script uses the REPO'S ACTUAL definition (hard voting) rather than the
task's verbal description (soft voting), since the whole point is
comparability with the exact object that produced Table 1.11 -- not with a
differently-behaving reimplementation. Reported to the user directly; not
silently "corrected" in either direction.

*** FLAG #2 -- no canonical AUROC exists for this ensemble, on either dataset ***
sklearn's VotingClassifier only exposes predict_proba when voting='soft';
with voting='hard' (as canonically defined), calling predict_proba raises
AttributeError. This is not a bug introduced here: results_master.csv's own
row for Anxolotl's "Ensemble(ET+RF+HGB)" (source: friedman_nemenyi_anxolotl_
wrist_loso.txt) already lists AUROC=MISSING for exactly this reason -- Table
1.11 itself has never had a real AUROC for this row. This script:
  (a) reports Macro F1 and Balanced Accuracy from the actual hard-voting
      predictions -- fully comparable to Table 1.11's row.
  (b) does NOT claim a canonical AUROC for the hard-voting ensemble itself.
  (c) ADDITIONALLY computes a clearly-labeled SUPPLEMENTARY, NON-CANONICAL
      AUROC by averaging the predict_proba of the three already-fitted
      sub-estimators (model.named_estimators_) -- these are the exact same
      fitted et/rf/hgb objects inside the canonical VotingClassifier, not new
      models -- uniformly averaged, per fold. This is offered only so Table
      1.7 is not left with a completely empty AUROC cell; it is NOT the same
      quantity Table 1.11 reports (which is nothing), and should be labeled
      as such if used.
"""
import pickle, os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
)

T0 = time.time()
ROOT    = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
OUT_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_blnorm_2026-07-26')
assert os.path.isdir(OUT_DIR), f'expected output dir to already exist: {OUT_DIR}'

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

# ===========================================================================
# STEP 0 -- same 55-column WESAD wrist feature set as today's standalone run
# ===========================================================================
w('='*78)
w('STEP 0 -- DATA SOURCE + FEATURE SET VALIDATION (must match the standalone run)')
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
w('ASSERTION PASSED: exactly 55 columns (identical set to the standalone-model run).')
w()

X_reduced = X_all[reduced_cols].reset_index(drop=True)
y = y_all.reset_index(drop=True)
groups = groups_all.reset_index(drop=True)

# ===========================================================================
# STEP 1 -- BL-norm, copied verbatim from 2_cross_dataset/cross_dataset_blnorm.py
# (identical inlining rationale as run_wesad_wrist55_blnorm.py: that module
# executes relative-path data loading at import time)
# ===========================================================================
w('='*78)
w('STEP 1 -- BL-NORM (2_cross_dataset/cross_dataset_blnorm.py :: baseline_relative_scale, '
  'same function used for the 7 standalone models in this directory)')
w('='*78)

def baseline_relative_scale(X, y, groups):
    """Uses the leading run of label=0 windows per subject as baseline
    reference. Fixed 2026-07-19 leak fix -- see 3_anxolotl/demographics_analysis.py
    / this directory's run_wesad_wrist55_blnorm.py for the full writeup."""
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

y_np = y.values.astype(int)
groups_np = groups.values
X_bl = baseline_relative_scale(X_reduced.values.astype(float), y_np, groups_np)
X_bl_df = pd.DataFrame(X_bl, columns=reduced_cols)
w(f'X_bl shape: {X_bl.shape}. Applied once globally (safe under LOSO -- per-subject '
  f'reference only depends on that subject\'s own leading baseline run; see '
  f'run_wesad_wrist55_blnorm.py STEP 2 for the full argument).')
w()

# ===========================================================================
# STEP 2 -- Ensemble(ET+RF+HGB), copied VERBATIM from
# 3_anxolotl/friedman_nemenyi_anxolotl.py line 70-74 (== recompute_macro_f1.py
# line 173-176) -- the exact object behind Table 1.11's Anxolotl row.
# ===========================================================================
w('='*78)
w('STEP 2 -- ENSEMBLE DEFINITION (copied verbatim, not reimplemented)')
w('='*78)
w('Source: 3_anxolotl/friedman_nemenyi_anxolotl.py:70-74 '
  '(identical copy at 3_anxolotl/recompute_macro_f1.py:173-176)')
w('  VotingClassifier(voting=\'hard\', estimators=[')
w('      (\'et\',  ExtraTreesClassifier(n_estimators=300, class_weight=\'balanced\', random_state=42, n_jobs=-1)),')
w('      (\'rf\',  RandomForestClassifier(n_estimators=300, class_weight=\'balanced\', random_state=42, n_jobs=-1)),')
w('      (\'hgb\', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),')
w('  ])')
w()
w('*** FLAG: task description said "soft-voting"; the repo\'s actual canonical definition '
  'is voting=\'hard\'. Using the repo\'s real definition for comparability with Table 1.11, '
  'per instructions to reuse rather than reimplement. See this file\'s module docstring. ***')
w()
w('*** FLAG: hard voting has no predict_proba in sklearn -> no canonical AUROC is possible '
  'for this ensemble. results_master.csv already lists AUROC=MISSING for the Anxolotl '
  'Ensemble(ET+RF+HGB) row (source: friedman_nemenyi_anxolotl_wrist_loso.txt) for the same '
  'reason -- this is not a new gap. A supplementary, NON-CANONICAL AUROC (soft-averaging the '
  'three already-fitted sub-estimators\' probabilities) is computed below and reported '
  'separately, clearly labeled. ***')
w()

def make_ensemble():
    return VotingClassifier(voting='hard', estimators=[
        ('et',  ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf',  RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ])

# ===========================================================================
# STEP 3 -- LOSO-CV, full dataset (no subsample -- the SVM-only subsample
# flag from the standalone run does not apply to any of ET/RF/HGB)
# ===========================================================================
w('='*78)
w('STEP 3 -- LOSO-CV, WESAD (15 subjects), full dataset, no subsample')
w('='*78)

logo = LeaveOneGroupOut()
fold_f1s, fold_ba = [], []
all_true, all_pred = [], []
soft_avg_probs, soft_avg_true = [], []
proba_error = None

n_folds = 0
for train_idx, test_idx in logo.split(X_bl_df, y, groups):
    n_folds += 1
    X_train, X_test = X_bl_df.iloc[train_idx], X_bl_df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    imputer = SimpleImputer(strategy='mean')
    X_train_p = imputer.fit_transform(X_train)
    X_test_p = imputer.transform(X_test)

    model = make_ensemble()
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)

    fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    all_true.extend(y_test)
    all_pred.extend(y_pred)

    # supplementary, non-canonical soft-averaged probability (see FLAG above)
    try:
        probs = np.mean([est.predict_proba(X_test_p)[:, 1]
                          for est in model.named_estimators_.values()], axis=0)
        soft_avg_probs.extend(probs)
        soft_avg_true.extend(y_test)
    except Exception as e:
        proba_error = f'{type(e).__name__}: {e}'

all_true = np.array(all_true)
all_pred = np.array(all_pred)

macro_f1_mean = float(np.mean(fold_f1s))
macro_f1_sd = float(np.std(fold_f1s))
balanced_acc = float(balanced_accuracy_score(all_true, all_pred))
accuracy = float(accuracy_score(all_true, all_pred))

w(f'Folds run: {n_folds}')
w(f'Macro F1 (mean +/- SD across subjects) : {macro_f1_mean:.4f} +/- {macro_f1_sd:.4f}')
w(f'Balanced Accuracy (pooled)             : {balanced_acc:.4f}')
w(f'Accuracy (pooled)                      : {accuracy:.4f}')

canonical_auroc_note = ("N/A -- voting='hard' has no predict_proba; matches the Anxolotl "
                         "Ensemble(ET+RF+HGB) row in results_master.csv, which also lists "
                         "AUROC=MISSING for the same structural reason.")
w(f'AUROC (canonical, hard-voting ensemble): {canonical_auroc_note}')

supp_auroc = None
if soft_avg_probs:
    supp_auroc = float(roc_auc_score(np.array(soft_avg_true), np.array(soft_avg_probs)))
    w(f'AUROC (SUPPLEMENTARY, non-canonical -- uniform average of the 3 fitted '
      f'sub-estimators\' predict_proba, NOT what Table 1.11 reports): {supp_auroc:.4f}')
elif proba_error:
    w(f'Supplementary soft-averaged AUROC could not be computed either: {proba_error}')
w()

results = {
    'model': 'Ensemble(ET+RF+HGB)',
    'source_definition': '3_anxolotl/friedman_nemenyi_anxolotl.py:70-74 (== recompute_macro_f1.py:173-176)',
    'voting': 'hard',
    'voting_mismatch_flag': "task described 'soft-voting'; repo's canonical definition is 'hard' -- used as-is for comparability with Table 1.11",
    'dataset': 'WESAD',
    'feature_set': '55 wrist-only (acc_wrist, hr_, ibi_, rmssd_, sdnn_)',
    'normalisation': 'BL-norm (2_cross_dataset/cross_dataset_blnorm.py :: baseline_relative_scale)',
    'cv': 'LOSO (LeaveOneGroupOut), 15 subjects, full dataset (no subsample)',
    'n_folds': n_folds,
    'macro_f1_mean': macro_f1_mean,
    'macro_f1_sd': macro_f1_sd,
    'balanced_accuracy': balanced_acc,
    'accuracy': accuracy,
    'auroc_canonical': None,
    'auroc_canonical_note': canonical_auroc_note,
    'auroc_supplementary_nonstandard': supp_auroc,
}
with open(os.path.join(OUT_DIR, 'ensemble_et_rf_hgb_blnorm.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ===========================================================================
# STEP 4 -- one-line Table 1.7 draft row + full explanatory text file
# ===========================================================================
with open(os.path.join(OUT_DIR, 'ensemble_et_rf_hgb_blnorm.txt'), 'w') as f:
    def fw(s=''):
        f.write(str(s) + '\n')
    fw('='*100)
    fw('Table 1.7 -- WESAD wrist-only, 55 features, BL-norm: Ensemble(ET+RF+HGB) row')
    fw('Generated: 2026-07-26')
    fw('='*100)
    fw()
    fw('Ensemble definition: copied VERBATIM from 3_anxolotl/friedman_nemenyi_anxolotl.py:70-74')
    fw('(identical copy at 3_anxolotl/recompute_macro_f1.py:173-176) -- the exact object behind')
    fw('Table 1.11\'s Anxolotl "Ensemble(ET+RF+HGB)" row. Not reimplemented.')
    fw('  VotingClassifier(voting=\'hard\', estimators=[')
    fw('      (\'et\',  ExtraTreesClassifier(n_estimators=300, class_weight=\'balanced\', random_state=42, n_jobs=-1)),')
    fw('      (\'rf\',  RandomForestClassifier(n_estimators=300, class_weight=\'balanced\', random_state=42, n_jobs=-1)),')
    fw('      (\'hgb\', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),')
    fw('  ])')
    fw()
    fw('*** FLAG 1: task description said "soft-voting"; repo\'s canonical definition is')
    fw('hard voting. Used as-is (repo definition wins) for exact comparability with Table 1.11.')
    fw('*** FLAG 2: hard voting -> no predict_proba -> no canonical AUROC. Table 1.11\'s own')
    fw('Ensemble(ET+RF+HGB) row already has AUROC=MISSING in results_master.csv for the same')
    fw('reason -- this is a pre-existing, structural gap, not something new introduced here.')
    fw()
    fw(f'Feature set   : 55 wrist-only (acc_wrist, hr_, ibi_, rmssd_, sdnn_ prefixes)')
    fw(f'Data source   : processed_data/wesad/processed_data.pkl (mtime {mtime})')
    fw(f'Normalisation : BL-norm (2_cross_dataset/cross_dataset_blnorm.py :: baseline_relative_scale)')
    fw(f'CV            : LOSO (LeaveOneGroupOut), {n_folds} subjects, full dataset, no subsample')
    fw()
    fw('-'*100)
    fw(f'{"Model":24s} {"Macro F1":>18s} {"BalAcc":>8s} {"AUROC":>8s}')
    fw('-'*60)
    auroc_display = f'{supp_auroc:.4f}*' if supp_auroc is not None else 'MISSING'
    fw(f'{"Ensemble(ET+RF+HGB)":24s} {macro_f1_mean:.4f} +/- {macro_f1_sd:.4f}  '
       f'{balanced_acc:.4f} {auroc_display}')
    fw('-'*60)
    fw('* AUROC marked with an asterisk is the SUPPLEMENTARY, NON-CANONICAL soft-averaged')
    fw('  value (see FLAG 2 above) -- NOT the same quantity as any other AUROC in this table,')
    fw('  since the canonical hard-voting ensemble has none. If Table 1.7 should show "--" or')
    fw('  "N/A" to stay strictly consistent with Table 1.11\'s own MISSING cell, drop the')
    fw('  asterisked value and use that instead.')
    fw()
    fw(f'Runtime: {(time.time()-T0)/60:.1f} min')

with open(os.path.join(OUT_DIR, 'ensemble_run_log.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')

w()
w('='*78)
w(f'DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
w(f'Wrote: ensemble_et_rf_hgb_blnorm.txt, ensemble_et_rf_hgb_blnorm.json, ensemble_run_log.txt')
w('='*78)
print(f'\nAll output under: {os.path.relpath(OUT_DIR, ROOT)}/')
