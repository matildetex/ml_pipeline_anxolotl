"""
Effect of adding ECG (BIOPAC) features -- canonical ET-1000 configuration,
Anxolotl. Corrected re-run of results/watch_ecg_vs_wesad.txt, whose
Anxolotl numbers (ET-1000-mf0.5: 0.9180 -> 0.9503) predate the BL-norm
leak fix of 2026-07-19 (that file's mtime is 2026-07-18 22:53, before the
2026-07-19 00:28 fix to demographics_analysis.py) and reported no SD/AUROC
or significance test per row.

ECG features are taken from processed_data/anxolotl/processed_data_anxolotl_ecg.pkl
(17 features per window, extracted from the BIOPAC ECG at the same 60s/30s
windowing as the canonical wrist pipeline). Row-alignment with the freshly
built canonical wrist matrix was verified before use: the ECG pkl's own
'y' and 'groups' arrays are identical, element-for-element, to the y/groups
produced by build_canonical_xy(window_sec=60, stride_sec=30,
exclude_phase3=False) in this run (699 rows each) -- so the ECG columns are
hstacked onto a FRESH canonical wrist-feature build, not onto the old
static processed_data_anxolotl.pkl, removing any doubt about whether the
old pkl's wrist features still match the current canonical build. The
same three ECG-artifact NaN-outlier rules as the original script
(watch_ecg_vs_wesad.py) are kept unchanged: ecg_rmssd>300 / ecg_rr_std>300
-> whole-row NaN, ecg_lf_hf>50 -> that column NaN.

Comparability note for the WESAD chapter: WESAD's 55-vs-72-feature ECG
comparison (results/wesad_55plus_ecg_loso.txt: 0.6902 -> 0.8900) is
labelled only "ExtraTrees + BL-norm" -- its source script no longer exists
in the repo and the exact ExtraTrees hyperparameters (n_estimators,
max_features) are not stated in the saved file, so I cannot confirm they
match ET-1000 (n_estimators=1000, max_features=0.5) exactly. Both numbers
can be placed side by side, but flag that the model configs are not
verified identical.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py)
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) -- 686 windows
Metric        : Macro F1 (mean across 37 subjects), SD, pooled out-of-fold AUROC

Statistical test: paired two-sided Wilcoxon signed-rank test on the 37
per-subject macro-F1 values, watch-only vs watch+ECG.

Does NOT overwrite results/watch_ecg_vs_wesad.txt or any other existing file.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, et1000_factory, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'ecg_effect_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'ecg_effect_et1000_leakfixed_per_subject.csv')
ECG_PKL = os.path.join(os.path.dirname(__file__), '..', 'processed_data',
                        'anxolotl', 'processed_data_anxolotl_ecg.pkl')

WINDOW_SEC, STRIDE_SEC = 60, 30

df_all = load_cache()

# Build the canonical wrist matrix WITHOUT phase3 exclusion first (699 rows),
# to verify row-for-row alignment against the ECG pkl before combining.
X55_699, y_699, groups_699, subjects_699, feat_cols, n_before_699, n_after_699 = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=False)

with open(ECG_PKL, 'rb') as f:
    ecg_data = pickle.load(f)
X_ecg_raw = ecg_data['X_ecg'].astype(float)
ecg_cols = ecg_data['ecg_cols']
ecg_y = np.array(ecg_data['y'])
ecg_groups = np.array(ecg_data['groups'])

assert X_ecg_raw.shape[0] == X55_699.shape[0], \
    f'ECG pkl row count ({X_ecg_raw.shape[0]}) does not match freshly-built canonical wrist matrix ({X55_699.shape[0]})'
assert np.array_equal(ecg_y, y_699), 'ECG pkl y does not match canonical y -- row alignment broken'
assert np.array_equal(ecg_groups, groups_699), 'ECG pkl groups does not match canonical groups -- row alignment broken'

# Same ECG-artifact cleaning as the original watch_ecg_vs_wesad.py (unchanged).
X_ecg = X_ecg_raw.copy()
X_ecg[X_ecg[:, ecg_cols.index('ecg_rmssd')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_rr_std')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_lf_hf')] > 50, ecg_cols.index('ecg_lf_hf')] = np.nan

# Now build the df_feat-derived phase3 mask fresh (must exactly match the mask
# build_canonical_xy applies internally when exclude_phase3=True), by rebuilding
# df_feat once more and reading its own stress_phase column.
import anxolotl_preprocessing as ap
df_feat, _ = ap.build_features_tabular(df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)
phase3_mask = (df_feat['stress_phase'] != 3).values
assert len(phase3_mask) == X55_699.shape[0], 'phase3 mask length does not match canonical wrist matrix row count'

X_watch = X55_699[phase3_mask]
X_ecg_f = X_ecg[phase3_mask]
y = y_699[phase3_mask]
groups = groups_699[phase3_mask]
subjects = np.unique(groups)
n_after = len(y)

X_combined = np.hstack([X_watch, X_ecg_f])


def run_loso(X_mat):
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


lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('EFFECT OF ADDING ECG (BIOPAC) -- canonical ET-1000 configuration, Anxolotl')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py)')
w(f'CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded. Windows: {n_after} '
  f'(of {len(y_699)} before exclusion).')
w('Metric        : Macro F1.')
w('Replaces      : results/watch_ecg_vs_wesad.txt (Anxolotl rows) -- LGBM/ET sweep, pre-leak-fix')
w('                (dated 2026-07-18, before the 2026-07-19 00:28 fix), no SD/AUROC/significance')
w('                reported per row -- kept on disk, not overwritten.')
w()
w(f'Row-alignment check: ECG pkl (processed_data_anxolotl_ecg.pkl) y/groups verified identical to')
w(f'freshly-built canonical wrist matrix, {X55_699.shape[0]} rows, before phase3 exclusion.')
w()

fold_f1_watch, auroc_watch = run_loso(X_watch)
fold_f1_comb, auroc_comb = run_loso(X_combined)

vals_watch = np.array([fold_f1_watch[s] for s in subjects])
vals_comb = np.array([fold_f1_comb[s] for s in subjects])

w(f'{"Config":<20}  {"N_feat":>7}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)
w(f'{"Watch-only":<20}  {X_watch.shape[1]:>7}  {vals_watch.mean():>8.4f}  {vals_watch.std():>7.4f}  '
  f'{auroc_watch:>7.4f}   <- should match canonical ET-1000 (0.9169)')
w(f'{"Watch+ECG":<20}  {X_combined.shape[1]:>7}  {vals_comb.mean():>8.4f}  {vals_comb.std():>7.4f}  '
  f'{auroc_comb:>7.4f}')
w(f'{"Delta":<20}  {"":>7}  {vals_comb.mean()-vals_watch.mean():>+8.4f}')
w()

stat, p = wilcoxon(vals_comb, vals_watch, alternative='two-sided')
sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
w('Paired Wilcoxon signed-rank test (two-sided), 37 subjects, watch+ECG vs watch-only:')
w(f'  statistic={stat:.2f}  p-value={p:.6f}  ({sig} at alpha=0.05)')
w()
w('-'*100)
w('For direct comparison with WESAD (results/wesad_55plus_ecg_loso.txt) -- WESAD chapter, not this one:')
w('  WESAD  55-feat (wrist)      : MacroF1=0.6902  SD=0.1848  AUROC=0.8536')
w('  WESAD  72-feat (wrist+ECG)  : MacroF1=0.8900  SD=0.1526  AUROC=0.9751')
w('  NOTE: WESAD file labels its model only "ExtraTrees + BL-norm"; the source script no longer')
w('  exists in the repo and exact hyperparameters (n_estimators, max_features) are not stated in')
w('  the saved file, so exact equivalence to ET-1000 (n_estimators=1000, max_features=0.5) used')
w('  here for Anxolotl is NOT confirmed -- flag this when placing the two side by side.')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

pd.DataFrame({
    'subject': subjects,
    'macro_f1_watch_only': [fold_f1_watch[s] for s in subjects],
    'macro_f1_watch_ecg': [fold_f1_comb[s] for s in subjects],
}).to_csv(OUT_CSV, index=False)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
