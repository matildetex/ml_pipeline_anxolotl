"""
Stride / overlap sensitivity -- canonical ET-1000 configuration.
Fixed window = 60s. Compares stride=30s (50% overlap, the canonical stride)
vs stride=60s (no overlap).

Replaces (does NOT overwrite):
  results/nbl_stride_marker_ablations_lgbm_loso.txt (Ablation 2) -- LGBM-300,
  phase3 included, pre-leak-fix.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale, LEAK-FIXED 2026-07-19
                (3_anxolotl/demographics_analysis.py) -- leading run of
                label=0 windows per subject.
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) at both strides
Features      : 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) x 11 stats = 55

Note: the stride=30s row here is the SAME experimental configuration as
the 60s row of window_size_ablation_et1000_leakfixed.py (60s window /
30s stride is the canonical config) -- both scripts run it independently
end-to-end (same cache, same seed), so the two files should agree; this
is a built-in consistency check, not a second, different measurement.
"""
import os
import numpy as np
import pandas as pd

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, run_loso_et1000, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'stride_ablation_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'stride_ablation_et1000_leakfixed_per_subject.csv')

WINDOW_SEC = 60
CONFIGS = [('stride=30s (50% overlap, canonical)', 30),
           ('stride=60s (no overlap)', 60)]

df_all = load_cache()

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('STRIDE / OVERLAP SENSITIVITY -- canonical ET-1000 configuration (window fixed at 60s)')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py),')
w('                leading run of label=0 windows per subject (adaptive, not fixed-count).')
w('CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded at both strides.')
w('Features      : 55 (5 wrist channels x 11 stats). Window fixed at 60s.')
w('Replaces      : results/nbl_stride_marker_ablations_lgbm_loso.txt Ablation 2 (LGBM-300,')
w('                phase3 included, pre-leak-fix -- kept on disk, not overwritten).')
w()
w(f'{"Config":<38}  {"N_win":>6}  {"N_BL":>6}  {"N_ST":>6}  {"Win/subj":>8}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)

per_subject_cols = {}

for label, stride_sec in CONFIGS:
    X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
        df_all, window_sec=WINDOW_SEC, stride_sec=stride_sec, exclude_phase3=True)
    n_bl = int((y == 0).sum())
    n_st = int((y == 1).sum())
    win_per_subj = n_after / len(subjects)

    fold_f1, auroc, _, _ = run_loso_et1000(X, y, groups, subjects, baseline_relative_scale)
    vals = np.array([fold_f1[s] for s in subjects])
    mean_f1, sd_f1 = vals.mean(), vals.std()

    per_subject_cols[label] = pd.Series(fold_f1)

    w(f'{label:<38}  {n_after:>6}  {n_bl:>6}  {n_st:>6}  {win_per_subj:>8.1f}  '
      f'{mean_f1:>8.4f}  {sd_f1:>7.4f}  {auroc:>7.4f}')
    print(f'    [detail] stride={stride_sec}s: {n_before} windows before phase3 exclusion -> {n_after} after')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

df_out = pd.DataFrame(per_subject_cols)
df_out.to_csv(OUT_CSV)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
