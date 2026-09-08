"""
Window-size sensitivity, extended to 5 points -- canonical ET-1000 configuration.

Adds a 40s window (20s stride, 50% overlap) to the original 4-point sweep
(30/60/90/120s) in results/window_size_ablation_et1000_leakfixed.txt, to
sample the curve where it rises most steeply (between 30 and 60s). Saved
to a NEW file -- results/window_size_ablation_et1000_leakfixed.txt is left
untouched.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
                -- identical factory to 3_anxolotl/friedman_nemenyi_anxolotl.py
Normalisation : baseline_relative_scale, LEAK-FIXED 2026-07-19
                (3_anxolotl/demographics_analysis.py) -- leading run of
                label=0 windows per subject.
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) at every window size
Features      : 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) x 11 stats = 55
Overlap       : 50% at every window size (stride = window / 2)
"""
import os
import numpy as np
import pandas as pd

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, run_loso_et1000, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'window_size_ablation_et1000_leakfixed_5point.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'window_size_ablation_et1000_leakfixed_5point_per_subject.csv')

CONFIGS = [(30, 15), (40, 20), (60, 30), (90, 45), (120, 60)]

df_all = load_cache()

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*96)
w('WINDOW-SIZE SENSITIVITY (5-point, adds 40s) -- canonical ET-1000 configuration')
w('='*96)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py),')
w('                leading run of label=0 windows per subject (adaptive, not fixed-count).')
w('CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded at every window size.')
w('Features      : 55 (5 wrist channels x 11 stats). Overlap: 50% at every window size.')
w('Extends       : results/window_size_ablation_et1000_leakfixed.txt (30/60/90/120s) with a new')
w('                40s point -- that file is NOT overwritten; this is a separate, complete table.')
w()
w(f'{"Win (s)":>7}  {"Stride":>6}  {"N_win":>6}  {"N_BL":>6}  {"N_ST":>6}  {"Win/subj":>8}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*96)

per_subject_cols = {}
summary_rows = []

for win_sec, stride_sec in CONFIGS:
    X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
        df_all, window_sec=win_sec, stride_sec=stride_sec, exclude_phase3=True)
    n_bl = int((y == 0).sum())
    n_st = int((y == 1).sum())
    win_per_subj = n_after / len(subjects)

    fold_f1, auroc, _, _ = run_loso_et1000(X, y, groups, subjects, baseline_relative_scale)
    vals = np.array([fold_f1[s] for s in subjects])
    mean_f1, sd_f1 = vals.mean(), vals.std()

    col_name = f'{win_sec}s'
    per_subject_cols[col_name] = pd.Series(fold_f1)
    summary_rows.append((win_sec, stride_sec, n_before, n_after, mean_f1))

    w(f'{win_sec:>7}  {stride_sec:>6}  {n_after:>6}  {n_bl:>6}  {n_st:>6}  {win_per_subj:>8.1f}  '
      f'{mean_f1:>8.4f}  {sd_f1:>7.4f}  {auroc:>7.4f}')
    print(f'    [detail] window={win_sec}s: {n_before} windows before phase3 exclusion -> {n_after} after')

w()
best = max(summary_rows, key=lambda r: r[4])
w(f'Best window size: {best[0]}s (stride {best[1]}s)  MacroF1={best[4]:.4f}')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

df_out = pd.DataFrame(per_subject_cols)
df_out.to_csv(OUT_CSV)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
