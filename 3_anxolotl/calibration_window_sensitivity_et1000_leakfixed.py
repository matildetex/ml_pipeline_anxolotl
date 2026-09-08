"""
Calibration-window sensitivity -- canonical ET-1000 configuration.

WHAT IS ACTUALLY VARIABLE HERE (read before interpreting the numbers):

The leak-fixed baseline_relative_scale (3_anxolotl/demographics_analysis.py,
fixed 2026-07-19) does NOT take a fixed N_BL parameter. For each subject it
uses the subject's own LEADING RUN of label==0 windows -- i.e. it walks the
subject's windows in chronological order and stops at the first non-baseline
label. That run length is determined entirely by the protocol/labelling, not
by a hyperparameter. At the canonical 60s window / 30s stride configuration,
this leading run is EXACTLY 9 windows for all 37 subjects (verified directly
from df_all_1hz_cache.pkl before writing this script -- min=max=median=9).
So there is no "N_BL" left to sweep in the old sense (there was never a
subject where a fixed count of 4, 6 or 8 was clamping anything different from
another subject's 4, 6 or 8 -- the old ablation's between-subject variation in
available baseline windows doesn't exist in the leak-fixed, chronologically
capped-at-9 setting).

What IS still a meaningful, current-pipeline-faithful thing to vary: how much
of that same 9-window leading run is actually used to compute the reference
mean/std, while still never looking past the true leading run (i.e. the
leak-fix's core safety property -- never touch a window past the point the
subject's own baseline period ends -- is preserved for every cap tested).
This answers "how much calibration data does BL-norm actually need", using
the current pipeline's own reference-selection rule, not the old blind
"first N chronological windows regardless of label" rule.

Swept values: cap in {2, 4, 6, 8, 9 (= uncapped = canonical)}.
cap=9 is mathematically identical to the canonical, uncapped
baseline_relative_scale (since no subject has more than 9 baseline windows
at 60s/30s) -- it exists here as an in-run consistency check against
results/anxolotl_wrist_loso_per_subject.csv (ET-1000 column, mean=0.9169).

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed)
Window/stride : 60s / 30s (canonical)
Features      : 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) x 11 stats = 55

Replaces (does NOT overwrite):
  results/nbl_stride_marker_ablations_lgbm_loso.txt (Ablation 1) -- LGBM-300,
  fixed first-N-chronological-windows N_BL design, phase3 included, pre-leak-fix.
"""
import os
from functools import partial
import numpy as np
import pandas as pd

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, run_loso_et1000, baseline_relative_scale_capped
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'calibration_window_sensitivity_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'calibration_window_sensitivity_et1000_leakfixed_per_subject.csv')

WINDOW_SEC, STRIDE_SEC = 60, 30
CAPS = [2, 4, 6, 8, 9]

df_all = load_cache()
X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)

# Confirm the "natural leading run = 9 for everyone" claim made in the docstring.
leading_run_lengths = []
for subj in subjects:
    idx = np.where(groups == subj)[0]
    n_bl = 0
    for v in y[idx]:
        if v == 0:
            n_bl += 1
        else:
            break
    leading_run_lengths.append(n_bl)
leading_run_lengths = np.array(leading_run_lengths)

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('CALIBRATION-WINDOW SENSITIVITY -- canonical ET-1000 configuration (60s/30s, phase3 excluded)')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale_capped -- same leak-fixed reference rule as')
w('                demographics_analysis.py (leading run of label=0 windows, never past it),')
w('                but capped to use only the first `cap` windows of that leading run.')
w('CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded. Windows: '
  f'{n_after} (of {n_before} before exclusion).')
w('Features      : 55 (5 wrist channels x 11 stats).')
w()
w(f'Natural (uncapped) leading-run length per subject at 60s/30s: '
  f'min={leading_run_lengths.min()}  median={np.median(leading_run_lengths):.0f}  '
  f'max={leading_run_lengths.max()}  (identical for all 37 subjects: '
  f'{"yes" if leading_run_lengths.min()==leading_run_lengths.max() else "no"})')
w('--> cap=9 is therefore mathematically identical to the canonical, uncapped BL-norm.')
w()
w('WHAT WAS VARIED: see script docstring (3_anxolotl/calibration_window_sensitivity_et1000_leakfixed.py)')
w('for the full reasoning. In short: the leak-fixed pipeline has no fixed N_BL hyperparameter to')
w('sweep (it adaptively uses the whole leading run, which is exactly 9 windows for every subject at')
w('60s/30s) -- so instead we cap how many of those 9 leading windows are used as the BL-norm')
w('reference, never touching data past the true leading run for any cap value.')
w()
w('Replaces      : results/nbl_stride_marker_ablations_lgbm_loso.txt Ablation 1 (LGBM-300,')
w('                blind first-N-chronological-windows design, phase3 included, pre-leak-fix)')
w('                -- kept on disk, not overwritten.')
w()
w(f'{"Cap":>5}  {"N_win":>6}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)

per_subject_cols = {}

for cap in CAPS:
    norm_fn = partial(baseline_relative_scale_capped, cap=cap)
    fold_f1, auroc, _, _ = run_loso_et1000(X, y, groups, subjects, norm_fn)
    vals = np.array([fold_f1[s] for s in subjects])
    mean_f1, sd_f1 = vals.mean(), vals.std()

    col_name = f'cap={cap}' + (' (=uncapped/canonical)' if cap == 9 else '')
    per_subject_cols[col_name] = pd.Series(fold_f1)

    w(f'{cap:>5}  {n_after:>6}  {mean_f1:>8.4f}  {sd_f1:>7.4f}  {auroc:>7.4f}'
      + ('   <- uncapped, should match canonical ET-1000 (0.9169)' if cap == 9 else ''))

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

df_out = pd.DataFrame(per_subject_cols)
df_out.to_csv(OUT_CSV)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
