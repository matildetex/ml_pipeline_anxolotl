"""
Overlap sweep -- canonical ET-1000 configuration. Window fixed at 60s,
stride varied to give 0% / 25% / 50% / 75% overlap.

Replaces (does NOT overwrite) the coarser 2-point stride comparison in
results/stride_ablation_et1000_leakfixed.txt (which only tested 0% vs 50%
overlap).

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale, LEAK-FIXED 2026-07-19
                (3_anxolotl/demographics_analysis.py) -- leading run of
                label=0 windows per subject.
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) at every stride
Features      : 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) x 11 stats = 55
Window        : fixed at 60s for every row; only stride/overlap varies.
"""
import os
import numpy as np
import pandas as pd

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, run_loso_et1000, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'overlap_sweep_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'overlap_sweep_et1000_leakfixed_per_subject.csv')

WINDOW_SEC = 60
# (stride_sec, overlap_pct)
CONFIGS = [(60, 0), (45, 25), (30, 50), (15, 75)]

df_all = load_cache()

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('OVERLAP SWEEP -- canonical ET-1000 configuration (window fixed at 60s)')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py),')
w('                leading run of label=0 windows per subject (adaptive, not fixed-count).')
w('CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded at every stride.')
w('Features      : 55 (5 wrist channels x 11 stats). Window fixed at 60s; only stride/overlap varies.')
w('Replaces      : results/stride_ablation_et1000_leakfixed.txt (2-point: 0% vs 50% overlap only)')
w('                -- kept on disk, not overwritten; this is a finer 4-point sweep.')
w()
w(f'{"Overlap":>8}  {"Stride":>6}  {"N_win":>6}  {"N_BL":>6}  {"N_ST":>6}  {"Win/subj":>8}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)

per_subject_cols = {}
summary_rows = []

for stride_sec, overlap_pct in CONFIGS:
    X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
        df_all, window_sec=WINDOW_SEC, stride_sec=stride_sec, exclude_phase3=True)
    n_bl = int((y == 0).sum())
    n_st = int((y == 1).sum())
    win_per_subj = n_after / len(subjects)

    fold_f1, auroc, _, _ = run_loso_et1000(X, y, groups, subjects, baseline_relative_scale)
    vals = np.array([fold_f1[s] for s in subjects])
    mean_f1, sd_f1 = vals.mean(), vals.std()

    col_name = f'{overlap_pct}pct_overlap_stride{stride_sec}s'
    per_subject_cols[col_name] = pd.Series(fold_f1)
    summary_rows.append((overlap_pct, stride_sec, n_after, mean_f1))

    w(f'{overlap_pct:>7}%  {stride_sec:>6}  {n_after:>6}  {n_bl:>6}  {n_st:>6}  {win_per_subj:>8.1f}  '
      f'{mean_f1:>8.4f}  {sd_f1:>7.4f}  {auroc:>7.4f}')
    print(f'    [detail] stride={stride_sec}s (overlap={overlap_pct}%): '
          f'{n_before} windows before phase3 exclusion -> {n_after} after')

w()
# Flag: does higher overlap inflate performance vs the 0%-overlap (independent-window) baseline?
baseline_f1 = next(r[3] for r in summary_rows if r[0] == 0)
w('-'*100)
w('Overlap-inflation check (each row vs the 0%-overlap / independent-window baseline):')
for overlap_pct, stride_sec, n_after, mean_f1 in summary_rows:
    delta = mean_f1 - baseline_f1
    w(f'  {overlap_pct:>3}% overlap: MacroF1={mean_f1:.4f}  delta vs 0%-overlap = {delta:+.4f}')
max_overlap_row = summary_rows[-1]  # 75%
delta_75 = max_overlap_row[3] - baseline_f1
w()
if delta_75 > 0.02:
    w(f'FLAG: 75% overlap scores {delta_75:+.4f} above the 0%-overlap baseline -- large enough that')
    w('correlated, overlapping train/test windows (shared underlying signal across adjacent windows')
    w('of the same subject) may be inflating this estimate rather than reflecting a genuine gain from')
    w('more training data. Treat the 75% figure as optimistic, not a confirmed improvement.')
else:
    w(f'No inflation flag: 75% overlap is only {delta_75:+.4f} vs the 0%-overlap baseline -- not a')
    w('large jump, so no strong evidence that correlated overlapping windows are inflating this result.')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

df_out = pd.DataFrame(per_subject_cols)
df_out.to_csv(OUT_CSV)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
