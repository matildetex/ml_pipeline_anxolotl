"""
Marker-confirmation sensitivity -- canonical ET-1000 configuration.

WHAT "MARKERS" MEANS: the Anxolotl protocol asks each subject to perform a
wrist-raise gesture at up to 8 points in the session, to timestamp phase
transitions (baseline start, breathing/rest segments, each stress-phase
start/end -- see results/canonical_numbers_for_thesis.txt section 3 for the
full 8-marker list). detect_markers() (3_anxolotl/anxolotl_preprocessing.py)
finds these gestures from the wrist accelerometer. Not every session yields
all 8 -- when a gesture isn't detected, that phase boundary is inferred
(from protocol timing) instead of directly observed. n_markers per subject
is this gesture-detection count; it is a raw-signal fact, independent of the
BL-norm leak fix, phase-3 exclusion, or model choice, so it is reused here
from the existing, already-verified mapping (NOT re-run from raw sessions --
that would re-derive gesture detection, which is unrelated to what changed
2026-07-19) rather than being recomputed. Source of the mapping: identical
in results/canonical_numbers_for_thesis.txt (section 3) and
results/nbl_stride_marker_ablations_lgbm_loso.txt (lines 50-54).

WHAT WAS CHANGED FROM THE OLD ABLATION: the old split was
"n_markers==8" (n=1 subject, p22) vs "n_markers<8" (n=36) -- not usable,
one group had a single subject. This re-run instead splits subjects into
n_markers>=6 (more than half the protocol's markers gesture-confirmed) vs
n_markers<=5 (majority inferred), which yields two groups of size 24 and 13.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale, LEAK-FIXED 2026-07-19
                (3_anxolotl/demographics_analysis.py)
CV            : LOSO over all 37 subjects (single run; groups are formed by
                POST-HOC splitting the per-subject fold results -- training
                set is always all-other-36-subjects, same as every other
                canonical LOSO result, exactly as the old ablation did it:
                "Training set = all 37 subjects; test subset varies").
Phase 3       : excluded (stress_phase == 3 removed)
Window/stride : 60s / 30s (canonical)
Features      : 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) x 11 stats = 55

Replaces (does NOT overwrite):
  results/nbl_stride_marker_ablations_lgbm_loso.txt (Ablation 3) -- LGBM-300,
  phase3 included, pre-leak-fix, confirmed-group n=1.
"""
import os
import numpy as np
import pandas as pd

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, run_loso_et1000, baseline_relative_scale
)

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'marker_sensitivity_et1000_leakfixed.txt')
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'marker_sensitivity_et1000_leakfixed_per_subject.csv')

WINDOW_SEC, STRIDE_SEC = 60, 30

# n_markers per subject -- reused from results/canonical_numbers_for_thesis.txt
# (section 3) / results/nbl_stride_marker_ablations_lgbm_loso.txt:50-54, both
# of which agree exactly. Raw-signal gesture-detection fact, not re-derived here.
N_MARKERS = {
    'p32': 4, 'p34': 4,
    'p15': 5, 'p19': 5, 'p20': 5, 'p21': 5, 'p25': 5, 'p28': 5, 'p37': 5,
    'p39': 5, 'p44': 5, 'p8': 5, 'p9': 5,
    'p10': 6, 'p11': 6, 'p13': 6, 'p14': 6, 'p16': 6, 'p17': 6, 'p18': 6,
    'p23': 6, 'p26': 6, 'p27': 6, 'p29': 6, 'p30': 6, 'p33': 6, 'p35': 6,
    'p38': 6, 'p40': 6, 'p41': 6, 'p42': 6, 'p43': 6, 'p7': 6,
    'p12': 7, 'p31': 7, 'p36': 7,
    'p22': 8,
}

df_all = load_cache()
X, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)

missing = set(subjects) - set(N_MARKERS.keys())
assert not missing, f'No marker count for subjects: {missing}'

fold_f1, auroc_all, y_true_all, y_prob_all = run_loso_et1000(
    X, y, groups, subjects, baseline_relative_scale)

group_hi = [s for s in subjects if N_MARKERS[s] >= 6]   # confirmed-majority
group_lo = [s for s in subjects if N_MARKERS[s] <= 5]   # inferred-majority

# rebuild pooled (y_true, y_prob) per group for group-level AUROC, aligned to
# the same subject iteration order used inside run_loso_et1000 (np.unique(subjects))
subj_order = list(subjects)
te_masks = {s: (groups == s) for s in subj_order}
# recompute per-subject probability arrays by re-running only bookkeeping:
# run_loso_et1000 already computed a single pooled array in subject order,
# so slice it back out per subject using the known per-fold sizes.
sizes = [te_masks[s].sum() for s in subj_order]
offsets = np.cumsum([0] + sizes)
y_true_by_subj = {s: y_true_all[offsets[i]:offsets[i+1]] for i, s in enumerate(subj_order)}
y_prob_by_subj = {s: y_prob_all[offsets[i]:offsets[i+1]] for i, s in enumerate(subj_order)}

from sklearn.metrics import roc_auc_score

def group_stats(group_subjs):
    vals = np.array([fold_f1[s] for s in group_subjs])
    yt = np.concatenate([y_true_by_subj[s] for s in group_subjs])
    yp = np.concatenate([y_prob_by_subj[s] for s in group_subjs])
    try:
        auroc = roc_auc_score(yt, yp)
    except ValueError:
        auroc = float('nan')
    n_testwin = len(yt)
    return vals.mean(), vals.std(), auroc, n_testwin

mean_hi, sd_hi, auroc_hi, ntw_hi = group_stats(group_hi)
mean_lo, sd_lo, auroc_lo, ntw_lo = group_stats(group_lo)

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('MARKER-CONFIRMATION SENSITIVITY -- canonical ET-1000 configuration (60s/30s, phase3 excluded)')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 (demographics_analysis.py)')
w(f'CV            : LOSO, 37 subjects. Phase 3 excluded. Windows: {n_after} (of {n_before} before exclusion).')
w('Features      : 55 (5 wrist channels x 11 stats).')
w()
w('What "markers" are and what changed in this split: see script docstring')
w('(3_anxolotl/marker_sensitivity_et1000_leakfixed.py). In short: n_markers = number of the')
w('protocol\'s 8 wrist-raise gestures gesture-detected for that subject (source: raw accelerometer,')
w('reused unchanged from results/canonical_numbers_for_thesis.txt sec.3 / the old ablation file --')
w('unaffected by the BL-norm leak fix). Old split (n_markers==8 vs <8) gave a 1-subject group; this')
w('split uses n_markers>=6 (more than half the markers confirmed) vs n_markers<=5.')
w()
w('n_markers distribution (37 subjects):')
for n in sorted(set(N_MARKERS.values())):
    ss = sorted([s for s in subjects if N_MARKERS[s] == n])
    w(f'  n_markers={n}: {len(ss)} subjects  {ss}')
w()
w(f'Group HI (n_markers>=6, majority confirmed): {len(group_hi)} subjects  {sorted(group_hi)}')
w(f'Group LO (n_markers<=5, majority inferred) : {len(group_lo)} subjects  {sorted(group_lo)}')
w()
w(f'{"Group":<45}  {"N_subj":>6}  {"N_testwin":>9}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)
w(f'{"HI (n_markers>=6)":<45}  {len(group_hi):>6}  {ntw_hi:>9}  {mean_hi:>8.4f}  {sd_hi:>7.4f}  {auroc_hi:>7.4f}')
w(f'{"LO (n_markers<=5)":<45}  {len(group_lo):>6}  {ntw_lo:>9}  {mean_lo:>8.4f}  {sd_lo:>7.4f}  {auroc_lo:>7.4f}')
w()
w(f'All 37 subjects (pooled, for reference; should match ET-1000 canonical 0.9169):')
all_vals = np.array([fold_f1[s] for s in subjects])
w(f'  MacroF1={all_vals.mean():.4f}  SD={all_vals.std():.4f}  AUROC={auroc_all:.4f}')
w()
w('Replaces      : results/nbl_stride_marker_ablations_lgbm_loso.txt Ablation 3 (LGBM-300,')
w('                phase3 included, pre-leak-fix, confirmed-group n=1) -- kept on disk, not overwritten.')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

per_subject_df = pd.DataFrame({
    'subject': subj_order,
    'n_markers': [N_MARKERS[s] for s in subj_order],
    'group': ['HI (>=6)' if N_MARKERS[s] >= 6 else 'LO (<=5)' for s in subj_order],
    'macro_f1': [fold_f1[s] for s in subj_order],
})
per_subject_df.to_csv(OUT_CSV, index=False)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV}')
