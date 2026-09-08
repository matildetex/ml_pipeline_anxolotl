"""
Table 4.13 — signal-group SHAP importance, ExtraTrees (ET-1000, mf=0.5),
trained on WESAD and on WDD over the 55 features common to WESAD/WDD/Anxolotl,
each model explained on its OWN dataset.

This is a fresh, from-scratch reproduction (2026-08-31) written because neither
existing candidate matched the thesis's described methodology:
  - scripts/xai.py::cross_dataset_groups() trains ONE model on WESAD only and
    explains that same model on WESAD/WDD/Anxolotl (a transfer-robustness
    check), not two independently-trained, self-explained models.
  - scripts/table_4_10_signal_group_shap_precision.py (2026-08-12) used the
    wrong parameters (1200-sample explainer, 3-way feature intersection) and
    reproduced the thesis table only approximately, with two rank inversions.

55-common-feature definition: WESAD ∩ Anxolotl feature_cols, per
archive/cross_dataset_blnorm.py:102 (path updated 2026-09-07 — this module
lived at 2_cross_dataset/ when verified below, archived as pre-leak-fix
code during the 2026-09-07 cross-dataset reorg; the column-selection logic
cited here is unaffected) (`COMMON_COLS = sorted(set(wesad
['feature_cols']) & set(anx['feature_cols']))`) — verified 2026-08-31 to be
exactly 55 columns (5 channels: acc_wrist, hr, ibi, rmssd, sdnn x 11 stats
each), and verified to be a strict subset of WDD's own feature_cols too, so
the same 55 column names can be selected from WDD's matrix directly.

Normalisation: baseline_relative_scale (leak-fixed, leading-run-of-baseline-
windows-per-subject rule), the same function used throughout the rest of the
cross-dataset chapter, via scripts/common.py::bl_norm_labels -- applied to
BOTH WESAD and WDD here, since both are only ever used as their own training/
explanation data in this script (never as a held-out evaluation target), so
there is no train/test asymmetry to preserve.

Explainer sample: TreeSHAP is run on a class-stratified sample of n=2000 rows
per dataset (random_state=42, half class 0 / half class 1, capped by class
availability) -- NOT the full dataset. An earlier version of this script used
the full dataset (no sub-sampling); on WESAD's 25807 rows that made TreeSHAP
impractically slow (killed after 54 min with no result) and was abandoned.
n=2000 is a deliberate, disclosed choice, larger than the two prior samples
this repo has used for the same kind of table (800 in the original 2026-07-21
run per project memory, 1200 in the 2026-08-12 script).

Model: ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
class_weight='balanced', random_state=42, n_jobs=-1) -- the canonical spec
used everywhere else in this repo's cross-dataset and Anxolotl-LOSO scripts.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import shap

from common import load_wesad_tabular, load_wdd, load_anxolotl, bl_norm_labels

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'HR':    lambda c: c.startswith('hr_'),
    'IBI':   lambda c: c.startswith('ibi'),
    'RMSSD': lambda c: c.startswith('rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn'),
}

def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col):
            return name
    return 'OTHER'


SHAP_SAMPLE_N = 2000  # stratified by class, per dataset — see module docstring

def stratified_sample(X, y, n=SHAP_SAMPLE_N):
    rng = np.random.default_rng(42)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx]


def train_and_explain(X_df, y, groups, feat_cols, label):
    X = X_df[feat_cols].values.astype(float)
    X = bl_norm_labels(X, y, groups)
    X = SimpleImputer(strategy='mean').fit_transform(X)

    clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X, y)

    Xs, ys = stratified_sample(X, y)
    exp = shap.TreeExplainer(clf)
    sv = exp.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    per_feat = np.abs(sv).mean(axis=0)

    g = {}
    for col, v in zip(feat_cols, per_feat):
        k = assign_group(col)
        if k in SIGNAL_GROUPS:
            g[k] = g.get(k, 0.0) + float(v)

    print(f'  {label}: fit on {X.shape[0]} rows x {X.shape[1]} cols, '
          f'SHAP on stratified sample n={Xs.shape[0]} (seed=42)')
    return g


print('Loading datasets…')
X_wesad, y_wesad, g_wesad, fc_wesad = load_wesad_tabular()
X_wdd,   y_wdd,   g_wdd,   fc_wdd   = load_wdd()
_,       _,       _,       fc_anx  = load_anxolotl()

fc_55 = sorted(set(fc_wesad) & set(fc_anx))
print(f'Common features (WESAD ∩ Anxolotl): {len(fc_55)}')
assert len(fc_55) == 55, f'expected 55 common features, got {len(fc_55)}'
assert set(fc_55) <= set(fc_wdd), 'not all 55 common columns exist in WDD'

print('\nTraining ET-1000 (mf=0.5) on WESAD, BL-norm, explaining on WESAD (full dataset)…')
g_w = train_and_explain(X_wesad, np.array(y_wesad), g_wesad, fc_55, 'WESAD')

print('Training ET-1000 (mf=0.5) on WDD, BL-norm, explaining on WDD (full dataset)…')
g_d = train_and_explain(X_wdd, np.array(y_wdd), g_wdd, fc_55, 'WDD')

rank_w = sorted(g_w, key=g_w.get, reverse=True)
rank_d = sorted(g_d, key=g_d.get, reverse=True)

print('\n=== Group importance (sum of mean|SHAP|) ===')
print(f'{"Group":8s}{"WESAD-trained":>16s}{"rank":>6s}{"WDD-trained":>16s}{"rank":>6s}')
for k in ['ACC', 'HR', 'IBI', 'RMSSD', 'SDNN']:
    print(f'{k:8s}{g_w[k]:16.6f}{rank_w.index(k)+1:6d}{g_d[k]:16.6f}{rank_d.index(k)+1:6d}')

print(f'\nWESAD ranking (highest to lowest): {rank_w}')
print(f'WDD ranking   (highest to lowest): {rank_d}')

out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'table_4_13_shap_groups.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['# shap_sample', f'stratified sample n={SHAP_SAMPLE_N} per dataset, seed=42'])
    writer.writerow(['group', 'wesad_trained', 'wesad_rank', 'wdd_trained', 'wdd_rank'])
    for k in ['ACC', 'HR', 'IBI', 'RMSSD', 'SDNN']:
        writer.writerow([k, g_w[k], rank_w.index(k) + 1, g_d[k], rank_d.index(k) + 1])
print(f'\nSaved: {out_path}')
