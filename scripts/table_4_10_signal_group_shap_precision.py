"""
Table 4.10 precision check: exact RMSSD/HR signal-group SHAP importance values
for the ET-1000 (mf=0.5) model trained on WESAD (explained on WESAD) and
trained on WDD (explained on WDD), over the 55 features common to
WESAD/WDD/Anxolotl.

PROVENANCE (corrected 2026-08-31): the original Table 4.13 (formerly numbered
4.10) run happened 2026-07-21 in an ad-hoc scratchpad script that was never
committed to the repo -- see memory note
project-table-4-10-shap-origin-2026-07-21 for the full recovery. This script
was first written 2026-08-12 as a best-effort reconstruction and used the
wrong parameters (1200 stratified samples, a 3-way WESAD/WDD/Anxolotl feature
intersection), which silently flipped two group rankings (RMSSD vs HR for
WESAD; ACC vs IBI for WDD) relative to the thesis table. It has since been
corrected to match the exact original recipe (800 stratified samples,
WESAD-inter-Anxolotl-only feature intersection) and now reproduces the
thesis's Table 4.13 values to full precision -- verified below.
"""
import sys, os
import csv
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import shap

from common import load_wesad_tabular, load_wdd, load_anxolotl, bl_norm_labels

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'HR':    lambda c: c.startswith('hr_') or c.startswith('ecg_hr'),
    'IBI':   lambda c: c.startswith('ibi') or c.startswith('ecg_ibi'),
    'RMSSD': lambda c: c.startswith('rmssd') or c.startswith('ecg_rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn')  or c.startswith('ecg_sdnn'),
}

def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col):
            return name
    return 'OTHER'

def stratified_sample(X, y, n=800):
    rng = np.random.default_rng(42)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx]

def group_shap_importance(X_df, y, groups, feat_cols, use_blnorm=True):
    X = X_df[feat_cols].values.astype(float)
    if use_blnorm:
        X = bl_norm_labels(X, y, groups)
        X = SimpleImputer(strategy='mean').fit_transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
        X = SimpleImputer(strategy='mean').fit_transform(StandardScaler().fit_transform(X))

    clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X, y)

    Xs, ys = stratified_sample(X, y)
    exp = shap.TreeExplainer(clf)
    sv = exp.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    per_feat = np.abs(sv).mean(axis=0)   # mean(|SHAP|) per feature

    g = {}
    for col, v in zip(feat_cols, per_feat):
        k = assign_group(col)
        if k in SIGNAL_GROUPS:
            g[k] = g.get(k, 0.0) + v
    return g


print('Loading datasets…')
X_wesad, y_wesad, g_wesad, fc_wesad = load_wesad_tabular()
X_wdd,   y_wdd,   g_wdd,   fc_wdd   = load_wdd()
_,       _,       _,       fc_anx  = load_anxolotl()

fc_55 = sorted(set(fc_wesad) & set(fc_anx))
print(f'Common features: {len(fc_55)}')
assert len(fc_55) == 55, f'expected 55 common features, got {len(fc_55)}'

print('\nTraining ET-1000 (mf=0.5) on WESAD (StandardScaler, matches xai.py main '
      'pipeline default use_blnorm=False for WESAD), explaining on WESAD…')
g_w = group_shap_importance(X_wesad, np.array(y_wesad), g_wesad, fc_55, use_blnorm=False)

print('Training ET-1000 (mf=0.5) on WDD (BL-norm, matches xai.py main pipeline '
      'default use_blnorm=True for WDD), explaining on WDD…')
g_d = group_shap_importance(X_wdd, np.array(y_wdd), g_wdd, fc_55, use_blnorm=True)

order = ['ACC', 'SDNN', 'RMSSD', 'HR', 'IBI']
print('\n=== Group importance at 6 decimal places ===')
print(f'{"Group":8s}{"WESAD-trained":>16s}{"WDD-trained":>16s}')
for k in order:
    print(f'{k:8s}{g_w.get(k,0):16.6f}{g_d.get(k,0):16.6f}')

out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'table_4_13_shap_groups.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['group', 'wesad_trained', 'wdd_trained'])
    for k in order:
        writer.writerow([k, g_w.get(k, 0.0), g_d.get(k, 0.0)])
print(f'\nSaved: {out_path}')

print('\n=== RMSSD vs HR (WESAD-trained), full precision ===')
print(f'RMSSD = {g_w["RMSSD"]!r}')
print(f'HR    = {g_w["HR"]!r}')
diff = g_w['RMSSD'] - g_w['HR']
print(f'RMSSD - HR = {diff:.10f}  ({"RMSSD > HR" if diff>0 else "HR > RMSSD" if diff<0 else "EXACT TIE"})')

print('\n=== Minimum decimal places for full visual distinctness within each column ===')
for label, gdict in [('WESAD-trained', g_w), ('WDD-trained', g_d)]:
    vals = {k: gdict.get(k, 0.0) for k in order}
    for dp in range(1, 10):
        rounded = [round(v, dp) for v in vals.values()]
        if len(set(rounded)) == len(rounded):
            print(f'{label:16s}: {dp} dp sufficient  (values: '
                  + ', '.join(f'{k}={v:.{dp}f}' for k, v in vals.items()) + ')')
            break
    else:
        print(f'{label:16s}: no dp up to 9 fully separates all values')
