"""
Section 1.2.6 -- SHAP (TreeSHAP) for WESAD wrist-only, 55-feature set, BL-norm,
ExtraTrees. Replaces Figure 1.3b (wrist-only panel only -- Figure 1.3a, the
full/137-feat panel, is untouched) and Figure 1.4. Companion to
results/section1_2_wesad_wrist55_blnorm_2026-07-26/run_wesad_wrist55_blnorm.py
(same 55 features, same BL-norm function, same ExtraTrees hyperparameters).

Model choice: ExtraTrees, not XGBoost (which scored higher F1 in today's
BL-norm table) -- per instruction, ET is the pipeline's canonical
explainability model across all three sections (1.2.6 wrist-only, 1.3.5
full-set, 1.4.8 cross-dataset), so the SHAP rankings stay comparable across
sections even though ET is not the top-F1 model on this particular table.

Everything this script writes lives under
results/section1_2_wesad_wrist55_blnorm_2026-07-26/figures/ -- no other
directory is touched, and no other SHAP (full-set, Anxolotl, cross-dataset)
is regenerated.
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')

# Pre-empt any CPU/GPU probe conflict in the SHAP explainer (same fix as the
# documented Ray/NVML issue elsewhere in this repo's AutoGluon runs) -- must be
# set before importing shap/sklearn.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
ROOT    = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
OUT_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_blnorm_2026-07-26')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

# ===========================================================================
# STEP 0 -- validate 55-column feature set (same as both reruns today)
# ===========================================================================
w('='*78)
w('STEP 0 -- FEATURE SET VALIDATION (must match both reruns today)')
w('='*78)

PKL_PATH = os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl')
with open(PKL_PATH, 'rb') as f:
    d = pickle.load(f)

X_all = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
y_all = pd.Series(np.asarray(d['y']))
groups_all = pd.Series(np.asarray(d['groups']))
feat_cols_all = list(d['feature_cols'])
w(f'File: {os.path.relpath(PKL_PATH, ROOT)} -- {X_all.shape[0]} rows x {X_all.shape[1]} cols, '
  f'{len(np.unique(groups_all))} subjects.')

WRIST_PREFIXES = ('acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_')
reduced_cols = [c for c in feat_cols_all if c.startswith(WRIST_PREFIXES)]
assert len(reduced_cols) == 55, f'STOP: expected 55 columns, got {len(reduced_cols)}: {reduced_cols}'
w(f'Reduced column selection (prefixes {WRIST_PREFIXES}): {len(reduced_cols)} columns.')
w('ASSERTION PASSED: exactly 55 columns.')
for i, c in enumerate(reduced_cols, 1):
    w(f'  {i:2d}. {c}')
w()

X_reduced = X_all[reduced_cols].reset_index(drop=True)
y = y_all.reset_index(drop=True)
groups = groups_all.reset_index(drop=True)
y_np = y.values.astype(int)
groups_np = groups.values

# ===========================================================================
# STEP 1 -- BL-norm (verbatim copy, same function used in today's blnorm rerun)
# ===========================================================================
w('='*78)
w('STEP 1 -- BL-NORM (same canonical function as today\'s rerun)')
w('='*78)
w('Source: 2_cross_dataset/cross_dataset_blnorm.py :: baseline_relative_scale '
  '(verbatim copy below -- that module executes top-level data-loading code '
  'with a relative path at import time, so it is inlined rather than imported, '
  'exactly as in run_wesad_wrist55_blnorm.py earlier today).')

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

X_bl = baseline_relative_scale(X_reduced.values.astype(float), y_np, groups_np)
w(f'X_bl shape: {X_bl.shape}. NaN count: {int(np.isnan(X_bl).sum())} '
  f'(handled by SimpleImputer(mean) below, applied globally since SHAP fits '
  f'one pooled model, matching the StandardScaler-run SHAP convention).')
w()

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
Xn = imp.fit_transform(X_bl)

# ===========================================================================
# STEP 2 -- fit ExtraTrees (pooled, same hyperparameters as both reruns today)
# ===========================================================================
w('='*78)
w('STEP 2 -- FIT ExtraTrees (pooled, hyperparameters identical to both LOSO reruns)')
w('='*78)
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, class_weight='balanced',
                            random_state=42, n_jobs=-1)
clf.fit(Xn, y_np)
w(f'ET fit on {Xn.shape[0]} rows x {Xn.shape[1]} cols (BL-norm + mean-impute, pooled, no SMOTE).')
w()

# ===========================================================================
# STEP 3 -- TreeSHAP
# ===========================================================================
w('='*78)
w('STEP 3 -- TreeSHAP')
w('='*78)

import shap

def stratified_sample(X, y, n=1200, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx], idx

Xs, ys, idx = stratified_sample(Xn, y_np, n=1200)

try:
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xs)
except Exception as e:
    w(f'  TreeExplainer default init/call FAILED ({type(e).__name__}: {e}) -- '
      f'retrying with explicit CPU-only settings (feature_perturbation="tree_path_dependent").')
    explainer = shap.TreeExplainer(clf, feature_perturbation='tree_path_dependent')
    sv = explainer.shap_values(Xs)

sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
imp_vals = np.abs(sv).mean(axis=0)
w(f'TreeSHAP computed cleanly on CPU, {Xs.shape[0]}-row stratified sample (seed=42), '
  f'no CPU/GPU explainer conflict encountered.')
w()

# ===========================================================================
# STEP 4 -- signal-group ranking + figures
# ===========================================================================
w('='*78)
w('STEP 4 -- SIGNAL-GROUP SHAP RANKING (for Section 1.2.6 prose) + FIGURES')
w('='*78)

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'HR':    lambda c: c.startswith('hr_'),
    'IBI':   lambda c: c.startswith('ibi'),
    'RMSSD': lambda c: c.startswith('rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn'),
}
def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col): return name
    return 'OTHER'

g = {}
for col, v in zip(reduced_cols, imp_vals):
    k = assign_group(col); g[k] = g.get(k, 0) + v
order = sorted(g, key=g.get, reverse=True)

w('Signal-group SHAP importance (sum of mean|SHAP|), BL-norm, ExtraTrees -- replaces Figure 1.3b:')
for rank, k in enumerate(order, 1):
    w(f'  {rank}. {k:6s} {g[k]:.4f}')
w()

top20 = np.argsort(imp_vals)[::-1][:20]
w('Top 20 individual features (mean|SHAP|):')
for rank, i in enumerate(top20, 1):
    w(f'  {rank:2d}. {reduced_cols[i]:22s} {imp_vals[i]:.6f}')
w()

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})

groups_fig = os.path.join(FIG_DIR, 'wesad_wrist55_ET_blnorm_shap_groups.png')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(order, [g[k] for k in order], color=plt.cm.tab10(np.linspace(0, 1, len(order))), alpha=0.85)
ax.set_ylabel('Summed |SHAP| Importance')
ax.set_title('Signal-Group SHAP Importance — WESAD wrist-55, BL-norm, ExtraTrees\n(replaces Figure 1.3b)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(groups_fig, dpi=200, bbox_inches='tight'); plt.close()
w(f'Saved: {os.path.relpath(groups_fig, ROOT)}')

beeswarm_fig = os.path.join(FIG_DIR, 'wesad_wrist55_ET_blnorm_shap_beeswarm.png')
try:
    expl = shap.Explanation(values=sv, data=Xs, feature_names=list(reduced_cols))
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl, max_display=20, show=False)
    plt.title('SHAP Beeswarm — WESAD wrist-55, BL-norm, ExtraTrees\n(replaces Figure 1.4)', pad=10)
    plt.tight_layout()
    plt.savefig(beeswarm_fig, dpi=200, bbox_inches='tight'); plt.close()
    w(f'Saved: {os.path.relpath(beeswarm_fig, ROOT)}')
except Exception as e:
    plt.close()
    w(f'  [beeswarm FAILED] {type(e).__name__}: {e}')
w()

w('='*78)
w(f'DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
w('Full-set (137-feat, Figure 1.3a) SHAP, Anxolotl SHAP, and cross-dataset SHAP were '
  'NOT touched by this script.')
w('='*78)

with open(os.path.join(OUT_DIR, 'shap_blnorm_log.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nAll output under: {os.path.relpath(OUT_DIR, ROOT)}/')
