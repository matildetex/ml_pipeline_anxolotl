"""
TreeSHAP feature importance for the canonical Anxolotl configuration:
  Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                  class_weight='balanced', random_state=42, n_jobs=-1)
  Normalisation : baseline_relative_scale, leak-fixed 2026-07-19
                  (demographics_analysis.py) -- leading run of label==0
                  windows per subject, not a fixed-count reference.
  Window/stride : 60s / 30s, boundary-fixed labelling
  Phase 3       : excluded (stress_phase == 3 removed) -- 686 windows
  Features      : 55 wrist-only (acc_wrist, hr, ibi, rmssd, sdnn x 11 stats)
  Subjects      : 37

Distinct from figures/shap/Anxolotl_55all_* (2026-07-17), which used the
699-window pre-phase3-exclusion build and predates the BL-norm leak fix
being exercised on this config -- not comparable to the canonical numbers
used everywhere else in 3_anxolotl/*_leakfixed.py.

The model is a SINGLE ET-1000 fit on all 686 windows pooled across all 37
subjects (not a LOSO fold model -- SHAP needs one fixed estimator). TreeSHAP
is exact and run on the FULL 686-row matrix (no subsampling needed at this
size, unlike the 1200-row stratified sample used for the larger WESAD/WDD
datasets in scripts/xai.py).

Output: figures/shap/Anxolotl_canonical686_ET1000_{bar,groups,beeswarm}.png
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale  # leak-fixed version

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'figures', 'shap')
CACHE_PATH = os.path.join(ROOT, 'results', 'df_all_1hz_cache.pkl')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})

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

def save(fname):
    plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()
    print(f'  -> {os.path.basename(fname)}')

# ── Build canonical 686-window dataset ────────────────────────────────────────

with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)

df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=60, stride_sec=30)
n_before = len(df_feat)
df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
n_after = len(df_feat)

X_raw = df_feat[feat_cols].values.astype(float)
y = df_feat['label'].values.astype(int)
groups = df_feat['subject'].values
subjects = np.unique(groups)

assert len(feat_cols) == 55, f'expected 55 features, got {len(feat_cols)}'
assert n_after == 686, f'expected 686 windows after phase3 exclusion, got {n_after}'
assert len(subjects) == 37, f'expected 37 subjects, got {len(subjects)}'

print(f'Canonical config: 60s/30s stride, boundary-fixed, phase3 excluded.')
print(f'Windows: {n_before} -> {n_after} ({n_before - n_after} phase3 removed)')
print(f'Subjects: {len(subjects)}, features: {len(feat_cols)}')

# ── Leak-fixed BL-norm + median imputation ────────────────────────────────────

X_bl = baseline_relative_scale(X_raw, y, groups)
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X_bl)

# ── Fit ET-1000 on the full pooled dataset ────────────────────────────────────

clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                            class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X, y)

# ── TreeSHAP, exact, on the full 686-row matrix ───────────────────────────────

print('Computing TreeSHAP (exact, full 686-row matrix)…')
explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
imp_vals = np.abs(sv).mean(axis=0)

TAG = 'Anxolotl_canonical686_ET1000'
p = os.path.join(FIG_DIR, TAG)

# per-feature bar chart (top 20)
order = np.argsort(imp_vals)[::-1][:20]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(np.array(feat_cols)[order][::-1], imp_vals[order][::-1],
        color='steelblue', alpha=0.85)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title(f'Feature Importance — {TAG}')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); save(f'{p}_bar.png')

# signal-group summary
g = {}
for col, v in zip(feat_cols, imp_vals):
    k = assign_group(col); g[k] = g.get(k, 0) + v
g_order = sorted(g, key=g.get, reverse=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(g_order, [g[k] for k in g_order],
       color=plt.cm.tab10(np.linspace(0, 1, len(g_order))), alpha=0.85)
ax.set_ylabel('Summed mean |SHAP value|')
ax.set_title(f'Signal Group Importance — {TAG}')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); save(f'{p}_groups.png')

# beeswarm (directionality)
expl = shap.Explanation(values=sv, data=X, feature_names=list(feat_cols))
plt.figure(figsize=(10, 7))
shap.plots.beeswarm(expl, max_display=20, show=False)
plt.title(f'SHAP Beeswarm — {TAG}', pad=10)
plt.tight_layout(); save(f'{p}_beeswarm.png')

# ── Print numeric results (source of truth for report) ────────────────────────

print('\n=== Signal-group importance (summed mean |SHAP|) ===')
for k in g_order:
    print(f'  {k:8s} {g[k]:.6f}')

print('\n=== Top 20 features (mean |SHAP|) ===')
for rank, i in enumerate(order, 1):
    print(f'  {rank:2d}. {feat_cols[i]:20s} {imp_vals[i]:.6f}')

out_txt = os.path.join(ROOT, 'results', 'shap_canonical_et1000_leakfixed.txt')
with open(out_txt, 'w') as fh:
    fh.write(f'Canonical Anxolotl SHAP -- ET-1000, leak-fixed BL-norm, phase3 excluded\n')
    fh.write(f'Windows: {n_after} (from {n_before}, {n_before-n_after} phase3 removed), '
             f'subjects: {len(subjects)}, features: {len(feat_cols)}\n')
    fh.write(f'Model fit on all {n_after} windows pooled (not per-fold); '
             f'TreeSHAP exact on the full {n_after}-row matrix.\n\n')
    fh.write('Signal-group importance (summed mean |SHAP|):\n')
    for k in g_order:
        fh.write(f'  {k:8s} {g[k]:.6f}\n')
    fh.write('\nTop 20 features (mean |SHAP|):\n')
    for rank, i in enumerate(order, 1):
        fh.write(f'  {rank:2d}. {feat_cols[i]:20s} {imp_vals[i]:.6f}\n')
print(f'\nWrote {out_txt}')
print(f'Figures in {FIG_DIR} (prefix {TAG})')
