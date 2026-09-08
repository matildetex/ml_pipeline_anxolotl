"""
Model comparison with ECG features (BIOPAC) added as a 4th feature set.

Feature sets:
    1. Watch only (60 features)
    2. Watch + ECG (72 features — 12 ECG/HRV from BIOPAC)
    3. Watch + Demographics (71 features)
    4. Watch + EADS-21/VAS (65 features)

ECG NaN windows (p41 missing BIOPAC, p43 poor signal) are imputed with subject median.
BL-norm applied as before. LOSO-CV.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')

from demographics_analysis import (
    load_demographics, baseline_relative_scale,
    SESSION_TO_FORM,
)
from all_models_demographics import make_all_classifiers

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

# ── Load data ─────────────────────────────────────────────────────────────────

print('Loading wearable features...')
with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    meta = pickle.load(f)
X_all      = np.array(meta['X'], dtype=float)
y_all      = np.array(meta['y'])
groups_all = np.array(meta['groups'])
feat_cols  = meta['feature_cols']

print('Loading ECG features...')
with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
    ecg_data = pickle.load(f)
X_ecg_raw = ecg_data['X_ecg'].astype(float)   # (712, 12), NaN where missing
ecg_cols  = ecg_data['ecg_cols']

# Clip physiologically implausible ECG values (signal artifacts)
# RMSSD > 300ms, rr_std > 300ms → NaN (impossible for healthy heart over 60s)
X_ecg = X_ecg_raw.copy()
X_ecg[X_ecg[:, 2] > 300, :]  = np.nan   # ecg_rmssd
X_ecg[X_ecg[:, 1] > 300, :]  = np.nan   # ecg_rr_std
X_ecg[X_ecg[:, 7] > 50, 7]   = np.nan   # ecg_lf_hf outliers

valid_ecg = (~np.isnan(X_ecg[:, 0])).sum()
print(f'ECG features: {X_ecg.shape}, valid={valid_ecg}/{len(X_ecg)}')

print('Loading demographics...')
df_demo = load_demographics('Data_tese.xlsx')
demo_cols  = ['age', 'gender', 'height', 'weight', 'bmi',
              'respiratory', 'medication', 'exercises', 'exercise_freq',
              'athlete', 'stimulant']
eads_cols_ = ['eads_depression', 'eads_anxiety', 'eads_stress', 'eads_total', 'vas_stress']

def expand(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_watch      = X_all
X_watch_ecg  = np.hstack([X_all, X_ecg])
X_watch_demo = np.hstack([X_all, expand(demo_cols)])
X_watch_eads = np.hstack([X_all, expand(eads_cols_)])

print(f'Feature sets: watch={X_watch.shape[1]}, '
      f'+ECG={X_watch_ecg.shape[1]}, '
      f'+demo={X_watch_demo.shape[1]}, +eads={X_watch_eads.shape[1]}')

# ── LOSO-CV ───────────────────────────────────────────────────────────────────

def run_loso(X, y, groups, label=''):
    clfs = make_all_classifiers()
    subjects = np.unique(groups)
    results  = {name: {'f1': [], 'auc': []} for name in clfs}
    imp = SimpleImputer(strategy='median')

    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        # BL-norm
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups[test_mask]))

        # Impute remaining NaN (ECG columns)
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)

        for name, clf in clfs.items():
            try:
                clf.fit(X_tr_n, y_tr)
                y_pred = clf.predict(X_te_n)
                y_prob = clf.predict_proba(X_te_n)[:, 1]
                auc = roc_auc_score(y_te, y_prob)
            except Exception:
                y_pred = np.zeros(len(y_te), dtype=int)
                auc = float('nan')
            results[name]['f1'].append(
                f1_score(y_te, y_pred, average='binary', zero_division=0)
            )
            results[name]['auc'].append(auc)

    summary = {name: {'f1': np.mean(v['f1']), 'auc': np.nanmean(v['auc'])}
               for name, v in results.items()}

    print(f'\n{label}')
    print(f'  {"Model":<14} {"F1":>7}  {"AUC":>7}')
    print(f'  {"-"*32}')
    for name, m in sorted(summary.items(), key=lambda x: -x[1]['f1']):
        print(f'  {name:<14} {m["f1"]:>7.3f}  {m["auc"]:>7.3f}')
    return summary

# ── Main ──────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('LOSO-CV — BL-norm — Watch + ECG comparison')
print('='*60)

res_watch = run_loso(X_watch,      y_all, groups_all, '── Watch only ──')
res_ecg   = run_loso(X_watch_ecg,  y_all, groups_all, '── Watch + ECG ──')
res_demo  = run_loso(X_watch_demo, y_all, groups_all, '── Watch + Demographics ──')
res_eads  = run_loso(X_watch_eads, y_all, groups_all, '── Watch + EADS-21/VAS ──')

# ── Results table ─────────────────────────────────────────────────────────────

clf_names = list(make_all_classifiers().keys())
rows = []
for name in clf_names:
    rows.append({
        'model':      name,
        'F1_watch':   res_watch[name]['f1'],
        'F1_ecg':     res_ecg[name]['f1'],
        'F1_demo':    res_demo[name]['f1'],
        'F1_eads':    res_eads[name]['f1'],
        'dF1_ecg':    res_ecg[name]['f1']  - res_watch[name]['f1'],
        'dF1_demo':   res_demo[name]['f1'] - res_watch[name]['f1'],
        'dF1_eads':   res_eads[name]['f1'] - res_watch[name]['f1'],
    })
df_res = pd.DataFrame(rows).sort_values('F1_watch', ascending=False)
df_res.to_csv('results/all_models_ecg.csv', index=False)

print('\n' + '='*60)
print('FINAL TABLE — sorted by Watch-only F1')
print('='*60)
print(f'{"Model":<14} {"F1 Watch":>9} {"F1 +ECG":>9} {"F1 +Demo":>9} {"F1 +EADS":>9}'
      f'  {"Δ ECG":>7} {"Δ Demo":>7} {"Δ EADS":>7}')
print('-'*72)
for _, r in df_res.iterrows():
    print(f'{r["model"]:<14} {r["F1_watch"]:>9.3f} {r["F1_ecg"]:>9.3f}'
          f' {r["F1_demo"]:>9.3f} {r["F1_eads"]:>9.3f}'
          f'  {r["dF1_ecg"]:>+7.3f} {r["dF1_demo"]:>+7.3f} {r["dF1_eads"]:>+7.3f}')

# ── Figure: 4-set comparison ──────────────────────────────────────────────────

C_W = '#4C72B0'; C_E = '#C44E52'; C_D = '#55A868'; C_A = '#DD8452'
models_sorted = df_res['model'].tolist()
x = np.arange(len(models_sorted))
w = 0.20

fig, ax = plt.subplots(figsize=(20, 6))
f1_w = [res_watch[m]['f1'] for m in models_sorted]
f1_e = [res_ecg[m]['f1']   for m in models_sorted]
f1_d = [res_demo[m]['f1']  for m in models_sorted]
f1_a = [res_eads[m]['f1']  for m in models_sorted]
ax.bar(x - 1.5*w, f1_w, w, label='Watch only',          color=C_W, alpha=0.88)
ax.bar(x - 0.5*w, f1_e, w, label='Watch + ECG (BIOPAC)',color=C_E, alpha=0.88)
ax.bar(x + 0.5*w, f1_d, w, label='Watch + Demographics', color=C_D, alpha=0.88)
ax.bar(x + 1.5*w, f1_a, w, label='Watch + EADS-21/VAS', color=C_A, alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(models_sorted, rotation=30, ha='right')
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0, 1.15)
ax.set_title('All models — 4 feature sets (Anxolotl LOSO-CV, BL-norm)\n'
             'ECG features from BIOPAC (RR mean/std, RMSSD, pNN50, LF/HF, respiration)')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig14_ecg_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print('\nSaved fig14_ecg_comparison.png')

# ── Delta heatmap (gain from adding any modality) ─────────────────────────────

delta_data = pd.DataFrame({
    'Δ ECG':      {m: res_ecg[m]['f1']  - res_watch[m]['f1'] for m in models_sorted},
    'Δ Demo':     {m: res_demo[m]['f1'] - res_watch[m]['f1'] for m in models_sorted},
    'Δ EADS/VAS': {m: res_eads[m]['f1'] - res_watch[m]['f1'] for m in models_sorted},
})
fig, ax = plt.subplots(figsize=(7, 9))
sns.heatmap(delta_data, annot=True, fmt='+.3f', cmap='RdBu_r',
            center=0, vmin=-0.06, vmax=0.06, ax=ax, linewidths=0.3,
            cbar_kws={'label': 'ΔF1 vs watch-only'})
ax.set_title('ΔF1 from adding features\n(ECG vs Demographics vs EADS)', pad=10)
plt.tight_layout()
plt.savefig('figures/fig15_delta_ecg_demo.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig15_delta_ecg_demo.png')

print(f'\nSaved results/all_models_ecg.csv')
print(f'\nBest model (Watch only): {df_res.iloc[0]["model"]} F1={df_res.iloc[0]["F1_watch"]:.3f}')
print(f'Best model (+ECG):       {df_res.sort_values("F1_ecg",ascending=False).iloc[0]["model"]} F1={df_res["F1_ecg"].max():.3f}')
print(f'Mean ΔF1 from +ECG:      {df_res["dF1_ecg"].mean():+.4f}')
