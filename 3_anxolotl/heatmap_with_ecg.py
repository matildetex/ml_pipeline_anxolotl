"""
Heatmap: all models × feature sets, WITH and WITHOUT p20.
Columns: Watch only | +Demographics | +EADS-21/VAS | +ECG (last)
Two figures: all subjects, and p20 excluded.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale, SESSION_TO_FORM
from all_models_demographics import make_all_classifiers

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

OUTLIER = 'p20'

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading wearable features...')
with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    meta = pickle.load(f)
X_all      = np.array(meta['X'], dtype=float)
y_all      = np.array(meta['y'])
groups_all = np.array(meta['groups'])

print('Loading ECG features...')
with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
    ecg_data = pickle.load(f)
X_ecg_raw = ecg_data['X_ecg'].astype(float)
ecg_cols  = ecg_data['ecg_cols']
X_ecg = X_ecg_raw.copy()
rmssd_idx = ecg_cols.index('ecg_rmssd')   if 'ecg_rmssd'  in ecg_cols else 2
std_idx   = ecg_cols.index('ecg_rr_std')  if 'ecg_rr_std' in ecg_cols else 1
lfhf_idx  = ecg_cols.index('ecg_lf_hf')  if 'ecg_lf_hf'  in ecg_cols else 7
X_ecg[X_ecg[:, rmssd_idx] > 300, :] = np.nan
X_ecg[X_ecg[:, std_idx]   > 300, :] = np.nan
X_ecg[X_ecg[:, lfhf_idx]  > 50, lfhf_idx] = np.nan

print('Loading demographics...')
df_demo = load_demographics('Data_tese.xlsx')
demo_cols  = ['age', 'gender', 'height', 'weight', 'bmi',
              'respiratory', 'medication', 'exercises', 'exercise_freq',
              'athlete', 'stimulant']
eads_cols_ = ['eads_depression', 'eads_anxiety', 'eads_stress', 'eads_total', 'vas_stress']

def expand(cols, groups):
    rows = []
    for g in groups:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)


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

        X_tr_n = baseline_relative_scale(X_tr, y_tr, groups[train_mask])
        X_te_n = baseline_relative_scale(X_te, y_te,  groups[test_mask])

        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)

        for name, clf in clfs.items():
            try:
                clf.fit(X_tr_n, y_tr)
                pred = clf.predict(X_te_n)
                f1   = f1_score(y_te, pred, average='macro', zero_division=0)
                results[name]['f1'].append(f1)
            except Exception as e:
                results[name]['f1'].append(float('nan'))

    return {name: {'f1': float(np.nanmean(v['f1']))} for name, v in results.items()}


def make_heatmap(mask, tag, title_suffix):
    Xw = X_all[mask]
    ye = y_all[mask]
    gs = groups_all[mask]

    Xd = np.hstack([Xw, expand(demo_cols, gs)])
    Xe = np.hstack([Xw, expand(eads_cols_, gs)])
    Xc = np.hstack([Xw, X_ecg[mask]])

    print(f'\n{"="*60}\n  {tag} — Watch only\n{"="*60}')
    res_w = run_loso(Xw, ye, gs, 'watch')
    print(f'  {"="*60}\n  {tag} — +Demographics\n{"="*60}')
    res_d = run_loso(Xd, ye, gs, '+demo')
    print(f'  {"="*60}\n  {tag} — +EADS-21/VAS\n{"="*60}')
    res_ea = run_loso(Xe, ye, gs, '+eads')
    print(f'  {"="*60}\n  {tag} — +ECG\n{"="*60}')
    res_c = run_loso(Xc, ye, gs, '+ecg')

    models_sorted = sorted(res_w.keys(),
                           key=lambda m: res_w[m]['f1'], reverse=True)

    heat_data = pd.DataFrame({
        'Watch only':           {m: res_w[m]['f1']  for m in models_sorted},
        'Watch + Demographics': {m: res_d[m]['f1']  for m in models_sorted},
        'Watch + EADS-21/VAS':  {m: res_ea[m]['f1'] for m in models_sorted},
        'Watch + ECG':          {m: res_c[m]['f1']  for m in models_sorted},
    })

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Macro F1'})
    ax.set_title(f'F1 heatmap — all models × feature sets\n'
                 f'(Anxolotl LOSO-CV, BL-norm{title_suffix})', pad=10)
    ax.set_xlabel('')
    plt.tight_layout()
    fname = f'figures/fig12_heatmap_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')

    # delta heatmap
    delta_data = pd.DataFrame({
        'Δ Demographics': {m: res_d[m]['f1']  - res_w[m]['f1'] for m in models_sorted},
        'Δ EADS-21/VAS':  {m: res_ea[m]['f1'] - res_w[m]['f1'] for m in models_sorted},
        'Δ ECG':          {m: res_c[m]['f1']  - res_w[m]['f1'] for m in models_sorted},
    })
    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(delta_data, annot=True, fmt='+.3f', cmap='RdBu_r',
                center=0, vmin=-0.06, vmax=0.06, ax=ax, linewidths=0.3,
                cbar_kws={'label': 'ΔF1 vs watch-only'})
    ax.set_title(f'ΔF1 from adding features\n(Anxolotl LOSO-CV{title_suffix})', pad=10)
    plt.tight_layout()
    fname_d = f'figures/fig13_delta_{tag}.png'
    plt.savefig(fname_d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname_d}')

    return heat_data, delta_data


if __name__ == '__main__':
    mask_all    = np.ones(len(y_all), dtype=bool)
    mask_no_p20 = groups_all != OUTLIER

    print('\n' + '='*60)
    print(f'  ALL SUBJECTS (n={mask_all.sum()} windows)')
    print('='*60)
    make_heatmap(mask_all, 'all_subjects', '')

    print('\n' + '='*60)
    print(f'  WITHOUT {OUTLIER} (n={mask_no_p20.sum()} windows)')
    print('='*60)
    make_heatmap(mask_no_p20, f'no_{OUTLIER}', f', excl. {OUTLIER}')

    print('\nDone.')
