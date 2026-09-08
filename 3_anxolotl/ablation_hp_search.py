"""
Ablation with per-stage HP search (Watch → Demo → Calib → HP → ECG).

Grid search only over max_features (min_samples_leaf=1 fixed — optimal for ET).
Runs with unbuffered output so progress is visible immediately.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    meta = pickle.load(f)
X_watch    = np.array(meta['X'], dtype=float)
y_all      = np.array(meta['y'])
groups_all = np.array(meta['groups'])

with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
    ecg_data = pickle.load(f)
X_ecg_raw = ecg_data['X_ecg'].astype(float)
ecg_cols  = ecg_data['ecg_cols']
X_ecg = X_ecg_raw.copy()
X_ecg[X_ecg[:, ecg_cols.index('ecg_rmssd')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_rr_std')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_lf_hf')]  > 50,
      ecg_cols.index('ecg_lf_hf')] = np.nan

df_demo = load_demographics('Data_tese.xlsx')
demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
             'respiratory', 'medication', 'exercises', 'exercise_freq',
             'athlete', 'stimulant']

def expand_demo(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_demo = expand_demo(demo_cols)
X_watch_demo = np.hstack([X_watch, X_demo])
X_WIN        = np.hstack([X_watch, X_ecg, X_demo])

subjects = np.unique(groups_all)
print(f'Loaded: {len(subjects)} subjects, watch={X_watch.shape[1]} feat, '
      f'watch+demo={X_watch_demo.shape[1]} feat, full={X_WIN.shape[1]} feat', flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal    = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx  = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx])


def loso_f1(X_feat, use_calibration, max_features, seed=42):
    """LOSO-CV mean F1 for a given feature matrix + HP."""
    imp = SimpleImputer(strategy='median')
    scores = []
    for subj in subjects:
        te_mask = groups_all == subj
        tr_mask = ~te_mask
        X_tr, y_tr = X_feat[tr_mask], y_all[tr_mask]
        X_te, y_te = X_feat[te_mask], y_all[te_mask]
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[tr_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[te_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        if use_calibration:
            X_tr_n, y_tr, X_te_n, y_te = calibration_split(
                X_tr_n, y_tr, X_te_n, y_te, 0.5)
        clf = ExtraTreesClassifier(
            n_estimators=1000, max_features=max_features,
            min_samples_leaf=1, class_weight='balanced',
            random_state=seed, n_jobs=-1,
        )
        clf.fit(X_tr_n, y_tr)
        scores.append(f1_score(y_te, clf.predict(X_te_n),
                               average='binary', zero_division=0))
    return float(np.mean(scores))


def grid_search_mf(X_feat, use_calibration, label):
    """Search max_features → best LOSO-CV F1."""
    mf_grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    best_f1, best_mf = -1.0, None
    print(f'\n  Grid search [{label}]  ({X_feat.shape[1]} features):', flush=True)
    for mf in mf_grid:
        f1v = loso_f1(X_feat, use_calibration, max_features=mf)
        tag = f'    max_features={str(mf):<5} → F1={f1v:.4f}'
        if f1v > best_f1:
            best_f1 = f1v
            best_mf = mf
            tag += '  ★'
        print(tag, flush=True)
    print(f'  → Best: max_features={best_mf}  F1={best_f1:.4f}', flush=True)
    return best_f1, best_mf


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation: Watch → Demo → Calib → HP(searched) → ECG
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 65, flush=True)
print('  Ablation: Watch → Demo → Calib → HP → ECG', flush=True)
print('=' * 65, flush=True)

print('\n[1] Watch only  (max_features=0.5)', flush=True)
f1_watch = loso_f1(X_watch, use_calibration=False, max_features=0.5)
print(f'    F1 = {f1_watch:.4f}', flush=True)

print('\n[2] + Demographics  (max_features=0.5)', flush=True)
f1_demo = loso_f1(X_watch_demo, use_calibration=False, max_features=0.5)
print(f'    F1 = {f1_demo:.4f}', flush=True)

print('\n[3] + Subject calibration  (max_features=0.5)', flush=True)
f1_calib = loso_f1(X_watch_demo, use_calibration=True, max_features=0.5)
print(f'    F1 = {f1_calib:.4f}', flush=True)

# HP search for watch+demo+calib (no ECG)
f1_hp, best_mf_nodemo = grid_search_mf(
    X_watch_demo, use_calibration=True, label='watch+demo+calib')

# HP search for full feature set (with ECG)
f1_final, best_mf_full = grid_search_mf(
    X_WIN, use_calibration=True, label='watch+demo+calib+ECG')

# ── Summary ───────────────────────────────────────────────────────────────────

print('\n' + '=' * 65, flush=True)
print('  RESULTS', flush=True)
print('=' * 65, flush=True)

results = [
    ('Watch only',                f1_watch),
    ('+ Demographics',            f1_demo),
    ('+ Subject calibration',     f1_calib),
    ('+ HP tuning',               f1_hp),
    ('+ ECG (BIOPAC)  [FINAL]',   f1_final),
]
for lbl, f1v in results:
    print(f'  {lbl:<40} F1={f1v:.4f}', flush=True)

print(f'\n  Best HP (watch+demo+calib) : max_features={best_mf_nodemo}', flush=True)
print(f'  Best HP (full incl. ECG)   : max_features={best_mf_full}', flush=True)

df_abl = pd.DataFrame(results, columns=['stage', 'f1'])
df_abl.to_csv('results/ablation_demo_first_hp_searched.csv', index=False)
print('\nSaved results/ablation_demo_first_hp_searched.csv', flush=True)


# ── Figure ────────────────────────────────────────────────────────────────────

labels = [
    'Watch only\n(ET-1000)',
    '+ Demographics',
    '+ Subject\ncalibration',
    '+ HP tuning\n(max_feat searched)',
    '+ ECG\n(BIOPAC) [FINAL]',
]
f1_vals = [r[1] for r in results]

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))
bars   = ax.bar(labels, f1_vals, color=colors, alpha=0.9, width=0.6)
for bar, f1v in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, f1v + 0.004,
            f'{f1v:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(0.95, color='red', linestyle='--', alpha=0.6, label='F1=0.95')
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0.85, 1.0)
ax.set_title(
    'Anxolotl LOSO-CV — Watch → Demo → Calib → HP → ECG\n'
    f'HP tunado por etapa  |  watch+demo: max_feat={best_mf_nodemo}  '
    f'|  full: max_feat={best_mf_full}'
)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig18c_ablation_hp_searched.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/fig18c_ablation_hp_searched.png', flush=True)

print('\nDONE', flush=True)
