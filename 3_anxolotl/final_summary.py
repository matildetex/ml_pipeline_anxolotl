"""
Final consolidated summary for the dissertation: the full improvement journey
from watch-only baseline to the best configuration found.

Stages:
  1. Watch-only (ET-1000, original params)                       F1=0.928
  2. + ECG (BIOPAC)                                               F1=0.946
  3. + Demographics                                               F1=0.947
  4. + Subject calibration                                        F1=0.950
  5. + Hyperparameter tuning (max_features=0.3)                   F1=0.954

Produces:
  - results/final_summary.csv        (progression table)
  - figures/fig18_final_progression.png
  - figures/fig19_final_confusion_matrix.png
  - results/final_model_per_subject.csv (per-subject F1 for the winning config)

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

# ── Load winning feature set ─────────────────────────────────────────────────

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
rmssd_idx = ecg_cols.index('ecg_rmssd')
std_idx   = ecg_cols.index('ecg_rr_std')
lfhf_idx  = ecg_cols.index('ecg_lf_hf')
X_ecg[X_ecg[:, rmssd_idx] > 300, :] = np.nan
X_ecg[X_ecg[:, std_idx]   > 300, :] = np.nan
X_ecg[X_ecg[:, lfhf_idx]  > 50, lfhf_idx] = np.nan

df_demo = load_demographics('Data_tese.xlsx')
demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
             'respiratory', 'medication', 'exercises', 'exercise_freq',
             'athlete', 'stimulant']

def expand(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_demo = expand(demo_cols)
X_WIN  = np.hstack([X_watch, X_ecg, X_demo])
X_WD   = np.hstack([X_watch, X_demo])           # watch + demo (no ECG)

# ── ECG quality fallback ──────────────────────────────────────────────────────
# Subjects with >50% of windows flagged as ECG artefact use watch+demo only.
_ecg_nan_rows = np.all(np.isnan(X_ecg), axis=1)
ECG_QUALITY_THRESH = 0.50
subjects = np.unique(groups_all)
fallback_subjects = {s for s in subjects
                     if _ecg_nan_rows[groups_all == s].mean() > ECG_QUALITY_THRESH}
print(f'ECG fallback subjects (watch+demo): {sorted(fallback_subjects)}')

# ── Re-run final winning configuration cleanly ───────────────────────────────

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx  = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx])

imp = SimpleImputer(strategy='median')

BASELINE_PARAMS = dict(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                        class_weight='balanced', random_state=42, n_jobs=-1)
FINAL_PARAMS    = dict(n_estimators=1000, max_features=0.3, min_samples_leaf=1,
                        class_weight='balanced', random_state=42, n_jobs=-1)
FINAL_PARAMS_WD = dict(n_estimators=1000, max_features=0.6, min_samples_leaf=1,
                        class_weight='balanced', random_state=42, n_jobs=-1)

def per_subject_f1(X_feat, use_calibration, clf_params):
    """Mean per-subject LOSO-CV F1 for a given feature matrix/config."""
    scores = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_tr, y_tr = X_feat[train_mask], y_all[train_mask]
        X_te, y_te = X_feat[test_mask],  y_all[test_mask]
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        if use_calibration:
            X_tr_n, y_tr, X_te_n, y_te = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)
        clf = ExtraTreesClassifier(**clf_params)
        clf.fit(X_tr_n, y_tr)
        y_pred = clf.predict(X_te_n)
        scores.append(f1_score(y_te, y_pred, average='binary', zero_division=0))
    return np.mean(scores)

def per_subject_f1_adaptive(use_calibration):
    """LOSO-CV F1 with ECG-quality fallback per test subject."""
    scores = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_feat  = X_WD  if subj in fallback_subjects else X_WIN
        params  = FINAL_PARAMS_WD if subj in fallback_subjects else FINAL_PARAMS
        X_tr, y_tr = X_feat[train_mask], y_all[train_mask]
        X_te, y_te = X_feat[test_mask],  y_all[test_mask]
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        if use_calibration:
            X_tr_n, y_tr, X_te_n, y_te = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)
        clf = ExtraTreesClassifier(**params)
        clf.fit(X_tr_n, y_tr)
        scores.append(f1_score(y_te, clf.predict(X_te_n), average='binary', zero_division=0))
    return np.mean(scores)

# ── Intermediate stages (live-recomputed on the current, corrected data) ──────
f1_watch_only = per_subject_f1(X_watch, use_calibration=False, clf_params=BASELINE_PARAMS)
f1_plus_ecg   = per_subject_f1(np.hstack([X_watch, X_ecg]), use_calibration=False, clf_params=BASELINE_PARAMS)
f1_plus_demo  = per_subject_f1(X_WIN, use_calibration=False, clf_params=BASELINE_PARAMS)
f1_plus_cal   = per_subject_f1(X_WIN, use_calibration=True,  clf_params=BASELINE_PARAMS)
print(f'Stage 1 (watch only)       F1={f1_watch_only:.4f}')
print(f'Stage 2 (+ ECG)            F1={f1_plus_ecg:.4f}')
print(f'Stage 3 (+ demographics)   F1={f1_plus_demo:.4f}')
print(f'Stage 4 (+ calibration)    F1={f1_plus_cal:.4f}')

all_y_true, all_y_pred, per_subject = [], [], []

for subj in subjects:
    test_mask, train_mask = groups_all == subj, groups_all != subj
    X_feat  = X_WD  if subj in fallback_subjects else X_WIN
    params  = FINAL_PARAMS_WD if subj in fallback_subjects else FINAL_PARAMS
    X_tr, y_tr = X_feat[train_mask], y_all[train_mask]
    X_te, y_te = X_feat[test_mask],  y_all[test_mask]
    X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
    X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
    X_tr_n = imp.fit_transform(X_tr_n)
    X_te_n = imp.transform(X_te_n)

    X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)

    clf = ExtraTreesClassifier(**params)
    clf.fit(X_tr_n2, y_tr2)
    y_pred = clf.predict(X_te_n2)
    y_prob = clf.predict_proba(X_te_n2)[:, 1]

    f1_s = f1_score(y_te2, y_pred, average='binary', zero_division=0)
    try:
        auc_s = roc_auc_score(y_te2, y_prob)
    except Exception:
        auc_s = np.nan
    per_subject.append({'subject': subj, 'f1': f1_s, 'auc': auc_s, 'n_eval': len(y_te2)})
    all_y_true.append(y_te2)
    all_y_pred.append(y_pred)

y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
pooled_f1  = f1_score(y_true_all, y_pred_all, average='binary', zero_division=0)
pooled_acc = (y_true_all == y_pred_all).mean()

df_subj = pd.DataFrame(per_subject)
df_subj.to_csv('results/final_model_per_subject.csv', index=False)
# Canonical metric throughout this work = mean of per-subject (per-LOSO-fold) F1,
# NOT the pooled/micro F1 across concatenated folds (which is higher here since
# folds are unequal size and pooling lets large-N subjects dominate).
final_f1  = df_subj['f1'].mean()
final_acc = pooled_acc
print(f'Final model: mean per-subject F1={final_f1:.4f}  '
      f'(pooled F1={pooled_f1:.4f}, pooled Acc={pooled_acc:.4f})')

# ── Confusion matrix figure ───────────────────────────────────────────────────

cm = confusion_matrix(y_true_all, y_pred_all)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', 'Stress'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Baseline', 'Stress'])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Final model — ET-1000 (tuned)\nF1={final_f1:.4f}, Acc={final_acc:.4f}')
plt.tight_layout()
plt.savefig('figures/fig19_final_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig19_final_confusion_matrix.png')

# ── Progression table + figure ────────────────────────────────────────────────

stages = [
    ('Watch only\n(ET-1000 baseline)',          f1_watch_only),
    ('+ ECG\n(BIOPAC)',                          f1_plus_ecg),
    ('+ Demographics',                            f1_plus_demo),
    ('+ Subject\ncalibration',                    f1_plus_cal),
    ('+ Hyperparameter\ntuning (final)',          final_f1),
]
df_prog = pd.DataFrame(stages, columns=['stage', 'f1'])
df_prog['delta_vs_baseline'] = df_prog['f1'] - df_prog['f1'].iloc[0]
df_prog.to_csv('results/final_summary.csv', index=False)
print('\n' + df_prog.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(stages)))
bars = ax.bar(df_prog['stage'], df_prog['f1'], color=colors, alpha=0.9, width=0.6)
for bar, f1v in zip(bars, df_prog['f1']):
    ax.text(bar.get_x() + bar.get_width()/2, f1v + 0.005, f'{f1v:.4f}',
            ha='center', fontsize=10, fontweight='bold')
ax.axhline(0.95, color='red', linestyle='--', alpha=0.6, label='F1=0.95 target')
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0.85, 1.0)
ax.set_title('Anxolotl LOSO-CV — full improvement journey\n'
             'Watch-only baseline → ECG (BIOPAC) → Demographics → Calibration → HP tuning')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig18_final_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig18_final_progression.png')

print('\nDONE')
