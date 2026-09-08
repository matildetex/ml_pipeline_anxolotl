"""
Watch+ECG (Anxolotl) LOSO-CV, phase3 excluded, for comparison against WESAD.

ECG features (processed_data_anxolotl_ecg.pkl) are only extracted at the
original 60s/30s window (699 windows) — combining with a 90s watch-only
window isn't possible without re-extracting ECG, so this comparison is at
60s, matching the row order of the watch pkl exactly (verified: same y,
same groups arrays).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from demographics_analysis import baseline_relative_scale

with open('../processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    watch = pickle.load(f)
with open('../processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
    ecg = pickle.load(f)

assert np.array_equal(np.array(watch['y']), np.array(ecg['y']))
assert np.array_equal(np.array(watch['groups']), np.array(ecg['groups']))

X_watch = np.array(watch['X'], dtype=float)
y_all = np.array(watch['y'])
groups_all = np.array(watch['groups'])
stress_phase = watch['df']['stress_phase'].values

X_ecg_raw = ecg['X_ecg'].astype(float)
ecg_cols = ecg['ecg_cols']
X_ecg = X_ecg_raw.copy()
X_ecg[X_ecg[:, ecg_cols.index('ecg_rmssd')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_rr_std')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_lf_hf')] > 50, ecg_cols.index('ecg_lf_hf')] = np.nan

X_combined = np.hstack([X_watch, X_ecg])

# ── phase3 exclusion ─────────────────────────────────────────────────────
mask = stress_phase != 3
print(f'Windows: {len(y_all)} -> {mask.sum()} after excluding stress_phase==3 '
      f'({(~mask).sum()} removed)')
X_watch_f = X_watch[mask]; X_comb_f = X_combined[mask]
y_f = y_all[mask]; g_f = groups_all[mask]

def loso_eval(X, y, groups, clf_factory):
    subjects = np.unique(groups)
    imp = SimpleImputer(strategy='median')
    m_scores, ba_scores = [], []
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n); X_te_n = imp.transform(X_te_n)
        clf = clf_factory(); clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        m_scores.append(f1_score(y[te], pred, average='macro', zero_division=0))
        ba_scores.append(balanced_accuracy_score(y[te], pred))
        prob = clf.predict_proba(X_te_n)[:, 1] if hasattr(clf, 'predict_proba') else pred.astype(float)
        y_true_all.extend(y[te]); y_prob_all.extend(prob)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc = float('nan')
    return float(np.mean(m_scores)), float(np.std(m_scores)), float(np.mean(ba_scores)), auc


MODELS = {
    'ET-1000-mf0.5': lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'ET-1000-mf0.3': lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.3, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'ET-300-mf0.6':  lambda: ExtraTreesClassifier(n_estimators=300,  max_features=0.6, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'RF-1000':       lambda: RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=-1),
    'HGB':           lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.1, random_state=42),
    'LGBM-gbdt':     lambda: LGBMClassifier(boosting_type='gbdt', n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'LGBM-dart':     lambda: LGBMClassifier(boosting_type='dart', n_estimators=300, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'XGBoost':       lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='logloss', verbosity=0),
    'Ensemble-ET-RF-HGB': lambda: VotingClassifier(voting='hard', estimators=[
        ('et', ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ]),
}

print('\n' + '='*70); print('WATCH-ONLY (60s, phase3 excluded) — for apples-to-apples ref'); print('='*70)
watch_results = {}
for name, f in MODELS.items():
    m, sd, ba, auc = loso_eval(X_watch_f, y_f, g_f, f)
    watch_results[name] = (m, sd, ba, auc)
    print(f'  {name:<22} macro={m:.4f}  sd={sd:.4f}  auc={auc:.4f}')

print('\n' + '='*70); print(f'WATCH + ECG (60s, {X_comb_f.shape[1]} features, phase3 excluded)'); print('='*70)
comb_results = {}
for name, f in MODELS.items():
    m, sd, ba, auc = loso_eval(X_comb_f, y_f, g_f, f)
    comb_results[name] = (m, sd, ba, auc)
    print(f'  {name:<22} macro={m:.4f}  sd={sd:.4f}  auc={auc:.4f}')

best_watch = max(watch_results.items(), key=lambda kv: kv[1][0])
best_comb  = max(comb_results.items(),  key=lambda kv: kv[1][0])
print(f'\nBest watch-only  (60s): {best_watch[0]}  macro={best_watch[1][0]:.4f}')
print(f'Best watch+ECG   (60s): {best_comb[0]}  macro={best_comb[1][0]:.4f}')
print(f'ECG delta at 60s: {best_comb[1][0] - best_watch[1][0]:+.4f}')

with open('../results/watch_ecg_vs_wesad.txt', 'w') as f:
    f.write('Anxolotl Watch-only vs Watch+ECG (60s, phase3 excluded) — LOSO-CV, macro F1\n')
    f.write('='*80 + '\n\n')
    f.write(f'{"Model":<22} {"Watch-only":>12} {"Watch+ECG":>12} {"Delta":>8}\n')
    f.write('-'*80 + '\n')
    for name in MODELS:
        w = watch_results[name][0]; c = comb_results[name][0]
        f.write(f'{name:<22} {w:>12.4f} {c:>12.4f} {c-w:>+8.4f}\n')
    f.write('\n')
    f.write(f'Best watch-only (60s) : {best_watch[0]}  macro={best_watch[1][0]:.4f}\n')
    f.write(f'Best watch+ECG  (60s) : {best_comb[0]}  macro={best_comb[1][0]:.4f}\n')
    f.write(f'Best watch-only (90s, from earlier sweep, no ECG possible at this window): 0.9689 (ET n=300 mf=0.6 msl=1)\n')
print('\nSaved results/watch_ecg_vs_wesad.txt')
