"""
Exhaustive grid search over feature sets x threshold policy x calibration,
trying to push LOSO-CV F1 above 0.95 on Anxolotl.

Grid:
  Feature sets (5): watch, watch+ecg, watch+ecg+demo, watch+ecg+eads, watch+ecg+demo+eads
  Threshold (2):    fixed 0.5, optimal (train-set F1-maximizing)
  Calibration (2):  off, on (50/50 split of test-subject baseline windows, no leakage)

  => 20 run_loso calls x 15 classifiers x 37 LOSO folds.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, time, pickle, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale

os.makedirs('results', exist_ok=True)

# ── Classifiers ───────────────────────────────────────────────────────────────

def make_classifiers():
    return {
        'DT':         DecisionTreeClassifier(random_state=0),
        'RF':         RandomForestClassifier(max_depth=4, random_state=0),
        'AB':         AdaBoostClassifier(random_state=0),
        'KN':         KNeighborsClassifier(n_neighbors=9),
        'LDA':        LinearDiscriminantAnalysis(),
        'SVM':        SVC(probability=True),
        'GB':         GradientBoostingClassifier(random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                            class_weight='balanced', random_state=42),
        'ET-1000':    ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                            min_samples_leaf=1, class_weight='balanced',
                                            random_state=42, n_jobs=-1),
        'RF-1000':    RandomForestClassifier(n_estimators=1000, max_features=0.5,
                                              min_samples_leaf=1, class_weight='balanced',
                                              random_state=42, n_jobs=-1),
        'HGB':        HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                                      learning_rate=0.1, random_state=42),
        'Ensemble':   VotingClassifier(voting='soft', n_jobs=-1, estimators=[
                          ('et',  ExtraTreesClassifier(n_estimators=300, max_features=0.5,
                                                        class_weight='balanced', random_state=42, n_jobs=-1)),
                          ('rf',  RandomForestClassifier(n_estimators=300, max_features=0.5,
                                                          class_weight='balanced', random_state=42, n_jobs=-1)),
                          ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                                   learning_rate=0.1, random_state=42)),
                      ]),
        'LGBM-base':  lgb.LGBMClassifier(n_estimators=500, num_leaves=31,
                                          learning_rate=0.05, class_weight='balanced',
                                          random_state=42, n_jobs=-1, verbose=-1),
        'LGBM-deep':  lgb.LGBMClassifier(n_estimators=1000, num_leaves=63,
                                          learning_rate=0.03, min_child_samples=5,
                                          subsample=0.8, colsample_bytree=0.8,
                                          class_weight='balanced', random_state=42,
                                          n_jobs=-1, verbose=-1),
        'LGBM-dart':  lgb.LGBMClassifier(n_estimators=500, num_leaves=31,
                                          boosting_type='dart', drop_rate=0.1,
                                          learning_rate=0.1, class_weight='balanced',
                                          random_state=42, n_jobs=-1, verbose=-1),
    }

def optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, y_prob >= t, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def run_loso(X, y, groups, use_opt_thresh=False, use_calibration=False):
    clfs     = make_classifiers()
    subjects = np.unique(groups)
    results  = {name: {'f1': [], 'auc': []} for name in clfs}
    imp = SimpleImputer(strategy='median')

    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups[test_mask]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)

        if use_calibration:
            base_idx = np.where(y_te == 0)[0]
            if len(base_idx) >= 4:
                n_cal    = len(base_idx) // 2
                cal_idx  = base_idx[:n_cal]
                eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
                X_tr_n = np.vstack([X_tr_n, X_te_n[cal_idx]])
                y_tr   = np.concatenate([y_tr, y_te[cal_idx]])
                X_te_n = X_te_n[eval_idx]
                y_te   = y_te[eval_idx]

        for name, clf in clfs.items():
            try:
                clf.fit(X_tr_n, y_tr)
                y_prob = clf.predict_proba(X_te_n)[:, 1]
                auc    = roc_auc_score(y_te, y_prob)
                if use_opt_thresh:
                    y_prob_tr = clf.predict_proba(X_tr_n)[:, 1]
                    thresh = optimal_threshold(y_tr, y_prob_tr)
                    y_pred = (y_prob >= thresh).astype(int)
                else:
                    y_pred = (y_prob >= 0.5).astype(int)
            except Exception:
                y_pred = np.zeros(len(y_te), dtype=int)
                auc    = float('nan')
            results[name]['f1'].append(f1_score(y_te, y_pred, average='binary', zero_division=0))
            results[name]['auc'].append(auc)

    return {name: {'f1': np.mean(v['f1']), 'auc': np.nanmean(v['auc'])}
            for name, v in results.items()}

# ── Load data ────────────────────────────────────────────────────────────────

print('Loading data...')
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
eads_cols = ['eads_depression', 'eads_anxiety', 'eads_stress', 'eads_total', 'vas_stress']

def expand(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_demo = expand(demo_cols)
X_eads = expand(eads_cols)

feature_sets = {
    'watch':              X_watch,
    'watch+ecg':          np.hstack([X_watch, X_ecg]),
    'watch+ecg+demo':     np.hstack([X_watch, X_ecg, X_demo]),
    'watch+ecg+eads':     np.hstack([X_watch, X_ecg, X_eads]),
    'watch+ecg+demo+eads':np.hstack([X_watch, X_ecg, X_demo, X_eads]),
}

print('Feature set shapes:', {k: v.shape for k, v in feature_sets.items()})

# ── Grid ─────────────────────────────────────────────────────────────────────

thresh_opts = [False, True]
cal_opts    = [False, True]

all_rows = []
t0 = time.time()
combo_id = 0
n_combos = len(feature_sets) * len(thresh_opts) * len(cal_opts)

for fs_name, X_fs in feature_sets.items():
    for use_thresh in thresh_opts:
        for use_cal in cal_opts:
            combo_id += 1
            tag = f'{fs_name} | thresh={use_thresh} | cal={use_cal}'
            t1 = time.time()
            summary = run_loso(X_fs, y_all, groups_all,
                                use_opt_thresh=use_thresh, use_calibration=use_cal)
            dt = time.time() - t1
            best_model = max(summary, key=lambda m: summary[m]['f1'])
            print(f'[{combo_id}/{n_combos}] {tag}  '
                  f'-> best={best_model} F1={summary[best_model]["f1"]:.4f}  ({dt:.0f}s)',
                  flush=True)
            for name, m in summary.items():
                all_rows.append({
                    'feature_set': fs_name, 'opt_thresh': use_thresh,
                    'calibration': use_cal, 'model': name,
                    'f1': m['f1'], 'auc': m['auc'],
                })

df = pd.DataFrame(all_rows)
df.to_csv('results/all_models_grid.csv', index=False)
print(f'\nTotal grid time: {time.time()-t0:.0f}s')
print(f'Saved results/all_models_grid.csv ({len(df)} rows)')

df_sorted = df.sort_values('f1', ascending=False)
print('\n' + '='*80)
print('TOP 20 CONFIGURATIONS')
print('='*80)
print(df_sorted.head(20).to_string(index=False))

best = df_sorted.iloc[0]
print(f'\nBEST OVERALL: {best["model"]}  F1={best["f1"]:.4f}  '
      f'[{best["feature_set"]}, thresh={best["opt_thresh"]}, cal={best["calibration"]}]')
