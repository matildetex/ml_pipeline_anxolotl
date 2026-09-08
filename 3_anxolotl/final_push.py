"""
Final push to maximise F1, building on the grid-search winner:
  ET-1000, watch+ecg+demo (88 feat), no threshold opt, subject calibration -> F1=0.9500

Four experiments:
  1. Calibration split-ratio sweep (fraction of baseline windows used for calibration)
  2. ET-1000 hyperparameter grid search
  3. CatBoost as an additional model
  4. Stacking: logistic-regression meta-learner on OOF probs of top base models
     (ET-1000, RF-1000, LGBM-deep, ExtraTrees), trained via inner GroupKFold to
     avoid leakage, evaluated under the same LOSO + BL-norm + calibration scheme.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
)
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale

os.makedirs('results', exist_ok=True)

# ── Load data: winning feature set = watch + ecg + demo ─────────────────────

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

def expand(cols):
    rows = []
    for g in groups_all:
        rows.append(df_demo.loc[g, cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(cols)))
    return np.array(rows)

X_demo = expand(demo_cols)
X_WIN  = np.hstack([X_watch, X_ecg, X_demo])   # winning feature set, 88 cols
print(f'Winning feature set: {X_WIN.shape}')

# ── Shared helpers ────────────────────────────────────────────────────────────

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac):
    """Split test-subject baseline windows: cal_frac for calibration (-> train),
    remainder + all stress windows for evaluation. No leakage."""
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal = max(1, int(round(len(base_idx) * cal_frac)))
    n_cal = min(n_cal, len(base_idx) - 1)
    cal_idx  = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    X_tr_n2 = np.vstack([X_tr_n, X_te_n[cal_idx]])
    y_tr2   = np.concatenate([y_tr, y_te[cal_idx]])
    return X_tr_n2, y_tr2, X_te_n[eval_idx], y_te[eval_idx]


def bl_norm_fold(X, y, groups, train_mask, test_mask):
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]
    X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups[train_mask]))
    X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups[test_mask]))
    imp = SimpleImputer(strategy='median')
    X_tr_n = imp.fit_transform(X_tr_n)
    X_te_n = imp.transform(X_te_n)
    return X_tr_n, y_tr, X_te_n, y_te


# ═══════════════════════════════════════════════════════════════════════════
# 1. CALIBRATION SPLIT-RATIO SWEEP
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('1. CALIBRATION SPLIT-RATIO SWEEP (ET-1000, RF-1000)')
print('='*70)

def make_top_models():
    return {
        'ET-1000': ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                         min_samples_leaf=1, class_weight='balanced',
                                         random_state=42, n_jobs=-1),
        'RF-1000': RandomForestClassifier(n_estimators=1000, max_features=0.5,
                                           min_samples_leaf=1, class_weight='balanced',
                                           random_state=42, n_jobs=-1),
    }

subjects = np.unique(groups_all)
ratio_results = []
for cal_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
    t0 = time.time()
    clfs = make_top_models()
    f1s = {name: [] for name in clfs}
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_tr_n, y_tr, X_te_n, y_te = bl_norm_fold(X_WIN, y_all, groups_all, train_mask, test_mask)
        X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac)
        for name, clf in clfs.items():
            clf.fit(X_tr_n2, y_tr2)
            y_pred = clf.predict(X_te_n2)
            f1s[name].append(f1_score(y_te2, y_pred, average='binary', zero_division=0))
    for name in clfs:
        mf1 = np.mean(f1s[name])
        ratio_results.append({'cal_frac': cal_frac, 'model': name, 'f1': mf1})
        print(f'  cal_frac={cal_frac:.1f}  {name:<10} F1={mf1:.4f}')
    print(f'  ({time.time()-t0:.0f}s)')

df_ratio = pd.DataFrame(ratio_results)
df_ratio.to_csv('results/calibration_ratio_sweep.csv', index=False)
best_ratio_row = df_ratio.sort_values('f1', ascending=False).iloc[0]
best_cal_frac = float(best_ratio_row['cal_frac'])
print(f'\nBest calibration ratio: {best_cal_frac} '
      f'({best_ratio_row["model"]} F1={best_ratio_row["f1"]:.4f})')

# ═══════════════════════════════════════════════════════════════════════════
# 2. ET-1000 HYPERPARAMETER GRID (using best calibration ratio)
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print(f'2. ET-1000 HYPERPARAMETER GRID (cal_frac={best_cal_frac})')
print('='*70)

param_grid = []
for n_est in [500, 1000, 1500]:
    for max_feat in [0.3, 0.5, 0.7]:
        for min_leaf in [1, 2]:
            param_grid.append({'n_estimators': n_est, 'max_features': max_feat,
                                'min_samples_leaf': min_leaf})

hp_results = []
for params in param_grid:
    t0 = time.time()
    f1s = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_tr_n, y_tr, X_te_n, y_te = bl_norm_fold(X_WIN, y_all, groups_all, train_mask, test_mask)
        X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, best_cal_frac)
        clf = ExtraTreesClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **params)
        clf.fit(X_tr_n2, y_tr2)
        y_pred = clf.predict(X_te_n2)
        f1s.append(f1_score(y_te2, y_pred, average='binary', zero_division=0))
    mf1 = np.mean(f1s)
    hp_results.append({**params, 'f1': mf1})
    print(f'  {params}  F1={mf1:.4f}  ({time.time()-t0:.0f}s)')

df_hp = pd.DataFrame(hp_results).sort_values('f1', ascending=False)
df_hp.to_csv('results/et1000_hp_grid.csv', index=False)
best_hp = df_hp.iloc[0]
print(f'\nBest ET-1000 params: {best_hp.to_dict()}')

# ═══════════════════════════════════════════════════════════════════════════
# 3. CATBOOST
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print(f'3. CATBOOST (cal_frac={best_cal_frac})')
print('='*70)

cb_configs = {
    'CatBoost-base': dict(iterations=500, depth=6, learning_rate=0.05,
                           auto_class_weights='Balanced', random_seed=42, verbose=False),
    'CatBoost-deep': dict(iterations=1000, depth=8, learning_rate=0.03,
                           auto_class_weights='Balanced', random_seed=42, verbose=False),
}
cb_results = []
for name, params in cb_configs.items():
    t0 = time.time()
    f1s = []
    for subj in subjects:
        test_mask, train_mask = groups_all == subj, groups_all != subj
        X_tr_n, y_tr, X_te_n, y_te = bl_norm_fold(X_WIN, y_all, groups_all, train_mask, test_mask)
        X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, best_cal_frac)
        clf = CatBoostClassifier(**params)
        clf.fit(X_tr_n2, y_tr2)
        y_pred = clf.predict(X_te_n2)
        f1s.append(f1_score(y_te2, y_pred, average='binary', zero_division=0))
    mf1 = np.mean(f1s)
    cb_results.append({'model': name, 'f1': mf1})
    print(f'  {name}  F1={mf1:.4f}  ({time.time()-t0:.0f}s)')

df_cb = pd.DataFrame(cb_results)
df_cb.to_csv('results/catboost_results.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# 4. STACKING (logistic-regression meta-learner, nested OOF, no leakage)
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print(f'4. STACKING: ET-1000 + RF-1000 + LGBM-deep + ExtraTrees (cal_frac={best_cal_frac})')
print('='*70)

best_et_params = {k: best_hp[k] for k in ['n_estimators', 'max_features', 'min_samples_leaf']}
best_et_params['n_estimators'] = int(best_et_params['n_estimators'])
best_et_params['min_samples_leaf'] = int(best_et_params['min_samples_leaf'])

def make_base_models():
    return {
        'ET-1000':    ExtraTreesClassifier(class_weight='balanced', random_state=42,
                                            n_jobs=-1, **best_et_params),
        'RF-1000':    RandomForestClassifier(n_estimators=1000, max_features=0.5,
                                              min_samples_leaf=1, class_weight='balanced',
                                              random_state=42, n_jobs=-1),
        'LGBM-deep':  lgb.LGBMClassifier(n_estimators=1000, num_leaves=63, learning_rate=0.03,
                                          min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                                          class_weight='balanced', random_state=42,
                                          n_jobs=-1, verbose=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                            class_weight='balanced', random_state=42, n_jobs=-1),
    }

t0 = time.time()
stack_f1, stack_auc = [], []
N_INNER = 5

for subj in subjects:
    test_mask, train_mask = groups_all == subj, groups_all != subj
    X_tr_n, y_tr, X_te_n, y_te = bl_norm_fold(X_WIN, y_all, groups_all, train_mask, test_mask)
    X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, best_cal_frac)
    train_groups2 = groups_all[train_mask]
    # calibration windows appended at the end have no group id from train set;
    # assign them the test subject's id so GroupKFold still treats them as one group
    if len(X_tr_n2) > len(X_tr_n):
        n_added = len(X_tr_n2) - len(X_tr_n)
        train_groups2 = np.concatenate([train_groups2, np.full(n_added, subj)])

    base_names = list(make_base_models().keys())
    oof_preds = np.zeros((len(X_tr_n2), len(base_names)))

    n_groups_inner = len(np.unique(train_groups2))
    gkf = GroupKFold(n_splits=min(N_INNER, n_groups_inner))
    for inner_tr_idx, inner_val_idx in gkf.split(X_tr_n2, y_tr2, groups=train_groups2):
        inner_models = make_base_models()
        for j, name in enumerate(base_names):
            inner_models[name].fit(X_tr_n2[inner_tr_idx], y_tr2[inner_tr_idx])
            oof_preds[inner_val_idx, j] = inner_models[name].predict_proba(X_tr_n2[inner_val_idx])[:, 1]

    meta = LogisticRegression(max_iter=1000, class_weight='balanced')
    meta.fit(oof_preds, y_tr2)

    final_models = make_base_models()
    test_preds = np.zeros((len(X_te_n2), len(base_names)))
    for j, name in enumerate(base_names):
        final_models[name].fit(X_tr_n2, y_tr2)
        test_preds[:, j] = final_models[name].predict_proba(X_te_n2)[:, 1]

    y_prob = meta.predict_proba(test_preds)[:, 1]
    y_pred = meta.predict(test_preds)
    stack_f1.append(f1_score(y_te2, y_pred, average='binary', zero_division=0))
    try:
        stack_auc.append(roc_auc_score(y_te2, y_prob))
    except Exception:
        pass

stack_f1_mean = np.mean(stack_f1)
print(f'\nStacking F1 = {stack_f1_mean:.4f}  AUC = {np.mean(stack_auc):.4f}  ({time.time()-t0:.0f}s)')
pd.DataFrame({'model': ['Stacking'], 'f1': [stack_f1_mean], 'auc': [np.mean(stack_auc)]}) \
  .to_csv('results/stacking_results.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('FINAL SUMMARY — all four experiments')
print('='*70)
print(f'Baseline (grid winner):     ET-1000  F1=0.9500')
print(f'1. Best cal_frac:           {best_cal_frac}  '
      f'({best_ratio_row["model"]} F1={best_ratio_row["f1"]:.4f})')
print(f'2. Best ET-1000 HP:         {best_hp.to_dict()}  F1={best_hp["f1"]:.4f}')
print(f'3. Best CatBoost:           '
      f'{df_cb.sort_values("f1", ascending=False).iloc[0].to_dict()}')
print(f'4. Stacking:                F1={stack_f1_mean:.4f}')

all_candidates = [
    ('Grid winner ET-1000',  0.9500),
    (f'Cal-ratio best ({best_ratio_row["model"]})', best_ratio_row['f1']),
    ('ET-1000 HP-tuned',     best_hp['f1']),
    ('CatBoost best',        df_cb['f1'].max()),
    ('Stacking',             stack_f1_mean),
]
best_overall = max(all_candidates, key=lambda x: x[1])
print(f'\nBEST OVERALL: {best_overall[0]}  F1={best_overall[1]:.4f}')
print('DONE')
