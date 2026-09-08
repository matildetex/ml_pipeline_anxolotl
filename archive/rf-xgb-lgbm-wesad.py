"""
WESAD classical models -- RF (RandomizedSearchCV) + XGBoost (grid) + LightGBM (Optuna),
COM/SEM SMOTE, LOSO-CV, wrist-reduced 40-feature set.

Built 2026-07-24 from:
  - RF: 3_anxolotl/rf-smote-version.ipynb (Anxolotl paths -> WESAD paths), demographic
    columns dropped so all three models share the identical 40-feature set.
  - XGBoost + LightGBM: user-provided notebook content (originally Kaggle-hosted,
    /kaggle/input/notebooks/matildeteixeira/wesad-preprocess/processed_data.pkl,
    file itself not found anywhere on this disk -- see RECONCILIATION.md SS G7).

Data source: the OLD/recovered WESAD schema (DANN/wesad/processed_data.pkl), matching
exactly the 8-stats-per-channel column names printed in the pasted notebook
(acc_wrist_mean...kurtosis, no slope/lf_hf/delta_mean) -- NOT the current in-repo
processed_data/wesad/processed_data.pkl (which is the new 11-stats/channel schema
and would select 55, not 40, features with the same prefix filter).
"""
import pickle, os, itertools, warnings, sys, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    make_scorer,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

t0 = time.time()

# ---------------------------------------------------------------------------
# Load WESAD (old/recovered schema -- matches the pasted notebook's column names)
# ---------------------------------------------------------------------------
DANN_WESAD_PKL = '/home/matildxi/Secretária/Documentos/Tese/DANN/wesad/processed_data.pkl'
with open(DANN_WESAD_PKL, 'rb') as f:
    data = pickle.load(f)

df = data['df']
X_full = data['X'] if isinstance(data['X'], pd.DataFrame) else pd.DataFrame(data['X'], columns=data['feature_cols'])
y = pd.Series(data['y'])
groups = np.array(data['groups'])
feature_cols = list(data['feature_cols'])

print('WESAD (old schema) loaded:', X_full.shape, ' subjects:', len(np.unique(groups)))
print(y.value_counts())

selected_features = [c for c in feature_cols if any(
    c.startswith(sig) for sig in ['acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_'])]
print(f'{len(selected_features)} features: {selected_features}')

X = X_full[selected_features]
logo = LeaveOneGroupOut()

def summarize(name, label, y_test, y_pred, y_prob, fold_f1s, fold_mccs):
    print(classification_report(y_test, y_pred, zero_division=0))
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mccm = float(np.mean(fold_mccs))
    f1m = float(np.mean(fold_f1s))
    f1s = float(np.std(fold_f1s))
    print(f'ROC-AUC:            {auc:.3f}')
    print(f'Balanced Accuracy:  {ba:.3f}')
    print(f'MCC:                {mccm:.3f}')
    print(f'F1 Std (por fold):  {f1s:.3f}')
    return dict(model=name, smote=label, f1=f1m, f1_std=f1s, accuracy=acc,
                bal_acc=ba, mcc=mccm, roc_auc=auc)

all_results = []

# ===========================================================================
# Modelo Random Forest  (adapted from 3_anxolotl/rf-smote-version.ipynb)
# ===========================================================================
print('\n' + '=' * 60)
print('  MODELO RANDOM FOREST')
print('=' * 60)

param_dist = {
    'model__n_estimators':      [100, 200],
    'model__max_depth':         [10, None],
    'model__min_samples_leaf':  [1, 4],
    'model__max_features':      ['sqrt', 0.3],
    'model__min_samples_split': [2, 5, 10],
}
scoring = {
    'F1':       make_scorer(f1_score, average='weighted', zero_division=0),
    'Accuracy': make_scorer(accuracy_score),
    'Recall':   make_scorer(recall_score, average='weighted', zero_division=0),
}

def rf_evaluate_best_model(best_estimator, use_smote=False):
    all_y_test, all_y_pred, all_y_prob = [], [], []
    fold_f1s, fold_mccs = [], []
    X_arr = X.values
    y_arr = y.values
    for train_idx, test_idx in logo.split(X_arr, y_arr, groups):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        model = clone(best_estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        all_y_test.extend(y_test); all_y_pred.extend(y_pred); all_y_prob.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
    return np.array(all_y_test), np.array(all_y_pred), np.array(all_y_prob), fold_f1s, fold_mccs

def rf_run_search(n_iter, use_smote=False):
    label = 'COM SMOTE' if use_smote else 'SEM SMOTE'
    steps = [
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(random_state=42, n_jobs=1)),
    ]
    if use_smote:
        steps.insert(2, ('smote', SMOTE(random_state=42)))
    pipeline = Pipeline(steps)
    print(f'\nIniciando busca {label} | n_iter={n_iter}...')
    tA = time.time()
    search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring=scoring, refit='F1', cv=logo, n_jobs=-1, random_state=42, verbose=1,
    )
    search.fit(X, y, groups=groups)
    print(f'  ({time.time()-tA:.1f}s)  Melhores parâmetros: {search.best_params_}')
    return search.best_estimator_, label

RUN_RF = False  # 2026-07-24: skipped this round -- RF RandomizedSearchCV(n_iter=40) took
                # 74.3min for the COM-SMOTE half alone (600 fits), way over budget.
                # XGBoost/LightGBM run first; RF gets its own, cheaper pass separately.
if RUN_RF:
    for use_smote in [True, False]:
        best_est, label = rf_run_search(n_iter=40, use_smote=use_smote)
        yt, yp, ypr, ff1, fmcc = rf_evaluate_best_model(best_est, use_smote=use_smote)
        res = summarize('RF', label, yt, yp, ypr, ff1, fmcc)
        all_results.append(res)
        os.makedirs('roc_data', exist_ok=True)
        with open(f"roc_data/rf_{'smote' if use_smote else 'no_smote'}_wesad.pkl", 'wb') as f:
            pickle.dump({'y_test': yt, 'y_pred': yp, 'y_prob': ypr, **res}, f)
else:
    print('\n(RF search skipped this run -- see RUN_RF flag)')
    os.makedirs('roc_data', exist_ok=True)

# ===========================================================================
# Modelo XGBoost
# ===========================================================================
print('\n' + '=' * 60)
print('  MODELO XGBOOST')
print('=' * 60)

xgb_param_grid = {
    'n_estimators':  [75, 100, 200],
    'max_depth':     [5, 6, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}
xgb_keys = list(xgb_param_grid.keys())
xgb_values = list(xgb_param_grid.values())

def evaluate_xgb(best_params, use_smote=True):
    all_y_true, all_y_pred, all_y_probs = [], [], []
    fold_f1s, fold_mccs = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        if use_smote:
            X_train_p, y_train = SMOTE(random_state=42).fit_resample(X_train_p, y_train)
        model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_y_true.extend(y_test); all_y_pred.extend(y_pred); all_y_probs.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_probs), fold_f1s, fold_mccs

def run_xgb(use_smote):
    label = 'COM SMOTE' if use_smote else 'SEM SMOTE'
    print(f'\nA correr XGBoost {label}...')
    best_f1, best_params = -1, None
    for combo in itertools.product(*xgb_values):
        params = dict(zip(xgb_keys, combo))
        fold_f1s = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
            X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
            X_test_p = imputer.transform(scaler.transform(X_test))
            if use_smote:
                X_train_p, y_train = SMOTE(random_state=42).fit_resample(X_train_p, y_train)
            model = XGBClassifier(**params, eval_metric='logloss', random_state=42)
            model.fit(X_train_p, y_train)
            fold_f1s.append(f1_score(y_test, model.predict(X_test_p), zero_division=0))
        mean_f1 = np.mean(fold_f1s)
        if mean_f1 > best_f1:
            best_f1, best_params = mean_f1, params
            print(f'Novo melhor F1: {best_f1:.3f} -> {best_params}')
    print(f'Melhores parâmetros {label}: {best_params}')
    yt, yp, ypr, ff1, fmcc = evaluate_xgb(best_params, use_smote=use_smote)
    res = summarize('XGBoost', label, yt, yp, ypr, ff1, fmcc)
    res['best_params'] = best_params
    all_results.append(res)
    with open(f"roc_data/xgb_{'smote' if use_smote else 'no_smote'}_wesad.pkl", 'wb') as f:
        pickle.dump({'y_test': yt, 'y_pred': yp, 'y_prob': ypr, **res}, f)

run_xgb(use_smote=True)
run_xgb(use_smote=False)

# ===========================================================================
# Modelo LightGBM
# ===========================================================================
print('\n' + '=' * 60)
print('  MODELO LIGHTGBM')
print('=' * 60)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgbm_objective(trial, use_smote):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves':        trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    f1_scores = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(scaler.fit_transform(X_train))
        X_te = imputer.transform(scaler.transform(X_test))
        if use_smote:
            X_tr, y_train = SMOTE(random_state=42).fit_resample(X_tr, y_train)
        model = LGBMClassifier(**params, random_state=42, verbose=-1)
        model.fit(X_tr, y_train)
        f1_scores.append(f1_score(y_test, model.predict(X_te), zero_division=0))
    return float(np.mean(f1_scores))

def evaluate_lgbm(best_params, use_smote=True):
    all_y_true, all_y_pred, all_y_probs = [], [], []
    fold_f1s, fold_mccs = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        if use_smote:
            X_train_p, y_train = SMOTE(random_state=42).fit_resample(X_train_p, y_train)
        model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_y_true.extend(y_test); all_y_pred.extend(y_pred); all_y_probs.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
    return np.array(all_y_true), np.array(all_y_pred), np.array(all_y_probs), fold_f1s, fold_mccs

def run_lgbm(use_smote):
    label = 'COM SMOTE' if use_smote else 'SEM SMOTE'
    print(f'\nA correr Optuna LightGBM {label}...')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: lgbm_objective(trial, use_smote), n_trials=30)
    best_params = study.best_params
    print(f'Melhores parâmetros {label}: {best_params}')
    yt, yp, ypr, ff1, fmcc = evaluate_lgbm(best_params, use_smote=use_smote)
    res = summarize('LightGBM', label, yt, yp, ypr, ff1, fmcc)
    res['best_params'] = best_params
    all_results.append(res)
    with open(f"roc_data/lgbm_{'smote' if use_smote else 'no_smote'}_wesad.pkl", 'wb') as f:
        pickle.dump({'y_test': yt, 'y_pred': yp, 'y_prob': ypr, **res}, f)

run_lgbm(use_smote=True)
run_lgbm(use_smote=False)

# ===========================================================================
print('\n' + '=' * 60)
print('  RESUMO FINAL')
print('=' * 60)
for r in all_results:
    bp = f"  params={r['best_params']}" if 'best_params' in r else ''
    print(f"{r['model']:10} {r['smote']:10} F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}  "
          f"BalAcc={r['bal_acc']:.4f}  AUC={r['roc_auc']:.4f}  MCC={r['mcc']:.4f}{bp}")

print(f'\nTotal elapsed: {(time.time()-t0)/60:.1f} min')
