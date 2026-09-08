"""
Faithful re-run of the user's pasted Kaggle LightGBM notebook cells (Optuna
tuning + LOSO evaluation), to check whether Balanced Accuracy reproduces
across independent Optuna sessions (the search has no fixed sampler seed,
only the final LGBMClassifier has random_state=42 -- so different sessions
can land on different "best" hyperparameters).

Data: same 137-feat/27543-window/15-subject OLD WESAD schema the notebook
used -- processed_data/wesad_archive/processed_data_137feat_snapshot.pkl
(archived copy of the exact pkl the Kaggle notebook loaded).

Everything else copied verbatim from the pasted cells: same param search
space, n_trials=50, same LOSO/StandardScaler/SimpleImputer(mean)/SMOTE
protocol, device='gpu' (confirmed working on this machine, RTX 3050 Ti,
same as the original).

Output confined to this same job directory.
"""
import pickle, os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
import optuna
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
OUT_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_blnorm_2026-07-26')
OLD_PKL = os.path.join(ROOT, 'processed_data', 'wesad_archive', 'processed_data_137feat_snapshot.pkl')

T0 = time.time()
with open(OLD_PKL, 'rb') as f:
    data = pickle.load(f)

df = data['df']
X = data['X']
y = data['y']
groups = data['groups']
feature_cols = data['feature_cols']
print(df.shape)
print(pd.Series(y).value_counts())

logo = LeaveOneGroupOut()

def objective(trial, use_smote):
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

        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(scaler.fit_transform(X_train))
        X_te = imputer.transform(scaler.transform(X_test))

        if use_smote:
            X_tr, y_train = SMOTE(random_state=42).fit_resample(X_tr, y_train)

        model = LGBMClassifier(**params, device='gpu', random_state=42, verbose=-1)
        model.fit(X_tr, y_train)
        f1_scores.append(f1_score(y_test, model.predict(X_te), zero_division=0))
    return np.mean(f1_scores)

def evaluate_lgbm(X, y, groups, best_params, use_smote=True):
    logo = LeaveOneGroupOut()
    all_y_true, all_y_pred, all_y_probs = [], [], []
    fold_f1s, fold_mccs, all_cms = [], [], []
    last_model = None

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test = imputer.transform(scaler.transform(X_test))

        if use_smote:
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = LGBMClassifier(**best_params, device='gpu', random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        last_model = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_probs.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
        all_cms.append(confusion_matrix(y_test, y_pred))

    return {
        'y_test': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_prob': np.array(all_y_probs),
        'f1': float(np.mean(fold_f1s)),
        'f1_std': float(np.std(fold_f1s)),
        'accuracy': float(accuracy_score(all_y_true, all_y_pred)),
        'bal_acc': float(balanced_accuracy_score(all_y_true, all_y_pred)),
        'mcc': float(np.mean(fold_mccs)),
        'roc_auc': float(roc_auc_score(all_y_true, all_y_probs)),
        'cm': np.sum(all_cms, axis=0).tolist(),
        'model': last_model,
    }

results_out = {}

def run_lgbm(use_smote):
    label = "COM SMOTE" if use_smote else "SEM SMOTE"
    print(f"\n{'='*50}\nA correr Optuna LightGBM {label}...\n{'='*50}")

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, use_smote), n_trials=50)
    best_params = study.best_params
    print(f"Melhores parâmetros {label}: {best_params}")

    res = evaluate_lgbm(X, y, groups, best_params, use_smote=use_smote)

    macro_f1 = f1_score(res['y_test'], res['y_pred'], average='macro', zero_division=0)
    print(classification_report(res['y_test'], res['y_pred'], target_names=['Negativo', 'Positivo']))
    print(f"Macro F1 (pooled):  {macro_f1:.4f}")
    print(f"ROC-AUC:            {res['roc_auc']:.4f}")
    print(f"Balanced Accuracy:  {res['bal_acc']:.4f}")
    print(f"MCC:                {res['mcc']:.4f}")
    print(f"F1 Std (por fold):  {res['f1_std']:.4f}")
    print(f"Accuracy:           {res['accuracy']:.4f}")

    results_out[label] = {
        'best_params': best_params,
        'macro_f1_pooled': float(macro_f1),
        'accuracy': res['accuracy'],
        'balanced_accuracy': res['bal_acc'],
        'mcc': res['mcc'],
        'roc_auc': res['roc_auc'],
        'f1_std': res['f1_std'],
        'binary_f1_mean': res['f1'],
    }

run_lgbm(use_smote=True)
run_lgbm(use_smote=False)

with open(os.path.join(OUT_DIR, 'lightgbm_kaggle_notebook_rerun.json'), 'w') as f:
    json.dump(results_out, f, indent=2)

print(f"\nTotal elapsed: {(time.time()-T0)/60:.1f} min")
print("\n=== SUMMARY: Balanced Accuracy ===")
for label, r in results_out.items():
    print(f"  {label:12s} BalAcc={r['balanced_accuracy']:.4f}  Acc={r['accuracy']:.4f}  "
          f"MacroF1={r['macro_f1_pooled']:.4f}  AUC={r['roc_auc']:.4f}")
