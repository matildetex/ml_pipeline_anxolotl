"""
SVM SEM SMOTE recompute -- 2026-09-07, companion to table_4_9_macro_f1_recompute.py.

Same exact config as that script's SVM (WITH SMOTE) run: 100% of the 137-feat
snapshot, LOSO, SVC(C=0.1, kernel='poly', degree=3, gamma='scale',
class_weight='balanced', probability=True, cache_size=1000) -- no
random_state, matching svm-wesad-full.ipynb's original (not bit-exact
reproducible; already flagged and accepted by the user for the COM-SMOTE
side). SimpleImputer(strategy='median'), StandardScaler. Only difference
from the COM-SMOTE run: no SMOTE.fit_resample() call on the training fold.

Purpose: SVM is being made the single canonical source for BOTH Table 4.5
(SEM/COM SMOTE columns) and Table 4.9 (COM-SMOTE row), replacing
svm-wesad-full.ipynb (weighted-F1, and its AUROC didn't reproduce on
retrain either). The COM-SMOTE side already exists in
results/table_4_9_macro_f1_recompute.json ("SVM" key); this script adds
the SEM-SMOTE side under a new "SVM_no_smote" key in the same file.
"""
import pickle, os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
)

T0 = time.time()
ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
RESULTS_DIR = os.path.join(ROOT, 'results')
OUT_PATH = os.path.join(RESULTS_DIR, 'table_4_9_macro_f1_recompute.json')

with open(os.path.join(ROOT, 'processed_data/wesad_archive/processed_data_137feat_snapshot.pkl'), 'rb') as f:
    d = pickle.load(f)
X_full = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
y_full = pd.Series(np.array(d['y'])).reset_index(drop=True)
groups_full = np.array(d['groups'])
feat_cols = list(d['feature_cols'])
X_full = X_full.reset_index(drop=True)
assert len(feat_cols) == 137, f'Expected 137 features, got {len(feat_cols)} -- see DATA.md.'
print(f'FULL: {X_full.shape}, subjects={len(np.unique(groups_full))}')

logo = LeaveOneGroupOut()

def run_loso_no_smote(X, y, groups):
    all_true, all_pred, all_prob = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='median')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        model = SVC(C=0.1, kernel='poly', degree=3, gamma='scale',
                    class_weight='balanced', probability=True, cache_size=1000)
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_true.extend(y_test); all_pred.extend(y_pred); all_prob.extend(y_prob)
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    return dict(
        y_test=all_true.tolist(), y_pred=all_pred.tolist(), y_prob=all_prob.tolist(),
        macro_f1=float(f1_score(all_true, all_pred, average='macro', zero_division=0)),
        weighted_f1=float(f1_score(all_true, all_pred, average='weighted', zero_division=0)),
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
    )

results = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results = json.load(f)

if 'SVM_no_smote' in results:
    print('SVM_no_smote already present, skipping')
else:
    print('\n-- SVM (100% data, WITHOUT SMOTE, median imputer) --')
    t0 = time.time()
    res = run_loso_no_smote(X_full, y_full, groups_full)
    print(f"   macro_f1={res['macro_f1']:.4f}  weighted_f1={res['weighted_f1']:.4f}  "
          f"BalAcc={res['balanced_accuracy']:.4f}  AUC={res['roc_auc']:.4f}")
    results['SVM_no_smote'] = res
    print(f'   [SVM_no_smote] {(time.time()-t0)/60:.1f} min')
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f)
    print(f'   (saved to {OUT_PATH})')

print(f'\nALL DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
