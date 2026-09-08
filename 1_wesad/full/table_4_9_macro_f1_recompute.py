"""
Table 4.9 macro-F1 recompute -- 2026-09-06.

Table 4.9's F1 column is not internally comparable: the original per-model
notebooks that produced its numbers use different F1 averaging --
`logreg-smote.ipynb`, `rf-smote-version.ipynb`, `svm-wesad-full.ipynb` use
average='weighted' (or read the classification_report's rows by eye);
`xgboost-light-gbm.ipynb` uses sklearn's default (positive-class-only F1).
None use macro-F1 as a computed quantity, even though the printed
classification_report's "macro avg" row happens to match the chapter's
cited F1 for LogReg (0.80) and XGBoost (0.90) at 2-decimal precision.

This script recomputes macro-F1 = f1_score(y_true, y_pred, average='macro')
for the 4 models whose exact per-sample predictions are NOT saved anywhere
in the repo (checked first -- see chat): LogisticRegression, RandomForest,
LightGBM, SVM. It retrains each LOCALLY on the archived 137-feature
snapshot (NOT Kaggle), reusing each model's EXACT best hyperparameters as
printed in its original notebook's own final cell (not the possibly-
incomplete hyperparameters already used by wesad_classical_rerun.py --
concretely: RF is missing min_samples_split=10, LightGBM is missing
min_child_samples/subsample/colsample_bytree, LogReg is missing tol=0.001
and uses max_iter=1000 instead of 5000; all three are corrected here).

XGBoost is NOT retrained here: `xgboost-light-gbm.ipynb`'s best
hyperparameters (n_estimators=200, max_depth=6, learning_rate=0.1,
SimpleImputer(mean), random_state=42) are IDENTICAL to
wesad_classical_rerun.py's make_xgb(), so that script's already-computed
XGBoost_smote macro_f1 (0.8984, in results/wesad_classical_rerun_task1.json
and results_master.csv) is reused as-is -- no new computation needed.

Only the WITH-SMOTE condition is run for each model -- Table 4.9 is a
"with SMOTE" table; the without-SMOTE condition is not needed here (that's
Table 4.5's job, already resolved separately).

FLAGS, not silently absorbed:
  - SVM: no random_state on SVC in the original notebook -- kept unset
    here for fidelity; not bit-exact reproducible.
  - LightGBM: original notebook used device='gpu' (Kaggle); this runs on
    CPU -- LightGBM's histogram binning can differ in the last few bits
    between GPU and CPU, a much smaller effect than the hyperparameters
    it's actually being corrected for here.
  - LogReg/RF/SVM use SimpleImputer(strategy='median'), matching their
    original notebooks; LightGBM (and XGBoost) use strategy='mean',
    matching theirs -- this split is a pre-existing chapter inconsistency,
    reproduced deliberately here for fidelity, not introduced by this
    script.
"""
import pickle, os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

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

def run_loso(model_builder, imputer_strategy, X, y, groups):
    all_true, all_pred, all_prob = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy=imputer_strategy)
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        X_train_p, y_train = SMOTE(random_state=42).fit_resample(X_train_p, y_train)
        model = model_builder()
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

# Exact best hyperparameters from each model's own notebook (not the rerun's
# approximate/incomplete versions) -- see docstring.
MODELS = {
    'LogisticRegression': (
        lambda: LogisticRegression(C=0.001, penalty='l1', solver='saga',
                                    max_iter=5000, tol=0.001, random_state=42),
        'median',
    ),
    'RandomForest': (
        lambda: RandomForestClassifier(n_estimators=200, max_depth=10, max_features='sqrt',
                                        min_samples_split=10, min_samples_leaf=1,
                                        random_state=42, n_jobs=-1),
        'median',
    ),
    'LightGBM': (
        lambda: LGBMClassifier(n_estimators=305, max_depth=4, learning_rate=0.19503165674949738,
                                num_leaves=37, min_child_samples=30, subsample=0.6537551310524808,
                                colsample_bytree=0.8522263467378263, random_state=42,
                                verbose=-1, n_jobs=-1),
        'mean',
    ),
    'SVM': (
        lambda: SVC(C=0.1, kernel='poly', degree=3, gamma='scale', class_weight='balanced',
                    probability=True, cache_size=1000),  # no random_state -- matches original
        'median',
    ),
}

results = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results = json.load(f)

for name, (builder, imputer_strategy) in MODELS.items():
    if name in results:
        print(f'-- {name}: already in {OUT_PATH}, skipping --')
        continue
    print(f'\n-- {name} (100% data, WITH SMOTE, {imputer_strategy} imputer) --')
    t0 = time.time()
    try:
        res = run_loso(builder, imputer_strategy, X_full, y_full, groups_full)
        print(f"   macro_f1={res['macro_f1']:.4f}  weighted_f1={res['weighted_f1']:.4f}  "
              f"BalAcc={res['balanced_accuracy']:.4f}  AUC={res['roc_auc']:.4f}")
        results[name] = res
    except Exception as e:
        print(f'   ERROR: {type(e).__name__}: {e}')
        results[name] = {'error': f'{type(e).__name__}: {e}'}
    print(f'   [{name}] {(time.time()-t0)/60:.1f} min')
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f)
    print(f'   (saved progress to {OUT_PATH})')

print(f'\nALL DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
