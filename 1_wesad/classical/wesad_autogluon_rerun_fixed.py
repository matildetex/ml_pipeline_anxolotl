"""
AutoGluon rerun WITH the Ray/NVML fix -- WESAD FULL (137 feat) + WRIST (56 feat),
old schema (archived snapshot), matching the feature counts results/wesad_autogluon.json
originally used but that predates the fix (see RECONCILIATION.md SS G7).

Fix: CUDA_VISIBLE_DEVICES="" before importing autogluon/ray (must be set before import).
Settings matched to the Task 4 rerun that worked: presets='good_quality', time_limit=90s/fold,
dynamic_stacking=False, calibrate_decision_threshold=False (consistency with plain 0.5-threshold
predict() elsewhere in this audit), refit_full=True (required for predict() to work post-fit).

Does NOT overwrite results/wesad_autogluon.json -- writes to a new file.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # must precede autogluon/ray import
import pickle, json, time, warnings, shutil
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                              roc_auc_score, balanced_accuracy_score, matthews_corrcoef)
from autogluon.tabular import TabularPredictor

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
RESULTS_DIR = os.path.join(ROOT, 'results')
T0 = time.time()

with open(os.path.join(ROOT, 'processed_data/wesad_archive/processed_data_137feat_snapshot.pkl'), 'rb') as f:
    d = pickle.load(f)
X_full = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
y = pd.Series(np.array(d['y'])).reset_index(drop=True)
groups = np.array(d['groups'])
feat_cols = list(d['feature_cols'])
X_full = X_full.reset_index(drop=True)
print(f'FULL: {X_full.shape}, subjects={len(np.unique(groups))}')
assert len(feat_cols) == 137, (
    f'Expected 137 features from the old-schema WESAD snapshot (thesis Section 4.2), '
    f'got {len(feat_cols)} -- see DATA.md.'
)

wrist_cols = [c for c in feat_cols if any(
    c.startswith(sig) for sig in ['acc_wrist', 'eda_wrist', 'temp_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_'])]
X_wrist = X_full[wrist_cols].reset_index(drop=True)
print(f'WRIST: {X_wrist.shape} ({len(wrist_cols)} feat)')
assert len(wrist_cols) == 56, (
    f'Expected 56 wrist features from the old-schema WESAD snapshot, got {len(wrist_cols)} -- see DATA.md.'
)

logo = LeaveOneGroupOut()

def run_autogluon_loso(Xd, tag):
    t0 = time.time()
    all_true, all_pred, all_prob, fold_f1s = [], [], [], []
    for fi, (train_idx, test_idx) in enumerate(logo.split(Xd, y, groups)):
        fold_path = f'/tmp/ag_wesad_{tag}_fold{fi}'
        if os.path.exists(fold_path):
            shutil.rmtree(fold_path)
        train_df = Xd.iloc[train_idx].copy(); train_df['label'] = y.iloc[train_idx].values
        test_df = Xd.iloc[test_idx]
        y_test = y.iloc[test_idx].values
        predictor = TabularPredictor(label='label', eval_metric='f1_macro', verbosity=0,
                                      path=fold_path).fit(
            train_df, time_limit=90, presets='good_quality', dynamic_stacking=False,
            calibrate_decision_threshold=False)
        preds = predictor.predict(test_df).values
        proba = predictor.predict_proba(test_df)[1].values
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_test); all_pred.extend(preds); all_prob.extend(proba)
        print(f'  [{tag}] fold {fi+1:2d}  F1={f1:.4f}  ({(time.time()-t0)/60:.1f} min elapsed)', flush=True)
        shutil.rmtree(fold_path, ignore_errors=True)

    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    res = dict(
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        fold_f1s=[float(v) for v in fold_f1s],
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        precision_macro=float(precision_score(all_true, all_pred, average='macro', zero_division=0)),
        recall_macro=float(recall_score(all_true, all_pred, average='macro', zero_division=0)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        mcc=float(matthews_corrcoef(all_true, all_pred)),
        n_features=Xd.shape[1], elapsed_min=(time.time()-t0)/60,
    )
    print(f'[{tag}] FINAL: {res}')
    return res

results = {}
results['full_137feat'] = run_autogluon_loso(X_full, 'full')
with open(os.path.join(RESULTS_DIR, 'wesad_autogluon_rerun_fixed.json'), 'w') as f:
    json.dump(results, f, indent=1)
print('Saved after FULL.')

results['wrist_56feat'] = run_autogluon_loso(X_wrist, 'wrist')
with open(os.path.join(RESULTS_DIR, 'wesad_autogluon_rerun_fixed.json'), 'w') as f:
    json.dump(results, f, indent=1)
print('Saved after WRIST.')

with open(os.path.join(RESULTS_DIR, 'wesad_autogluon_rerun_fixed.txt'), 'w') as f:
    f.write('AutoGluon RERUN with Ray/NVML fix -- WESAD old schema (137/56 feat) -- 2026-07-25\n')
    f.write('=' * 70 + '\n\n')
    for tag, r in results.items():
        f.write(f'{tag} (n_feat={r["n_features"]}):\n')
        f.write(f"  MacroF1={r['macro_f1']:.4f}+-{r['macro_f1_std']:.4f}  Acc={r['accuracy']:.4f}  "
                f"BalAcc={r['balanced_accuracy']:.4f}  Prec={r['precision_macro']:.4f}  "
                f"Rec={r['recall_macro']:.4f}  AUC={r['roc_auc']:.4f}  MCC={r['mcc']:.4f}\n")
        f.write(f"  elapsed: {r['elapsed_min']:.1f} min\n\n")
    f.write(f'TOTAL: {(time.time()-T0)/60:.1f} min\n')

print(f'\nALL DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
