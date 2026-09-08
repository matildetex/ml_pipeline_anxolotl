"""AutoGluon-only rerun for Task 4 (reduced 55-feat, current schema) -- 2026-07-24.
Appends its result into results/wesad_classical_rerun_task4.json and the combined file.

FIX (3rd root cause, confirmed 2026-07-24): LightGBM/CatBoost/NeuralNet base models
were failing with an unhelpful "Unknown Error" that traced to
`ray._private.thirdparty.pynvml.pynvml.NVMLError_Unknown` -- Ray's GPU-detection
probe via NVML crashing on startup, even though these models request num_gpus=0.
Setting CUDA_VISIBLE_DEVICES="" before importing autogluon/ray makes them skip GPU
probing entirely. Verified standalone: LightGBMXT_BAG_L1 went from instant-fail to
a real 0.9987 val score with this set. NOT a library-version incompatibility --
that was an earlier, wrong guess (versions are within AutoGluon's own declared
compatible ranges).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # must be set before autogluon/ray import
import pickle, json, time, warnings, shutil
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                              roc_auc_score, balanced_accuracy_score, matthews_corrcoef)

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
RESULTS_DIR = os.path.join(ROOT, 'results')
t0 = time.time()

with open(os.path.join(ROOT, 'processed_data/wesad/processed_data.pkl'), 'rb') as f:
    d_cur = pickle.load(f)
X_cur_all = d_cur['X'] if isinstance(d_cur['X'], pd.DataFrame) else pd.DataFrame(d_cur['X'], columns=d_cur['feature_cols'])
y_cur = pd.Series(np.array(d_cur['y'])).reset_index(drop=True)
groups_cur = np.array(d_cur['groups'])
reduced_cols = [c for c in d_cur['feature_cols'] if any(
    c.startswith(sig) for sig in ['acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_'])]
Xr_df = X_cur_all[reduced_cols].reset_index(drop=True)
print(f'REDUCED: {Xr_df.shape}, subjects={len(np.unique(groups_cur))}')

logo = LeaveOneGroupOut()
from autogluon.tabular import TabularPredictor

all_true, all_pred, all_prob, fold_f1s = [], [], [], []
for fi, (train_idx, test_idx) in enumerate(logo.split(Xr_df, y_cur, groups_cur)):
    fold_path = f'/tmp/ag_wesad_reduced_v2_fold{fi}'
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    train_df = Xr_df.iloc[train_idx].copy(); train_df['label'] = y_cur.iloc[train_idx].values
    test_df = Xr_df.iloc[test_idx]
    y_test = y_cur.iloc[test_idx].values
    predictor = TabularPredictor(label='label', eval_metric='f1_macro', verbosity=0,
                                  path=fold_path).fit(
        train_df, time_limit=90, presets='good_quality', dynamic_stacking=False,
        calibrate_decision_threshold=False)
    preds = predictor.predict(test_df).values
    proba = predictor.predict_proba(test_df)[1].values
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    fold_f1s.append(f1)
    all_true.extend(y_test); all_pred.extend(preds); all_prob.extend(proba)
    print(f'  fold {fi+1:2d}  F1={f1:.4f}  ({(time.time()-t0)/60:.1f} min elapsed)', flush=True)
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
)
print('FINAL:', res)

task4_path = os.path.join(RESULTS_DIR, 'wesad_classical_rerun_task4.json')
task4 = json.load(open(task4_path))
task4['AutoGluon'] = res
json.dump(task4, open(task4_path, 'w'))

combined_path = os.path.join(RESULTS_DIR, 'wesad_classical_rerun.json')
combined = json.load(open(combined_path))
combined['task4_reduced']['AutoGluon'] = res
combined.setdefault('timings_seconds', {})['task4_AutoGluon_v2'] = time.time() - t0
json.dump(combined, open(combined_path, 'w'), indent=1)

print(f'\nDone. Elapsed: {(time.time()-t0)/60:.1f} min. Updated task4 + combined JSON.')
