"""
WESAD classical-model rerun -- 2026-07-24, real results for Tables 1.1/1.2/1.3/1.8.

TASK 1: LogReg/SVM/RF/XGBoost/LightGBM x {COM,SEM} SMOTE, FULL feature set (137,
        old schema), LOSO-CV (15 folds), FIXED hyperparameters (chapter's own).
TASK 2: Soft-voting ensembles (3-model LGBM+XGB+RF, 5-model all) x TOP_K in
        {10,20,30,All} via mutual_info_classif (Information Gain), FULL set only.
TASK 3: ROC-curve figure (two panels: without/with SMOTE) from Task 1 results.
TASK 4: RF/XGBoost/AutoGluon on the REDUCED set (55 feat, CURRENT schema --
        confirmed by user), no SMOTE (matches wesad_55plus_ecg_loso.txt convention).

Hyperparameters are FIXED (no search) -- taken verbatim from the chapter / from
scripts/xai.py's train_all_models_std() where they coincide (XGBoost, LightGBM).

Practical deviation, flagged explicitly (not silent): SVM runs on a 40% stratified
subsample, matching the established convention already used in this exact repo
for WESAD SVM (results/wesad_svm.json, 1_wesad/classical/support-vector-machine.ipynb
both do this) -- full-data poly-kernel SVM at n~27k is computationally impractical.
The 5-model ensemble (which includes SVM) runs on that same subsample for internal
consistency; the 3-model ensemble (no SVM) runs on the full data.
"""
import pickle, os, json, time, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, roc_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

T0 = time.time()
ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TIMINGS = {}
def tic(): return time.time()
def toc(t0, label):
    dt = time.time() - t0
    TIMINGS[label] = dt
    print(f'  [{label}] {dt/60:.1f} min')
    return dt

# ---------------------------------------------------------------------------
# Fixed hyperparameters (chapter's own; XGB/LGBM match scripts/xai.py exactly)
# ---------------------------------------------------------------------------
def make_logreg():
    return LogisticRegression(penalty='l1', C=0.001, solver='saga', max_iter=1000, random_state=42)

def make_svm():
    return SVC(kernel='poly', degree=3, C=0.1, gamma='scale', class_weight='balanced',
               probability=True, random_state=42)

def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=10, max_features='sqrt',
                                   random_state=42, n_jobs=-1)

def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          eval_metric='logloss', random_state=42, n_jobs=-1)

def make_lgbm():
    return LGBMClassifier(n_estimators=305, max_depth=4, learning_rate=0.195, num_leaves=37,
                           random_state=42, verbose=-1, n_jobs=-1)

MODEL_BUILDERS = {
    'LogisticRegression': make_logreg,
    'SVM': make_svm,
    'RandomForest': make_rf,
    'XGBoost': make_xgb,
    'LightGBM': make_lgbm,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print('Loading WESAD FULL (old schema, 137 feat, archived snapshot)...')
with open(os.path.join(ROOT, 'processed_data/wesad_archive/processed_data_137feat_snapshot.pkl'), 'rb') as f:
    d_full = pickle.load(f)
X_full = d_full['X'] if isinstance(d_full['X'], pd.DataFrame) else pd.DataFrame(d_full['X'], columns=d_full['feature_cols'])
y_full = pd.Series(np.array(d_full['y']))
groups_full = np.array(d_full['groups'])
feat_cols_full = list(d_full['feature_cols'])
print(f'  FULL: {X_full.shape}, subjects={len(np.unique(groups_full))}')

print('Loading WESAD REDUCED (current schema, 55 feat, for Task 4)...')
with open(os.path.join(ROOT, 'processed_data/wesad/processed_data.pkl'), 'rb') as f:
    d_cur = pickle.load(f)
X_cur_all = d_cur['X'] if isinstance(d_cur['X'], pd.DataFrame) else pd.DataFrame(d_cur['X'], columns=d_cur['feature_cols'])
y_cur = pd.Series(np.array(d_cur['y']))
groups_cur = np.array(d_cur['groups'])
reduced_cols = [c for c in d_cur['feature_cols'] if any(
    c.startswith(sig) for sig in ['acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_'])]
X_reduced = X_cur_all[reduced_cols]
print(f'  REDUCED: {X_reduced.shape} ({len(reduced_cols)} feat), subjects={len(np.unique(groups_cur))}')

logo = LeaveOneGroupOut()

def stratified_subsample(X, y, groups, frac=0.4, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y.values == 0)[0]
    idx1 = np.where(y.values == 1)[0]
    n0, n1 = int(len(idx0) * frac), int(len(idx1) * frac)
    sel = np.sort(np.concatenate([rng.choice(idx0, n0, replace=False),
                                   rng.choice(idx1, n1, replace=False)]))
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True), groups[sel]

X_svm, y_svm, groups_svm = stratified_subsample(X_full, y_full, groups_full, frac=0.4)
print(f'  SVM subsample (40%, stratified): {X_svm.shape}')

# ---------------------------------------------------------------------------
# LOSO-CV runner
# ---------------------------------------------------------------------------
def run_loso(model_builder, X, y, groups, use_smote, feature_subset=None):
    Xs = X[feature_subset] if feature_subset is not None else X
    all_true, all_pred, all_prob = [], [], []
    fold_f1s, fold_mccs = [], []
    for train_idx, test_idx in logo.split(Xs, y, groups):
        X_train, X_test = Xs.iloc[train_idx], Xs.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        if use_smote:
            X_train_p, y_train = SMOTE(random_state=42).fit_resample(X_train_p, y_train)
        model = model_builder()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_true.extend(y_test); all_pred.extend(y_pred); all_prob.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    return dict(
        y_test=all_true.tolist(), y_pred=all_pred.tolist(), y_prob=all_prob.tolist(),
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        fold_f1s=[float(v) for v in fold_f1s],
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        precision_macro=float(precision_score(all_true, all_pred, average='macro', zero_division=0)),
        recall_macro=float(recall_score(all_true, all_pred, average='macro', zero_division=0)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        mcc=float(np.mean(fold_mccs)),
        confusion_matrix=confusion_matrix(all_true, all_pred).tolist(),
    )

# ===========================================================================
# TASK 1 -- classical models, FULL set, COM/SEM SMOTE
# ===========================================================================
print('\n' + '=' * 70)
print('TASK 1 -- Classical models, FULL (137 feat), COM/SEM SMOTE')
print('=' * 70)
task1_results = {}
for mname, builder in MODEL_BUILDERS.items():
    Xd, yd, gd = (X_svm, y_svm, groups_svm) if mname == 'SVM' else (X_full, y_full, groups_full)
    for use_smote in [True, False]:
        label = f"{mname}_{'smote' if use_smote else 'no_smote'}"
        print(f'\n-- {label} --')
        t = tic()
        try:
            res = run_loso(builder, Xd, yd, gd, use_smote)
            print(f"   MacroF1={res['macro_f1']:.4f} Acc={res['accuracy']:.4f} "
                  f"BalAcc={res['balanced_accuracy']:.4f} AUC={res['roc_auc']:.4f} MCC={res['mcc']:.4f}")
            task1_results[label] = res
        except Exception as e:
            print(f'   ERROR: {type(e).__name__}: {e}')
            task1_results[label] = {'error': f'{type(e).__name__}: {e}'}
        toc(t, f'task1_{label}')

with open(os.path.join(RESULTS_DIR, 'wesad_classical_rerun_task1.json'), 'w') as f:
    json.dump(task1_results, f)
print('\nTask 1 saved.')

# ===========================================================================
# TASK 2 -- IG ensemble ablation, FULL set only
# ===========================================================================
print('\n' + '=' * 70)
print('TASK 2 -- Soft-voting ensembles x TOP_K Information Gain, FULL set')
print('=' * 70)
task2_results = {}

def run_ensemble_ig(model_names, X, y, groups, top_k):
    all_true, all_pred, all_prob = [], [], []
    fold_f1s, fold_mccs = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        if top_k != 'All':
            mi = mutual_info_classif(X_train_p, y_train, random_state=42)
            top_idx = np.argsort(mi)[::-1][:top_k]
            X_train_p = X_train_p[:, top_idx]
            X_test_p = X_test_p[:, top_idx]
        estimators = [(n, MODEL_BUILDERS[n]()) for n in model_names]
        ens = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ens.fit(X_train_p, y_train)
        y_pred = ens.predict(X_test_p)
        y_prob = ens.predict_proba(X_test_p)[:, 1]
        all_true.extend(y_test); all_pred.extend(y_pred); all_prob.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_mccs.append(matthews_corrcoef(y_test, y_pred))
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    return dict(
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        precision_macro=float(precision_score(all_true, all_pred, average='macro', zero_division=0)),
        recall_macro=float(recall_score(all_true, all_pred, average='macro', zero_division=0)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        mcc=float(np.mean(fold_mccs)),
    )

ENSEMBLE_3 = ['LightGBM', 'XGBoost', 'RandomForest']
ENSEMBLE_5 = ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost', 'LightGBM']
TOP_KS = [10, 20, 30, 'All']

for ens_name, models, (Xd, yd, gd) in [
    ('3-model (LGBM+XGB+RF)', ENSEMBLE_3, (X_full, y_full, groups_full)),
    ('5-model (all)',          ENSEMBLE_5, (X_svm, y_svm, groups_svm)),
]:
    for k in TOP_KS:
        label = f'{ens_name} | TOP_{k}'
        print(f'\n-- {label} --')
        t = tic()
        try:
            res = run_ensemble_ig(models, Xd, yd, gd, k)
            print(f"   MacroF1={res['macro_f1']:.4f} Acc={res['accuracy']:.4f} AUC={res['roc_auc']:.4f}")
            task2_results[label] = res
        except Exception as e:
            print(f'   ERROR: {type(e).__name__}: {e}')
            task2_results[label] = {'error': f'{type(e).__name__}: {e}'}
        toc(t, f'task2_{ens_name}_{k}')

with open(os.path.join(RESULTS_DIR, 'wesad_classical_rerun_task2.json'), 'w') as f:
    json.dump(task2_results, f)
print('\nTask 2 saved.')

# ===========================================================================
# TASK 3 -- ROC curves figure
# ===========================================================================
print('\n' + '=' * 70)
print('TASK 3 -- ROC curves figure')
print('=' * 70)
t = tic()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, smote_flag, title in [(axes[0], False, 'Without SMOTE'), (axes[1], True, 'With SMOTE')]:
    for mname in MODEL_BUILDERS:
        label = f"{mname}_{'smote' if smote_flag else 'no_smote'}"
        res = task1_results.get(label, {})
        if 'error' in res: continue
        yt, yp = np.array(res['y_test']), np.array(res['y_prob'])
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, label=f"{mname} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title); ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('ROC Curves -- WESAD Classical Models (FULL, 137 feat)', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig6_roc_wesad_classical_rerun.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
toc(t, 'task3_roc_figure')
print('Saved figures/fig6_roc_wesad_classical_rerun.png')

# ===========================================================================
# TASK 4 -- REDUCED set (55 feat, current schema): RF, XGBoost, AutoGluon
# ===========================================================================
print('\n' + '=' * 70)
print('TASK 4 -- REDUCED set (55 feat, current schema): RF, XGBoost, AutoGluon')
print('=' * 70)
task4_results = {}

for mname, builder in [('RandomForest', make_rf), ('XGBoost', make_xgb)]:
    print(f'\n-- {mname} (reduced, no SMOTE) --')
    t = tic()
    try:
        res = run_loso(builder, X_reduced, y_cur, groups_cur, use_smote=False)
        print(f"   MacroF1={res['macro_f1']:.4f} Acc={res['accuracy']:.4f} AUC={res['roc_auc']:.4f}")
        task4_results[mname] = res
    except Exception as e:
        print(f'   ERROR: {type(e).__name__}: {e}')
        task4_results[mname] = {'error': f'{type(e).__name__}: {e}'}
    toc(t, f'task4_{mname}')

print('\n-- AutoGluon (reduced, no SMOTE, good_quality, 120s/fold) --')
t = tic()
try:
    from autogluon.tabular import TabularPredictor
    all_true, all_pred, all_prob, fold_f1s = [], [], [], []
    Xr_df = X_reduced.reset_index(drop=True)
    yr = y_cur.reset_index(drop=True)
    for fi, (train_idx, test_idx) in enumerate(logo.split(Xr_df, yr, groups_cur)):
        train_df = Xr_df.iloc[train_idx].copy(); train_df['label'] = yr.iloc[train_idx].values
        test_df = Xr_df.iloc[test_idx]
        y_test = yr.iloc[test_idx].values
        predictor = TabularPredictor(label='label', eval_metric='f1_macro', verbosity=0,
                                      path=f'/tmp/ag_wesad_reduced_fold{fi}').fit(
            train_df, time_limit=120, presets='good_quality')
        preds = predictor.predict(test_df).values
        proba = predictor.predict_proba(test_df)[1].values
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_test); all_pred.extend(preds); all_prob.extend(proba)
        print(f'   fold {fi+1:2d}  F1={f1:.4f}')
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    task4_results['AutoGluon'] = dict(
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        accuracy=float(accuracy_score(all_true, all_pred)),
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        precision_macro=float(precision_score(all_true, all_pred, average='macro', zero_division=0)),
        recall_macro=float(recall_score(all_true, all_pred, average='macro', zero_division=0)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        mcc=float(matthews_corrcoef(all_true, all_pred)),
    )
    print(f"   FINAL MacroF1={task4_results['AutoGluon']['macro_f1']:.4f} "
          f"AUC={task4_results['AutoGluon']['roc_auc']:.4f}")
except Exception as e:
    print(f'   ERROR: {type(e).__name__}: {e}')
    task4_results['AutoGluon'] = {'error': f'{type(e).__name__}: {e}'}
toc(t, 'task4_AutoGluon')

with open(os.path.join(RESULTS_DIR, 'wesad_classical_rerun_task4.json'), 'w') as f:
    json.dump(task4_results, f)
print('\nTask 4 saved.')

# ===========================================================================
# Combined output + timing
# ===========================================================================
combined = {'task1_full_classical': task1_results, 'task2_ig_ensemble': task2_results,
            'task4_reduced': task4_results, 'timings_seconds': TIMINGS}
with open(os.path.join(RESULTS_DIR, 'wesad_classical_rerun.json'), 'w') as f:
    json.dump(combined, f, indent=1)

with open(os.path.join(RESULTS_DIR, 'wesad_classical_rerun.txt'), 'w') as f:
    f.write('WESAD CLASSICAL RERUN -- 2026-07-24\n')
    f.write('=' * 70 + '\n\n')
    f.write('TASK 1 -- FULL (137 feat), fixed hyperparams\n')
    for k, v in task1_results.items():
        if 'error' in v: f.write(f'{k}: ERROR {v["error"]}\n'); continue
        f.write(f"{k:30} MacroF1={v['macro_f1']:.4f}±{v['macro_f1_std']:.4f}  "
                f"Acc={v['accuracy']:.4f}  BalAcc={v['balanced_accuracy']:.4f}  "
                f"Prec={v['precision_macro']:.4f}  Rec={v['recall_macro']:.4f}  "
                f"AUC={v['roc_auc']:.4f}  MCC={v['mcc']:.4f}\n")
    f.write('\nTASK 2 -- IG ensemble ablation, FULL set\n')
    for k, v in task2_results.items():
        if 'error' in v: f.write(f'{k}: ERROR {v["error"]}\n'); continue
        f.write(f"{k:30} MacroF1={v['macro_f1']:.4f}  Acc={v['accuracy']:.4f}  "
                f"BalAcc={v['balanced_accuracy']:.4f}  AUC={v['roc_auc']:.4f}  MCC={v['mcc']:.4f}\n")
    f.write('\nTASK 4 -- REDUCED (55 feat, current schema)\n')
    for k, v in task4_results.items():
        if 'error' in v: f.write(f'{k}: ERROR {v["error"]}\n'); continue
        f.write(f"{k:30} MacroF1={v['macro_f1']:.4f}  Acc={v['accuracy']:.4f}  "
                f"BalAcc={v['balanced_accuracy']:.4f}  AUC={v['roc_auc']:.4f}  MCC={v['mcc']:.4f}\n")
    f.write('\nTIMINGS (minutes)\n')
    for k, v in TIMINGS.items():
        f.write(f'{k:30} {v/60:.1f}\n')
    f.write(f'\nTOTAL: {(time.time()-T0)/60:.1f} min\n')

print(f'\n\nALL DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
print('Wrote: results/wesad_classical_rerun.json, .txt, _task1/2/4.json, figures/fig6_roc_wesad_classical_rerun.png')
