"""
Section 1.2 -- WESAD wrist-only, 55-feature set (5 wrist channels x 11 stats,
EXCLUDING eda_wrist/temp_wrist). Self-contained job: everything this script
writes lives under this same directory (results/section1_2_wesad_wrist55_2026-07-26/).
Nothing outside it is touched -- Section 1.1 (WESAD-Full 137), 1.3 and 1.4 are
untouched, and no pre-existing figures/results are overwritten.

Data-source validation (printed before anything trains, per instructions):
  - processed_data/wesad/processed_data.pkl is the CURRENT (post-2026-07-17
    regen) WESAD schema: 130 total features, 25807 windows, 15 subjects.
  - The 5-channel/11-stat reduced set (acc_wrist, hr, ibi, rmssd, sdnn) is
    selected by column-prefix and asserted == 55 columns before any training.
  - The OLD schema snapshot (processed_data/wesad_archive/processed_data_137feat_snapshot.pkl,
    8 stats/channel) is NOT used here -- it would give 7*8=56 wrist columns,
    which is the "56" the request explicitly says NOT to use.

Preprocessing, matched to Section 1.1's own protocol (1_wesad/classical/
wesad_classical_rerun.py Task 1, "SEM SMOTE" table): per-LOSO-fold
StandardScaler + SimpleImputer(mean) fit on the train fold only, applied to
the test fold; no SMOTE.

Model hyperparameters: LogReg/SVM/RF/XGBoost/LightGBM taken verbatim from
1_wesad/classical/wesad_classical_rerun.py (the chapter's own fixed values).
ExtraTrees taken verbatim from scripts/xai.py / scripts/wesad_shap_reverification.py
(the exact model SHAP is computed on, for consistency between the standalone-ET
row and the SHAP figures). HistGradientBoosting taken from the canonical
Anxolotl HGB config used throughout 3_anxolotl/ (max_iter=300, max_depth=6,
learning_rate=0.1).

Deviation, flagged (not silent): SVM (poly kernel, O(n^2-n^3)) runs on the
same 40% stratified subsample convention already established in this repo
for WESAD SVM (results/wesad_svm.json, wesad_classical_rerun.py) -- full
25807-row poly-kernel SVM is impractical. All other 6 models run on the
full 25807 windows.
"""
import pickle, os, json, time, warnings
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
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

T0 = time.time()
ROOT    = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
OUT_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_2026-07-26')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

# ===========================================================================
# STEP 0 -- validate data source BEFORE training anything
# ===========================================================================
w('='*78)
w('STEP 0 -- DATA SOURCE VALIDATION')
w('='*78)

PKL_PATH = os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl')
mtime = pd.Timestamp(os.path.getmtime(PKL_PATH), unit='s')
w(f'Candidate file: {os.path.relpath(PKL_PATH, ROOT)}')
w(f'  mtime: {mtime}')

with open(PKL_PATH, 'rb') as f:
    d = pickle.load(f)

X_all = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
y_all = pd.Series(np.asarray(d['y']))
groups_all = pd.Series(np.asarray(d['groups']))
feat_cols_all = list(d['feature_cols'])

w(f'  shape: {X_all.shape[0]} rows x {X_all.shape[1]} cols')
w(f'  n_subjects: {y_all.groupby(groups_all).ngroups if False else len(np.unique(groups_all))}')
w(f'  class balance (y): {dict(zip(*np.unique(y_all, return_counts=True)))}')
w()

OLD_PKL = os.path.join(ROOT, 'processed_data', 'wesad_archive', 'processed_data_137feat_snapshot.pkl')
w(f'Other candidate considered and REJECTED: {os.path.relpath(OLD_PKL, ROOT)}')
w(f'  This is the OLD (pre-2026-07-17) 137-feature/8-stats-per-channel schema.')
w(f'  Its wrist subset would be 7 channels x 8 stats = 56 columns -- the "56" the')
w(f'  request explicitly excludes. Not used for anything in this script.')
w()

WRIST_PREFIXES = ('acc_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_')
reduced_cols = [c for c in feat_cols_all if c.startswith(WRIST_PREFIXES)]

w(f'Reduced (Section 1.2) column selection: prefixes {WRIST_PREFIXES}')
w(f'  -> {len(reduced_cols)} columns selected from the CURRENT ({X_all.shape[1]}-col) schema.')
assert len(reduced_cols) == 55, (
    f'STOP: expected exactly 55 columns, got {len(reduced_cols)}. '
    f'Selected: {reduced_cols}'
)
w('  ASSERTION PASSED: exactly 55 columns.')
w()
w('Full list of the 55 selected feature names:')
for i, c in enumerate(reduced_cols, 1):
    w(f'  {i:2d}. {c}')
w()

X_reduced = X_all[reduced_cols].reset_index(drop=True)
y = y_all.reset_index(drop=True)
groups = groups_all.reset_index(drop=True)
n_nan = int(X_reduced.isna().sum().sum())
w(f'X_reduced shape: {X_reduced.shape}, NaN count: {n_nan} '
  f'(handled by per-fold SimpleImputer(mean), see preprocessing note below).')
w()

w('CONCLUSION: current-schema file confirmed, 55/55 columns match spec exactly. '
  'Proceeding to training.')
w()

# ===========================================================================
# STEP 0b -- confirm preprocessing/normalization (Section 1.1's own protocol)
# ===========================================================================
w('='*78)
w('STEP 0b -- NORMALIZATION / PREPROCESSING (matched to Section 1.1)')
w('='*78)
w('Section 1.1 (1_wesad/classical/wesad_classical_rerun.py, Task 1, "SEM SMOTE" '
  'table = the chapter\'s Table 1.1) uses, per LOSO fold:')
w('  StandardScaler().fit_transform(X_train) -> SimpleImputer(strategy="mean").fit_transform(...)')
w('  then the SAME fitted scaler+imputer .transform()-ed onto X_test. No SMOTE.')
w('This script uses the identical per-fold fit-on-train-only StandardScaler + '
  'SimpleImputer(mean) protocol, no SMOTE, for every one of the 7 models below.')
w()

# ===========================================================================
# STEP 1 -- 7 standalone models, LOSO-CV
# ===========================================================================
w('='*78)
w('STEP 1 -- 7 STANDALONE MODELS, LOSO-CV (15 folds), 55-feature wrist set')
w('='*78)

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

def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.1,
                                           random_state=42)

def make_et():
    return ExtraTreesClassifier(n_estimators=1000, max_features=0.5, class_weight='balanced',
                                 random_state=42, n_jobs=-1)

MODEL_BUILDERS = {
    'LogisticRegression': make_logreg,
    'SVM': make_svm,
    'RandomForest': make_rf,
    'XGBoost': make_xgb,
    'LightGBM': make_lgbm,
    'HistGradientBoosting': make_hgb,
    'ExtraTrees': make_et,
}

logo = LeaveOneGroupOut()

def stratified_subsample(X, y, groups, frac=0.4, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y.values == 0)[0]
    idx1 = np.where(y.values == 1)[0]
    n0, n1 = int(len(idx0) * frac), int(len(idx1) * frac)
    sel = np.sort(np.concatenate([rng.choice(idx0, n0, replace=False),
                                   rng.choice(idx1, n1, replace=False)]))
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True), groups.iloc[sel].reset_index(drop=True)

X_svm, y_svm, groups_svm = stratified_subsample(X_reduced, y, groups, frac=0.4)
w(f'SVM deviation: running on 40% stratified subsample ({X_svm.shape[0]} of '
  f'{X_reduced.shape[0]} windows) -- matches this repo\'s established WESAD-SVM '
  f'convention (poly-kernel SVC does not scale to n~26k). All other 6 models '
  f'use the full {X_reduced.shape[0]} windows.')
w()

def run_loso(model_builder, X, y, groups):
    all_true, all_pred, all_prob = [], [], []
    fold_f1s = []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler(); imputer = SimpleImputer(strategy='mean')
        X_train_p = imputer.fit_transform(scaler.fit_transform(X_train))
        X_test_p = imputer.transform(scaler.transform(X_test))
        model = model_builder()
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]
        all_true.extend(y_test); all_pred.extend(y_pred); all_prob.extend(y_prob)
        fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    all_true, all_pred, all_prob = np.array(all_true), np.array(all_pred), np.array(all_prob)
    return dict(
        macro_f1=float(np.mean(fold_f1s)), macro_f1_std=float(np.std(fold_f1s)),
        fold_f1s=[float(v) for v in fold_f1s],
        balanced_accuracy=float(balanced_accuracy_score(all_true, all_pred)),
        accuracy=float(accuracy_score(all_true, all_pred)),
        roc_auc=float(roc_auc_score(all_true, all_prob)),
        n_folds=len(fold_f1s),
    )

results = {}
for mname, builder in MODEL_BUILDERS.items():
    Xd, yd, gd = (X_svm, y_svm, groups_svm) if mname == 'SVM' else (X_reduced, y, groups)
    w(f'-- {mname} --')
    t0 = time.time()
    try:
        res = run_loso(builder, Xd, yd, gd)
        dt = time.time() - t0
        res['runtime_sec'] = dt
        results[mname] = res
        w(f'   MacroF1={res["macro_f1"]:.4f}±{res["macro_f1_std"]:.4f}  '
          f'BalAcc={res["balanced_accuracy"]:.4f}  AUROC={res["roc_auc"]:.4f}  '
          f'({dt:.1f}s, {res["n_folds"]} folds)')
    except Exception as e:
        dt = time.time() - t0
        w(f'   ERROR after {dt:.1f}s: {type(e).__name__}: {e}')
        results[mname] = {'error': f'{type(e).__name__}: {e}'}
w()

with open(os.path.join(OUT_DIR, 'metrics_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ===========================================================================
# STEP 2 -- summary table, sorted by Macro F1
# ===========================================================================
w('='*78)
w('STEP 2 -- SUMMARY TABLE (sorted by Macro F1 desc) -- Table 1.7 draft')
w('='*78)
ok_rows = [(m, r) for m, r in results.items() if 'error' not in r]
ok_rows.sort(key=lambda kv: kv[1]['macro_f1'], reverse=True)
err_rows = [(m, r) for m, r in results.items() if 'error' in r]

header = f'{"Model":24s} {"Macro F1":>16s} {"BalAcc":>8s} {"AUROC":>8s}'
w(header); w('-' * len(header))
table_lines = [header, '-' * len(header)]
for m, r in ok_rows:
    line = (f'{m:24s} {r["macro_f1"]:.4f} ± {r["macro_f1_std"]:.4f}  '
            f'{r["balanced_accuracy"]:.4f} {r["roc_auc"]:.4f}')
    w(line); table_lines.append(line)
for m, r in err_rows:
    line = f'{m:24s} ERROR: {r["error"]}'
    w(line); table_lines.append(line)
w()

with open(os.path.join(OUT_DIR, 'table_1_7_draft.txt'), 'w') as f:
    f.write('\n'.join(table_lines) + '\n')

# ===========================================================================
# STEP 3 -- SHAP (TreeSHAP) on ExtraTrees, 55-feature set
# ===========================================================================
w('='*78)
w('STEP 3 -- SHAP (TreeSHAP) ON EXTRATREES, 55-FEATURE WRIST SET')
w('='*78)

if 'error' in results.get('ExtraTrees', {'error': 'not run'}):
    w(f'ExtraTrees LOSO run failed above -- attempting a fresh pooled fit for SHAP anyway.')

w('Fitting ET on the FULL pooled dataset (all 25807 windows, StandardScaler+'
  'SimpleImputer(mean), no SMOTE) -- matches scripts/xai.py / '
  'scripts/wesad_shap_reverification.py convention: SHAP explains a single '
  'pooled model, not per-LOSO-fold models.')

X_raw = X_reduced.values.astype(float)
sc = StandardScaler(); imp = SimpleImputer(strategy='mean')
Xn = imp.fit_transform(sc.fit_transform(X_raw))
yv = y.values

clf = make_et()
clf.fit(Xn, yv)
w(f'  ET fit on {Xn.shape[0]} rows x {Xn.shape[1]} cols.')

def stratified_sample(X, y, n=1200, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx], idx

Xs, ys, idx = stratified_sample(Xn, yv, n=1200)
explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
imp_vals = np.abs(sv).mean(axis=0)
w(f'  TreeSHAP computed on a {Xs.shape[0]}-row stratified sample (seed=42).')
w()

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'HR':    lambda c: c.startswith('hr_'),
    'IBI':   lambda c: c.startswith('ibi'),
    'RMSSD': lambda c: c.startswith('rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn'),
}
def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col): return name
    return 'OTHER'

g = {}
for col, v in zip(reduced_cols, imp_vals):
    k = assign_group(col); g[k] = g.get(k, 0) + v
order = sorted(g, key=g.get, reverse=True)

w('Signal-group SHAP importance (sum of mean|SHAP|), replacing Figure 1.3b:')
for rank, k in enumerate(order, 1):
    w(f'  {rank}. {k:6s} {g[k]:.4f}')
w()

top20 = np.argsort(imp_vals)[::-1][:20]
w('Top 20 individual features (mean|SHAP|):')
for rank, i in enumerate(top20, 1):
    w(f'  {rank:2d}. {reduced_cols[i]:22s} {imp_vals[i]:.6f}')
w()

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})

groups_fig = os.path.join(FIG_DIR, 'wesad_wrist55_ET_shap_groups.png')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(order, [g[k] for k in order], color=plt.cm.tab10(np.linspace(0, 1, len(order))), alpha=0.85)
ax.set_ylabel('Summed |SHAP| Importance')
ax.set_title('Signal-Group SHAP Importance — WESAD wrist-55, ExtraTrees\n(replaces Figure 1.3b)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(groups_fig, dpi=200, bbox_inches='tight'); plt.close()
w(f'Saved: {os.path.relpath(groups_fig, ROOT)}')

beeswarm_fig = os.path.join(FIG_DIR, 'wesad_wrist55_ET_shap_beeswarm.png')
try:
    expl = shap.Explanation(values=sv, data=Xs, feature_names=list(reduced_cols))
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl, max_display=20, show=False)
    plt.title('SHAP Beeswarm — WESAD wrist-55, ExtraTrees\n(replaces Figure 1.4)', pad=10)
    plt.tight_layout()
    plt.savefig(beeswarm_fig, dpi=200, bbox_inches='tight'); plt.close()
    w(f'Saved: {os.path.relpath(beeswarm_fig, ROOT)}')
except Exception as e:
    plt.close()
    w(f'  [beeswarm FAILED] {type(e).__name__}: {e}')
w()

w('='*78)
w(f'DONE. Total elapsed: {(time.time()-T0)/60:.1f} min')
w('='*78)

with open(os.path.join(OUT_DIR, 'full_log.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nAll output under: {os.path.relpath(OUT_DIR, ROOT)}/')
