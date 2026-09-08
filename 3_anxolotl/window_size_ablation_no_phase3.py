"""
Window-size ablation, corrected: excludes stress_phase == 3 (weak stress
induction, confirmed by the user as out-of-scope for this analysis).

Same model / normalisation / CV as the original ablation being corrected:
LightGBM (n_estimators=300, max_depth=4, lr=0.1, num_leaves=31,
class_weight='balanced'), LOSO-CV, baseline_relative_scale (BL-norm),
5-channel wrist features (acc_wrist, hr, ibi, rmssd, sdnn), no ECG.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier

import anxolotl_preprocessing as ap

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')


def baseline_relative_scale(X, y, groups):
    """Fixed 2026-07-19: leading-run reference, not every y==0 row — see
    3_anxolotl/demographics_analysis.py for the leak writeup."""
    X = np.array(X, dtype=float).copy()
    y = np.array(y); groups = np.array(groups)
    for subj in np.unique(groups):
        subj_idx = np.where(groups == subj)[0]
        n_baseline = 0
        for v in y[subj_idx]:
            if v == 0:
                n_baseline += 1
            else:
                break
        baseline_idx = subj_idx[:n_baseline]
        if len(baseline_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[baseline_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X


def run_loso(X, y, groups):
    logo = LeaveOneGroupOut()
    fold_f1, y_true_all, y_prob_all = [], [], []
    for tr, te in logo.split(X, y, groups):
        Xtr = baseline_relative_scale(X[tr], y[tr], groups[tr])
        Xte = baseline_relative_scale(X[te], y[te], groups[te])
        imp = SimpleImputer(strategy='mean').fit(Xtr)
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
        clf = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              num_leaves=31, class_weight='balanced',
                              n_jobs=-1, random_state=42, verbose=-1)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_f1.append(f1_score(y[te], pred, average='macro', zero_division=0))
        y_true_all.extend(y[te]); y_prob_all.extend(prob)
    macro_f1 = np.mean(fold_f1)
    sd = np.std(fold_f1)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc = float('nan')
    return macro_f1, sd, auc, fold_f1


print('STEP 1 — Preprocessing all sessions (1Hz, shared across window sizes)…')
pattern_files = sorted(
    f for f in os.listdir(SESSIONS_DIR) if f.startswith('session_') and f.endswith('.csv')
)
session_map = {}
for fn in pattern_files:
    number = fn.replace('session_', '').replace('.csv', '').lstrip('0') or '0'
    sid = f'p{number}'
    session_map[sid] = os.path.join(SESSIONS_DIR, fn)
if 'p6' in session_map:
    del session_map['p6']  # same manual exclusion as process_folder default

df_all, results = ap.preprocess_all_sessions(session_map)
print(f'\ndf_all: {df_all.shape}, subjects={df_all["subject"].nunique()}')
print(f'stress_phase counts:\n{df_all["stress_phase"].value_counts() if "stress_phase" in df_all.columns else "MISSING stress_phase COLUMN"}')

configs = [(30, 15), (60, 30), (90, 45), (120, 60)]
rows = []
fold_detail = {}

for win, stride in configs:
    print(f'\n{"="*60}\nWindow={win}s, stride={stride}s\n{"="*60}')
    df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=win, stride_sec=stride)

    n_before = len(df_feat)
    df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
    n_after = len(df_feat)
    print(f'  Excluded stress_phase==3: {n_before} -> {n_after} windows '
          f'({n_before - n_after} removed)')

    X = df_feat[feat_cols].values.astype(float)
    y = df_feat['label'].values.astype(int)
    groups = df_feat['subject'].values

    n_bl = int((y == 0).sum()); n_st = int((y == 1).sum())
    n_subj = df_feat['subject'].nunique()

    macro_f1, sd, auc, fold_f1 = run_loso(X, y, groups)
    rows.append((win, stride, n_after, n_bl, n_st, n_after / n_subj, macro_f1, sd, auc))
    fold_detail[win] = fold_f1
    print(f'  N_win={n_after}  N_BL={n_bl}  N_ST={n_st}  MacroF1={macro_f1:.4f}  '
          f'SD={sd:.4f}  AUROC={auc:.4f}')

out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                         'window_size_ablation_lgbm_loso_no_phase3.txt')
with open(out_path, 'w') as f:
    f.write("Window-size ablation study — CORRECTED: excludes stress_phase == 3.\n")
    f.write("Fixed model: LightGBM (n_estimators=300, max_depth=4, lr=0.1, num_leaves=31, "
            "class_weight=balanced), LOSO-CV (n=37), BL-norm (baseline_relative_scale), "
            "50% overlap stride, 5-channel wrist features (acc_wrist, hr, ibi, rmssd, sdnn), "
            "11 stats/channel = 55 features.\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Win (s)':>7}  {'Stride':>6}  {'N_win':>6}  {'N_BL':>6}  {'N_ST':>6}  "
            f"{'Win/subj':>8}  {'MacroF1':>8}  {'±SD':>7}  {'AUROC':>7}\n")
    f.write("-" * 80 + "\n")
    for win, stride, n_win, n_bl, n_st, win_subj, macro_f1, sd, auc in rows:
        f.write(f"{win:>7}  {stride:>6}  {n_win:>6}  {n_bl:>6}  {n_st:>6}  "
                f"{win_subj:>8.1f}  {macro_f1:>8.4f}  {sd:>7.4f}  {auc:>7.4f}\n")
    f.write("\n" + "=" * 80 + "\n\n")
    best = max(rows, key=lambda r: r[6])
    f.write(f"Best window size: {best[0]}s (stride {best[1]}s)  MacroF1={best[6]:.4f}\n")

print(f'\nSaved: {out_path}')
for win, fold_f1 in fold_detail.items():
    print(f'Window={win}s: n_folds={len(fold_f1)}  min={min(fold_f1):.4f}  '
          f'median={np.median(fold_f1):.4f}  max={max(fold_f1):.4f}')
