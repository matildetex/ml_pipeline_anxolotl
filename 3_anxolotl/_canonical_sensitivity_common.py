"""
Shared helpers for the ET-1000 / leak-fixed sensitivity re-runs
(window size, calibration window, stride, marker sensitivity).

Every sensitivity script in this batch imports from here so that the
model, normalisation, imputation and LOSO loop are byte-for-byte identical
to 3_anxolotl/friedman_nemenyi_anxolotl.py -- the only thing that varies
between scripts is the experimental variable under test.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale, leak-fixed 2026-07-19
                (3_anxolotl/demographics_analysis.py) -- leading run of
                label=0 windows per subject, NOT a fixed-count reference.
Imputer       : SimpleImputer(strategy='median') -- matches friedman_nemenyi_anxolotl.py
CV            : LOSO (Leave-One-Subject-Out)
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale  # leak-fixed version

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'df_all_1hz_cache.pkl')


def load_cache():
    with open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)


def et1000_factory():
    return ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                 class_weight='balanced', random_state=42, n_jobs=-1)


def build_canonical_xy(df_all, window_sec=60, stride_sec=30, exclude_phase3=True):
    """Mirrors friedman_nemenyi_anxolotl.py lines 47-55 exactly."""
    df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=window_sec, stride_sec=stride_sec)
    n_before = len(df_feat)
    if exclude_phase3:
        df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
    n_after = len(df_feat)
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat['label'].values.astype(int)
    groups = df_feat['subject'].values
    subjects = np.unique(groups)
    return X, y, groups, subjects, feat_cols, n_before, n_after


def run_loso_et1000(X, y, groups, subjects, norm_fn):
    """
    Runs LOSO with ET-1000 + the given normalisation function (must have
    signature norm_fn(X, y, groups) -> X_normalised, same contract as
    baseline_relative_scale). Returns per-subject macro-F1 dict and a
    single pooled out-of-fold AUROC (probabilities of class 1 pooled
    across all 37 test folds, then one roc_auc_score call).
    """
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(norm_fn(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(norm_fn(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = et1000_factory()
        clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y[te], pred, average='macro', zero_division=0)
        proba = clf.predict_proba(X_te_n)
        pos_idx = list(clf.classes_).index(1)
        y_true_all.append(y[te])
        y_prob_all.append(proba[:, pos_idx])
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    try:
        auroc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auroc = float('nan')
    return fold_f1, auroc, y_true_all, y_prob_all


def baseline_relative_scale_capped(X, y, groups, cap=None):
    """
    Same leak-fixed reference-selection rule as baseline_relative_scale
    (demographics_analysis.py: leading run of label==0 windows per subject,
    stopping at the first non-zero label -- never looks past a subject's
    own true baseline period), but optionally caps how many of those
    leading windows are actually used to compute the reference mean/std.

    cap=None -> uses the full leading run (identical to the canonical,
    uncapped baseline_relative_scale).
    cap=k    -> uses only the first k windows of that same leading run.

    This is the calibration-window sensitivity knob for the current
    (adaptive, label-verified) pipeline -- NOT the old fixed-N_BL design,
    which selected the first N chronological windows regardless of label
    and could not adapt to variable baseline lengths.
    """
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
        use_n = n_baseline if cap is None else min(n_baseline, cap)
        baseline_idx = subj_idx[:use_n]
        if len(baseline_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[baseline_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X
