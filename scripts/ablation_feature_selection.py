"""Feature selection ablation — all tabular WESAD models.

Tests K ∈ {10, 20, 30, None (all)} with mutual_info_classif on wrist features.
Models: LogReg, SVM, RF, ET.
Feature selection is applied strictly within each training fold (no leakage).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from common import load_wesad_tabular, wrist_cols

TOP_K_VALUES = [10, 20, 30, None]

LOGREG_PARAMS = {
    'C': 0.001, 'penalty': 'l1', 'solver': 'saga',
    'max_iter': 1000, 'tol': 1e-4, 'random_state': 42
}
SVM_PARAMS = {
    'C': 0.1, 'kernel': 'poly', 'gamma': 'auto',
    'degree': 3, 'class_weight': 'balanced', 'probability': True,
    'cache_size': 500,
}
RF_PARAMS = {
    'n_estimators': 500, 'max_features': 0.5,
    'min_samples_leaf': 2, 'class_weight': 'balanced',
    'random_state': 42, 'n_jobs': -1,
}
ET_PARAMS = {
    'n_estimators': 500, 'max_features': 0.5,
    'min_samples_leaf': 2, 'class_weight': 'balanced',
    'random_state': 42, 'n_jobs': -1,
}


def select_features(X_tr, y_tr, top_k):
    if top_k is None:
        return list(range(X_tr.shape[1]))
    scores = mutual_info_classif(X_tr, y_tr, random_state=42)
    return list(np.argsort(scores)[::-1][:top_k])


def run_loso_logreg(X_arr, y, groups, top_k):
    logo = LeaveOneGroupOut()
    all_true, all_prob = [], []
    for tr, te in logo.split(X_arr, y, groups):
        X_tr, X_te = X_arr[tr].copy(), X_arr[te].copy()
        y_tr, y_te = y[tr], y[te]
        imp = SimpleImputer(strategy='median')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))
        idx = select_features(X_tr, y_tr, top_k)
        clf = LogisticRegression(**LOGREG_PARAMS)
        try:
            clf.fit(X_tr[:, idx], y_tr)
        except Exception:
            continue
        prob = clf.predict_proba(X_te[:, idx])[:, 1]
        all_true.extend(y_te); all_prob.extend(prob)
    return f1_score(np.array(all_true),
                    (np.array(all_prob) >= 0.5).astype(int),
                    average='macro', zero_division=0)


def run_loso_svm(X_df_orig, y_orig, groups_orig, feat_cols, top_k):
    import pandas as pd
    df_tmp = X_df_orig[feat_cols].copy()
    df_tmp['__label__'] = y_orig
    df_tmp['__subject__'] = groups_orig
    df_s = (df_tmp.groupby(['__subject__', '__label__'])
            .sample(frac=0.4, random_state=42)
            .reset_index(drop=True))
    X_arr = df_s[feat_cols].values.astype(float)
    y_s   = df_s['__label__'].values
    g_s   = df_s['__subject__'].values

    logo = LeaveOneGroupOut()
    all_true, all_prob = [], []
    for tr, te in logo.split(X_arr, y_s, g_s):
        X_tr, X_te = X_arr[tr].copy(), X_arr[te].copy()
        y_tr, y_te = y_s[tr], y_s[te]
        imp = SimpleImputer(strategy='median')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        idx = select_features(X_tr, y_tr, top_k)
        clf = SVC(**SVM_PARAMS)
        try:
            clf.fit(X_tr[:, idx], y_tr)
        except Exception:
            continue
        prob = clf.predict_proba(X_te[:, idx])[:, 1]
        all_true.extend(y_te); all_prob.extend(prob)
    return f1_score(np.array(all_true),
                    (np.array(all_prob) >= 0.5).astype(int),
                    average='macro', zero_division=0)


def run_loso_tree(X_arr, y, groups, top_k, clf_class, clf_params):
    logo = LeaveOneGroupOut()
    all_true, all_prob = [], []
    for tr, te in logo.split(X_arr, y, groups):
        X_tr, X_te = X_arr[tr].copy(), X_arr[te].copy()
        y_tr, y_te = y[tr], y[te]
        imp = SimpleImputer(strategy='mean')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))
        idx = select_features(X_tr, y_tr, top_k)
        clf = clf_class(**clf_params)
        clf.fit(X_tr[:, idx], y_tr)
        prob = clf.predict_proba(X_te[:, idx])[:, 1]
        all_true.extend(y_te); all_prob.extend(prob)
    return f1_score(np.array(all_true),
                    (np.array(all_prob) >= 0.5).astype(int),
                    average='macro', zero_division=0)


if __name__ == '__main__':
    print('Loading WESAD...')
    X_df, y, groups, feat_cols = load_wesad_tabular()
    wcols = wrist_cols(feat_cols)
    X_wrist = X_df[wcols].values.astype(float)
    n_feat = len(wcols)
    print(f'{n_feat} wrist features, {len(np.unique(groups))} subjects\n')

    models = {
        'LogReg': None,
        'SVM':    None,
        'RF':     None,
        'ET':     None,
    }

    results = {m: {} for m in models}

    for k in TOP_K_VALUES:
        k_label = str(k) if k is not None else 'All'
        print(f'{"="*65}')
        print(f'  K = {k_label}')
        print(f'{"="*65}')

        print(f'  LogReg ...', flush=True)
        f1 = run_loso_logreg(X_wrist, y, groups, k)
        results['LogReg'][k_label] = f1
        print(f'  LogReg  K={k_label:3s}  F1 = {f1:.4f}')

        print(f'  SVM ...', flush=True)
        f1 = run_loso_svm(X_df, y, groups, wcols, k)
        results['SVM'][k_label] = f1
        print(f'  SVM     K={k_label:3s}  F1 = {f1:.4f}')

        print(f'  RF ...', flush=True)
        f1 = run_loso_tree(X_wrist, y, groups, k, RandomForestClassifier, RF_PARAMS)
        results['RF'][k_label] = f1
        print(f'  RF      K={k_label:3s}  F1 = {f1:.4f}')

        print(f'  ET ...', flush=True)
        f1 = run_loso_tree(X_wrist, y, groups, k, ExtraTreesClassifier, ET_PARAMS)
        results['ET'][k_label] = f1
        print(f'  ET      K={k_label:3s}  F1 = {f1:.4f}')
        print()

    print(f'\n{"="*65}')
    print(f'  ABLATION SUMMARY — feature selection, WESAD wrist ({n_feat} feat)')
    print(f'{"="*65}')
    header = f"  {'Model':<8}" + ''.join(f'  K={k if k else "All":>3}' for k in TOP_K_VALUES)
    print(header)
    print(f'  {"-"*60}')
    for model, res in results.items():
        row = f'  {model:<8}'
        base = res.get('All', None)
        for k in TOP_K_VALUES:
            k_label = str(k) if k is not None else 'All'
            f1 = res.get(k_label, float('nan'))
            delta = f' ({f1-base:+.3f})' if base is not None and k is not None else ''
            row += f'  {f1:.4f}{delta:<9}'
        print(row)
    print(f'{"="*65}')
