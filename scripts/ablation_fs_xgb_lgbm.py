"""Feature selection ablation — XGBoost and LightGBM individually.

Tests K ∈ {10, 20, 30, None} with mutual_info_classif on wrist features.
Feature selection applied strictly within each fold (no leakage).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from common import load_wesad_tabular, wrist_cols

TOP_K_VALUES = [10, 20, 30, None]

XGB_PARAMS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
    'n_jobs': -1, 'random_state': 42, 'eval_metric': 'mlogloss',
    'use_label_encoder': False,
}
LGBM_PARAMS = {
    'n_estimators': 305, 'max_depth': 4, 'learning_rate': 0.195,
    'num_leaves': 37, 'min_child_samples': 30,
    'subsample': 0.6537551310524808, 'colsample_bytree': 0.8522263467378263,
    'n_jobs': -1, 'random_state': 42, 'verbose': -1,
}


def select_features(X_tr, y_tr, top_k):
    if top_k is None:
        return list(range(X_tr.shape[1]))
    scores = mutual_info_classif(X_tr, y_tr, random_state=42)
    return list(np.argsort(scores)[::-1][:top_k])


def run_loso(X_arr, y, groups, top_k, clf_class, clf_params):
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

    results = {'XGBoost': {}, 'LightGBM': {}}

    for k in TOP_K_VALUES:
        k_label = str(k) if k is not None else 'All'
        print(f'{"="*60}  K = {k_label}')

        print(f'  XGBoost ...', flush=True)
        f1 = run_loso(X_wrist, y, groups, k, XGBClassifier, XGB_PARAMS)
        results['XGBoost'][k_label] = f1
        print(f'  XGBoost  K={k_label:3s}  F1 = {f1:.4f}')

        print(f'  LightGBM ...', flush=True)
        f1 = run_loso(X_wrist, y, groups, k, LGBMClassifier, LGBM_PARAMS)
        results['LightGBM'][k_label] = f1
        print(f'  LightGBM K={k_label:3s}  F1 = {f1:.4f}')
        print()

    print(f'\n{"="*65}')
    print(f'  ABLATION SUMMARY — XGBoost + LightGBM, WESAD wrist ({n_feat} feat)')
    print(f'{"="*65}')
    header = f"  {'Model':<10}" + ''.join(f'  K={k if k else "All":>3}' for k in TOP_K_VALUES)
    print(header)
    for model, res in results.items():
        base = res.get('All', None)
        row = f'  {model:<10}'
        for k in TOP_K_VALUES:
            k_label = str(k) if k is not None else 'All'
            f1 = res.get(k_label, float('nan'))
            delta = f'({f1-base:+.3f})' if base is not None and k is not None else ''
            row += f'  {f1:.4f}{delta:<8}'
        print(row)
    print(f'{"="*65}')
