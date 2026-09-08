"""WESAD LOSO — Logistic Regression (wrist features + all features)

EXPLORAÇÃO, NÃO É FONTE DA TESE — confirmado 2026-08-31: a linha LogReg da
Tabela 4.9 (0.80±0.20/0.81/0.95) bate exata com `1_wesad/full/logreg-smote.ipynb`
(grid search/SMOTE), não com este script (que dá 0.816/0.168/0.816/0.912).
Arquivado sem proteção de schema porque não alimenta nenhum número publicado.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from tqdm import tqdm
from common import load_wesad_tabular, wrist_cols, report_metrics, save_results

BEST_PARAMS = {
    'C': 0.001, 'penalty': 'l1', 'solver': 'saga',
    'max_iter': 1000, 'tol': 1e-4, 'random_state': 42
}

def run_loso(X_df, y, groups, feature_cols, label):
    X = X_df[feature_cols].values.astype(float)
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X, y, groups))
    _pbar = tqdm(_splits, desc="LogReg WESAD", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in enumerate(_pbar):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imp = SimpleImputer(strategy='median')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))

        clf = LogisticRegression(**BEST_PARAMS)
        try:
            clf.fit(X_tr, y_tr)
        except Exception as e:
            print(f'  fold {fold+1} error: {e}'); continue

        prob = clf.predict_proba(X_te)[:, 1]
        fold_f1s.append(f1_score(y_te, (prob >= 0.5).astype(int), average='macro', zero_division=0))
        all_true.extend(y_te); all_prob.extend(prob)
        print(f'  fold {fold+1:2d} F1={fold_f1s[-1]:.3f}')

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


if __name__ == '__main__':
    print('Loading WESAD...')
    X_df, y, groups, feat_cols = load_wesad_tabular()
    all_results = []

    print('\n' + '='*65)
    print('  WESAD LOSO — LogReg — WRIST features')
    print('='*65)
    wcols = wrist_cols(feat_cols)
    print(f'  Features: {len(wcols)} wrist')
    r = run_loso(X_df, y, groups, wcols, f'LogReg WESAD — wrist ({len(wcols)} feat)')
    all_results.append(r)

    print('\n' + '='*65)
    print('  WESAD LOSO — LogReg — ALL features')
    print('='*65)
    print(f'  Features: {len(feat_cols)} total')
    r = run_loso(X_df, y, groups, feat_cols, f'LogReg WESAD — all ({len(feat_cols)} feat)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_logreg.json'))
