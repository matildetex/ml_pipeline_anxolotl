"""WESAD LOSO — SVM (wrist + all features). Uses 40% sample for tractability.

EXPLORAÇÃO, NÃO É FONTE DA TESE — confirmado 2026-08-31: a linha SVM da Tabela 4.9
vem de `1_wesad/full/svm-wesad-full.ipynb` (avaliação a 100% dos dados com os
hiperparâmetros vindos do tuning em `svm-wesad-gridsearch-subsample.ipynb`),
não deste script (sem SMOTE, dá 0.765/0.199/0.785/0.874). Arquivado sem
proteção de schema porque não alimenta nenhum número publicado.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from common import load_wesad_tabular, wrist_cols, report_metrics, save_results

BEST_PARAMS = {'C': 0.1, 'kernel': 'poly', 'gamma': 'auto',
               'degree': 3, 'class_weight': 'balanced', 'probability': True}


def run_loso(X_df, y, groups, feature_cols, label):
    # 40% stratified sample per subject (same as original notebook, keeps SVM tractable)
    import pandas as pd
    df_tmp = X_df[feature_cols].copy()
    df_tmp['__label__'] = y
    df_tmp['__subject__'] = groups
    df_s = df_tmp.groupby(['__subject__', '__label__']).sample(frac=0.4, random_state=42).reset_index(drop=True)
    X = df_s[feature_cols].values.astype(float)
    y_s = df_s['__label__'].values
    g_s = df_s['__subject__'].values

    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X, y_s, g_s))
    _pbar = tqdm(_splits, desc="SVM WESAD", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in enumerate(_pbar):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_s[tr], y_s[te]

        imp = SimpleImputer(strategy='median')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))

        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

        clf = SVC(**BEST_PARAMS, cache_size=500)
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
    print('  WESAD LOSO — SVM — WRIST features')
    print('='*65)
    wcols = wrist_cols(feat_cols)
    print(f'  Features: {len(wcols)} wrist  (40% sample)')
    r = run_loso(X_df, y, groups, wcols, f'SVM WESAD — wrist ({len(wcols)} feat)')
    all_results.append(r)

    print('\n' + '='*65)
    print('  WESAD LOSO — SVM — ALL features')
    print('='*65)
    print(f'  Features: {len(feat_cols)} total  (40% sample)')
    r = run_loso(X_df, y, groups, feat_cols, f'SVM WESAD — all ({len(feat_cols)} feat)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_svm.json'))
