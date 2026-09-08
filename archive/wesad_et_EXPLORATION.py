"""WESAD LOSO — ExtraTrees (wrist 56 feat + all 137 feat)

EXPLORAÇÃO, NÃO É FONTE DA TESE — confirmado 2026-08-31: a Tabela 4.9 (WESAD-Full)
não tem linha de ExtraTrees isolado, e os números que este script produz para
LogReg/RF/XGBoost/LightGBM (via scripts irmãos) não batem com a tese — a fonte
real é `1_wesad/full/*.ipynb` (grid search/Optuna). Arquivado sem proteção de
schema porque não alimenta nenhum número publicado.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from tqdm import tqdm
from common import load_wesad_tabular, wrist_cols, report_metrics, save_results


def run_loso(X_df, y, groups, feature_cols, label):
    X = X_df[feature_cols].values.astype(float)
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X, y, groups))
    _pbar = tqdm(_splits, desc="ET WESAD", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in enumerate(_pbar):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imp = SimpleImputer(strategy='mean')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))

        clf = ExtraTreesClassifier(
            n_estimators=500, max_features=0.5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]

        f1 = f1_score(y_te, (prob >= 0.5).astype(int), average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te); all_prob.extend(prob)
        _pbar.set_postfix(F1=f'{f1:.3f}')

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


if __name__ == '__main__':
    print('Loading WESAD...')
    X_df, y, groups, feat_cols = load_wesad_tabular()
    all_results = []

    print('\n' + '='*65)
    print('  WESAD LOSO — ExtraTrees — WRIST features')
    print('='*65)
    wcols = wrist_cols(feat_cols)
    print(f'  Features: {len(wcols)} wrist')
    all_results.append(run_loso(X_df, y, groups, wcols,
                                f'ET WESAD — wrist ({len(wcols)} feat)'))

    print('\n' + '='*65)
    print('  WESAD LOSO — ExtraTrees — ALL features')
    print('='*65)
    print(f'  Features: {len(feat_cols)} total')
    all_results.append(run_loso(X_df, y, groups, feat_cols,
                                f'ET WESAD — all ({len(feat_cols)} feat)'))

    save_results(all_results,
                 os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_et.json'))
