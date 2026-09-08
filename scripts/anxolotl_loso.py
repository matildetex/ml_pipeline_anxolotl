"""Anxolotl LOSO — Classical + DANN + AutoGluon (BL-norm).
40-feat = common with WESAD  |  60-feat = all Anxolotl features
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               VotingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from autogluon.tabular import TabularPredictor
from _dann_tabular import dann_tabular_loso
from tqdm import tqdm
from common import load_anxolotl, load_wesad_tabular, report_metrics, save_results

N_BL = 8  # Anxolotl: 5-min baseline ≈ 8 windows of ~37 s


# ── BL-norm ───────────────────────────────────────────────────────────────────

def _bl_train(X, y, groups):
    X = X.astype(float).copy()
    for s in np.unique(groups):
        sm = groups == s; bm = sm & (y == 0)
        if bm.sum() > 1:
            for j in range(X.shape[1]):
                v = X[bm, j][~np.isnan(X[bm, j])]
                if len(v) > 1:
                    X[sm, j] = (X[sm, j] - v.mean()) / (v.std() + 1e-8)
    return X


def _bl_test(X, groups):
    X = X.astype(float).copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]; bl = idx[:N_BL]
        for j in range(X.shape[1]):
            v = X[bl, j][~np.isnan(X[bl, j])]
            if len(v) > 1:
                X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X


# ── Classical models ──────────────────────────────────────────────────────────

def make_logreg():
    return LogisticRegression(C=0.001, penalty='l1', solver='saga',
                               max_iter=1000, tol=1e-4, random_state=42)
def make_svm():
    return SVC(C=0.1, kernel='poly', gamma='auto', degree=3,
               class_weight='balanced', probability=True, cache_size=500)
def make_rf():
    return RandomForestClassifier(n_estimators=1000, max_features=0.5,
                                   min_samples_leaf=1, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
def make_et():
    return ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                 min_samples_leaf=1, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
def make_ensemble_rf_et():
    return VotingClassifier(voting='soft',
                             estimators=[('rf', make_rf()), ('et', make_et())])
def make_ensemble_full():
    return VotingClassifier(voting='soft', estimators=[
        ('xgb',  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                n_jobs=-1, random_state=42, eval_metric='mlogloss', verbosity=0)),
        ('rf',   RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                         max_depth=10, n_jobs=-1, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=305, max_depth=4, learning_rate=0.195,
                                  num_leaves=37, min_child_samples=30, subsample=0.654,
                                  colsample_bytree=0.852, n_jobs=-1, random_state=42, verbose=-1)),
    ])

MODELS = {
    'LogReg':         (make_logreg,        True),
    'SVM':            (make_svm,           True),
    'RF':             (make_rf,            False),
    'ET':             (make_et,            False),
    'Ensemble RF+ET': (make_ensemble_rf_et, False),
    'Ensemble full':  (make_ensemble_full,  True),
}


def run_classical(X_arr, y, groups, feat_names, tag):
    logo = LeaveOneGroupOut()
    acc = {n: {'true': [], 'prob': [], 'folds': []} for n in MODELS}

    _splits = list(logo.split(X_arr, y, groups))
    _pbar = tqdm(enumerate(_splits), total=len(_splits),
                 desc="LOSO", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in _pbar:
        subj = groups[te[0]]
        X_tr_bl = _bl_train(X_arr[tr], y[tr], groups[tr])
        X_te_bl = _bl_test(X_arr[te], groups[te])
        imp = SimpleImputer(strategy='mean')
        X_tr_n = imp.fit_transform(X_tr_bl)
        X_te_n = imp.transform(X_te_bl)
        y_tr, y_te = y[tr], y[te]

        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr_n, y_tr)

        for name, (fn, use_smote) in MODELS.items():
            clf = fn()
            try:
                clf.fit(X_sm if use_smote else X_tr_n,
                        y_sm if use_smote else y_tr)
                prob = clf.predict_proba(X_te_n)[:, 1]
            except Exception as e:
                print(f'    {name} fold {fold+1} err: {e}'); continue
            f1 = f1_score(y_te, (prob >= 0.5).astype(int), average='macro', zero_division=0)
            acc[name]['true'].extend(y_te)
            acc[name]['prob'].extend(prob)
            acc[name]['folds'].append(f1)

        print(f'  fold {fold+1:2d} | {subj}  ' +
              '  '.join(f'{n}={acc[n]["folds"][-1]:.3f}' for n in acc if acc[n]['folds']))

    return [report_metrics(np.array(d['true']), np.array(d['prob']),
                           f'{name} Anxolotl — {tag}', d['folds'])
            for name, d in acc.items() if d['true']]


# ── AutoGluon LOSO ────────────────────────────────────────────────────────────

def run_autogluon(X_arr, y, groups, feat_names, tag):
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X_arr, y, groups))
    _pbar = tqdm(enumerate(_splits), total=len(_splits),
                 desc="LOSO", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in _pbar:
        subj = groups[te[0]]
        X_tr_bl = _bl_train(X_arr[tr], y[tr], groups[tr])
        X_te_bl = _bl_test(X_arr[te], groups[te])
        imp = SimpleImputer(strategy='mean')
        X_tr_n = imp.fit_transform(X_tr_bl)
        X_te_n = imp.transform(X_te_bl)
        y_tr, y_te = y[tr], y[te]

        train_df = pd.DataFrame(X_tr_n, columns=feat_names)
        train_df['label'] = y_tr
        test_df  = pd.DataFrame(X_te_n, columns=feat_names)

        try:
            pred = TabularPredictor(label='label', eval_metric='f1_macro',
                                    verbosity=0).fit(train_df, time_limit=60,
                                                     presets='good_quality')
            prob = pred.predict_proba(test_df)[1].values
        except Exception as e:
            print(f'  AutoGluon fold {fold+1} err: {e}'); continue

        f1 = f1_score(y_te, (prob >= 0.5).astype(int), average='macro', zero_division=0)
        fold_f1s.append(f1); all_true.extend(y_te); all_prob.extend(prob)
        _pbar.set_postfix(F1=f'{f1:.3f}', subj=str(subj))

    return report_metrics(np.array(all_true), np.array(all_prob),
                          f'AutoGluon Anxolotl — {tag}', fold_f1s) if all_true else None


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading Anxolotl...')
    X_df, y, groups, feat_cols = load_anxolotl()
    print('Loading WESAD (for 40 common features)...')
    _, _, _, wesad_cols = load_wesad_tabular()

    fc_40 = sorted(set(wesad_cols) & set(feat_cols))
    fc_60 = list(feat_cols)
    print(f'  Anxolotl: {len(fc_60)} feat  |  common with WESAD: {len(fc_40)}')

    all_results = []

    for feat_cols_run, tag in [(fc_40, '40 common feat'), (fc_60, '60 all feat')]:
        X = X_df[feat_cols_run].values.astype(float)

        print(f'\n{"="*65}\n  Anxolotl LOSO — Classical — {tag}\n{"="*65}')
        all_results += run_classical(X, y, groups, feat_cols_run, tag)

        print(f'\n{"="*65}\n  Anxolotl LOSO — DANN — {tag}\n{"="*65}')
        r = dann_tabular_loso(X, y, groups,
                               f'DANN Anxolotl — {tag}', _bl_train, _bl_test)
        if r: all_results.append(r)

        print(f'\n{"="*65}\n  Anxolotl LOSO — AutoGluon — {tag}\n{"="*65}')
        r = run_autogluon(X, y, groups, feat_cols_run, tag)
        if r: all_results.append(r)

    save_results(all_results,
                 os.path.join(os.path.dirname(__file__), '..', 'results', 'anxolotl_loso.json'))
