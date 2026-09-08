"""WESAD LOSO — Soft-voting ensemble XGB+RF+LGBM (wrist + all features)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from common import wrist_cols, report_metrics, save_results

# Thesis Section 4.2 (Table 4.4/4.9) was computed on the OLD 137-feat/8-stat WESAD
# schema, NOT the current 130-feat processed_data/wesad/processed_data.pkl
# (regenerated 2026-07-17). See DATA.md.
WESAD_SNAPSHOT = os.path.join(os.path.dirname(__file__), '..', 'processed_data',
                               'wesad_archive', 'processed_data_137feat_snapshot.pkl')


def make_ensemble():
    return VotingClassifier(voting='soft', estimators=[
        ('xgb', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, eval_metric='mlogloss', verbosity=0)),
        ('rf', RandomForestClassifier(
            n_estimators=200, max_features='sqrt', max_depth=10,
            min_samples_leaf=1, n_jobs=-1, random_state=42)),
        ('lgbm', LGBMClassifier(
            n_estimators=305, max_depth=4, learning_rate=0.195,
            num_leaves=37, min_child_samples=30,
            subsample=0.654, colsample_bytree=0.852,
            n_jobs=-1, random_state=42, verbose=-1)),
    ])


def run_loso(X_df, y, groups, feature_cols, label):
    X = X_df[feature_cols].values.astype(float)
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X, y, groups))
    _pbar = tqdm(_splits, desc="Ensemble WESAD", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in enumerate(_pbar):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        imp = SimpleImputer(strategy='mean')
        sc  = StandardScaler()
        X_tr = imp.fit_transform(sc.fit_transform(X_tr))
        X_te = imp.transform(sc.transform(X_te))

        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        X_tr = np.asarray(X_tr)

        clf = make_ensemble()
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
    print(f'Loading WESAD from {WESAD_SNAPSHOT} (old 137-feat schema)...')
    with open(WESAD_SNAPSHOT, 'rb') as f:
        d = pickle.load(f)
    X_df = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
    y = np.array(d['y']); groups = np.array(d['groups']); feat_cols = list(d['feature_cols'])
    assert len(feat_cols) == 137, (
        f'Expected 137 features from the old-schema WESAD snapshot (thesis Section 4.2), '
        f'got {len(feat_cols)} -- see DATA.md.'
    )
    all_results = []

    print('\n' + '='*65)
    print('  WESAD LOSO — Ensemble XGB+RF+LGBM — WRIST features')
    print('='*65)
    wcols = wrist_cols(feat_cols)
    assert len(wcols) == 56, (
        f'Expected 56 wrist features from the old-schema WESAD snapshot, got {len(wcols)} -- see DATA.md.'
    )
    print(f'  Features: {len(wcols)} wrist')
    r = run_loso(X_df, y, groups, wcols, f'Ensemble WESAD — wrist ({len(wcols)} feat)')
    all_results.append(r)

    print('\n' + '='*65)
    print('  WESAD LOSO — Ensemble XGB+RF+LGBM — ALL features')
    print('='*65)
    print(f'  Features: {len(feat_cols)} total')
    r = run_loso(X_df, y, groups, feat_cols, f'Ensemble WESAD — all ({len(feat_cols)} feat)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_ensemble.json'))
