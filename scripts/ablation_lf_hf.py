"""Ablation: LF/HF ratio on non-HRV channels (ACC, HR) vs. HRV-only.

Conditions tested with ET + BL-norm LOSO on Anxolotl:
  A. All 60 features  (baseline — should match existing 0.9212)
  B. 59 feat: drop acc_wrist_lf_hf
  C. 58 feat: drop acc_wrist_lf_hf + hr_lf_hf  (LF/HF only on IBI/RMSSD/SDNN)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from common import load_anxolotl

N_BL = 8


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


def run_et_loso(X_arr, y, groups, label):
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    for fold, (tr, te) in enumerate(logo.split(X_arr, y, groups)):
        X_tr = _bl_train(X_arr[tr], y[tr], groups[tr])
        X_te = _bl_test(X_arr[te], groups[te])
        imp = SimpleImputer(strategy='mean')
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        y_tr, y_te = y[tr], y[te]

        clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                   min_samples_leaf=1, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]

        f1 = f1_score(y_te, (prob >= 0.5).astype(int), average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te)
        all_prob.extend(prob)

        subj = groups[te[0]]
        print(f'  fold {fold+1:2d} | {subj:4s}  F1={f1:.3f}', flush=True)

    macro_f1 = f1_score(np.array(all_true), (np.array(all_prob) >= 0.5).astype(int),
                        average='macro', zero_division=0)
    mean_fold = float(np.mean(fold_f1s))
    std_fold  = float(np.std(fold_f1s))
    print(f'\n  [{label}]  macro F1 = {macro_f1:.4f} | mean-fold = {mean_fold:.4f} ± {std_fold:.4f}')
    return macro_f1, mean_fold, std_fold, fold_f1s


if __name__ == '__main__':
    print('Loading Anxolotl...')
    X_df, y, groups, feat_cols = load_anxolotl()

    conditions = [
        ('A — all 60 feat',
         feat_cols),
        ('B — drop acc_wrist_lf_hf (59 feat)',
         [c for c in feat_cols if c != 'acc_wrist_lf_hf']),
        ('C — drop acc+hr lf_hf (58 feat)',
         [c for c in feat_cols if c not in ('acc_wrist_lf_hf', 'hr_lf_hf')]),
    ]

    results = {}
    for label, cols in conditions:
        print(f'\n{"="*65}')
        print(f'  {label}  ({len(cols)} features)')
        print('='*65)
        X = X_df[cols].values.astype(float)
        f1, mean_f, std_f, folds = run_et_loso(X, y, groups, label)
        results[label] = {'f1': f1, 'mean_fold': mean_f, 'std_fold': std_f,
                          'n_feat': len(cols)}

    print('\n' + '='*65)
    print('  ABLATION SUMMARY — ET + BL-norm, Anxolotl LOSO')
    print('='*65)
    base_f1 = results['A — all 60 feat']['f1']
    for label, r in results.items():
        delta = r['f1'] - base_f1
        sign  = '+' if delta >= 0 else ''
        print(f"  {label}")
        print(f"    macro F1 = {r['f1']:.4f}  (mean-fold {r['mean_fold']:.4f} ± {r['std_fold']:.4f})"
              f"  Δ={sign}{delta:+.4f}")
    print('='*65)
