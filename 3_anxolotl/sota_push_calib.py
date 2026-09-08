import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'df_all_1hz_cache.pkl')
with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)


def build(win, stride):
    df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=win, stride_sec=stride)
    df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat['label'].values.astype(int)
    groups = df_feat['subject'].values
    return X, y, groups


def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te
    n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx])


def loso_eval(X, y, groups, clf_factory, use_calibration=False, cal_frac=0.5):
    subjects = np.unique(groups)
    imp = SimpleImputer(strategy='median')
    m_scores, ba_scores, n_te = [], [], []
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n); X_te_n = imp.transform(X_te_n)
        y_tr_f, y_te_f = y[tr], y[te]
        if use_calibration:
            X_tr_n, y_tr_f, X_te_n, y_te_f = calibration_split(X_tr_n, y_tr_f, X_te_n, y_te_f, cal_frac)
        if len(X_te_n) == 0:
            continue
        clf = clf_factory(); clf.fit(X_tr_n, y_tr_f)
        pred = clf.predict(X_te_n)
        m_scores.append(f1_score(y_te_f, pred, average='macro', zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_te_f, pred))
        n_te.append(len(y_te_f))
        prob = clf.predict_proba(X_te_n)[:, 1] if hasattr(clf, 'predict_proba') else pred.astype(float)
        y_true_all.extend(y_te_f); y_prob_all.extend(prob)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc = float('nan')
    return float(np.mean(m_scores)), float(np.std(m_scores)), float(np.mean(ba_scores)), auc, np.mean(n_te)


X60, y60, g60 = build(60, 30)
X90, y90, g90 = build(90, 45)

best_et_90 = lambda: ExtraTreesClassifier(n_estimators=300, max_features=0.6, min_samples_leaf=1,
                                           class_weight='balanced', random_state=42, n_jobs=-1)
best_et_60 = lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                                           class_weight='balanced', random_state=42, n_jobs=-1)

runs = [
    ('60s no-calib',  X60, y60, g60, best_et_60, False, 0.5),
    ('60s calib=0.3', X60, y60, g60, best_et_60, True,  0.3),
    ('60s calib=0.5', X60, y60, g60, best_et_60, True,  0.5),
    ('60s calib=0.7', X60, y60, g60, best_et_60, True,  0.7),
    ('90s no-calib',  X90, y90, g90, best_et_90, False, 0.5),
    ('90s calib=0.3', X90, y90, g90, best_et_90, True,  0.3),
    ('90s calib=0.5', X90, y90, g90, best_et_90, True,  0.5),
]
print('='*70); print('Calibration comparison — 60s vs 90s, watch-only, phase3 excluded'); print('='*70)
for label, X, y, g, factory, use_cal, cf in runs:
    m, sd, ba, auc, n_te = loso_eval(X, y, g, factory, use_calibration=use_cal, cal_frac=cf)
    print(f'  {label:<16} macro={m:.4f}  sd={sd:.4f}  bal_acc={ba:.4f}  auc={auc:.4f}  avg_n_test={n_te:.1f}')
