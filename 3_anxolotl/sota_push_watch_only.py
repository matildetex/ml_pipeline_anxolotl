"""
Push watch-only (no ECG) Anxolotl LOSO-CV macro F1 as high as possible.

Constraints from the user:
  - No ECG features (watch-only: acc_wrist, hr, ibi, rmssd, sdnn).
  - Exclude stress_phase == 3 (weak stress induction, confirmed out of scope).
  - Re-check window size (don't assume 60s/30s is best).
  - Try multiple models/techniques, change whatever is needed.

Methodology matches the already-verified canonical script
(3_anxolotl/recompute_macro_f1.py): baseline_relative_scale (BL-norm) per
fold, median imputation fit on train only, manual per-subject LOSO.
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')
CACHE_PATH   = os.path.join(os.path.dirname(__file__), '..', 'results', 'df_all_1hz_cache.pkl')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'sota_push_watch_only.txt')

# ── STEP 0 — build / load cached 1Hz df_all (shared across window sizes) ─────
if os.path.exists(CACHE_PATH):
    print('Loading cached df_all…')
    with open(CACHE_PATH, 'rb') as f:
        df_all = pickle.load(f)
else:
    print('Preprocessing all sessions (1Hz)…')
    files = sorted(f for f in os.listdir(SESSIONS_DIR) if f.startswith('session_') and f.endswith('.csv'))
    session_map = {}
    for fn in files:
        number = fn.replace('session_', '').replace('.csv', '').lstrip('0') or '0'
        session_map[f'p{number}'] = os.path.join(SESSIONS_DIR, fn)
    if 'p6' in session_map:
        del session_map['p6']
    df_all, _ = ap.preprocess_all_sessions(session_map)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(df_all, f)
print(f'df_all: {df_all.shape}, subjects={df_all["subject"].nunique()}')

# ── Helpers ────────────────────────────────────────────────────────────────

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
    m_scores, b_scores, ba_scores = [], [], []
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        y_tr_f, y_te_f = y[tr], y[te]
        if use_calibration:
            X_tr_n, y_tr_f, X_te_n, y_te_f = calibration_split(X_tr_n, y_tr_f, X_te_n, y_te_f, cal_frac)
        if len(np.unique(y_te_f)) == 0 or len(X_te_n) == 0:
            continue
        clf = clf_factory()
        clf.fit(X_tr_n, y_tr_f)
        pred = clf.predict(X_te_n)
        m_scores.append(f1_score(y_te_f, pred, average='macro', zero_division=0))
        b_scores.append(f1_score(y_te_f, pred, average='binary', zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_te_f, pred))
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(X_te_n)[:, 1]
        else:
            prob = pred.astype(float)
        y_true_all.extend(y_te_f); y_prob_all.extend(prob)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc = float('nan')
    return float(np.mean(m_scores)), float(np.mean(b_scores)), float(np.mean(ba_scores)), auc


MODEL_FACTORIES = {
    'ET-1000-mf0.5': lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'ET-1000-mf0.3': lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.3, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'ET-1000-mf0.7': lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.7, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'ET-500-mf0.5':  lambda: ExtraTreesClassifier(n_estimators=500,  max_features=0.5, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1),
    'RF-1000':       lambda: RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=-1),
    'HGB':           lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.1, random_state=42),
    'HGB-shallow':   lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42),
    'LGBM-gbdt':     lambda: LGBMClassifier(boosting_type='gbdt', n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'LGBM-dart':     lambda: LGBMClassifier(boosting_type='dart', n_estimators=300, learning_rate=0.05, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'XGBoost':       lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='logloss', verbosity=0),
    'Ensemble-ET-RF-HGB': lambda: VotingClassifier(voting='hard', estimators=[
        ('et', ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ]),
    'Ensemble-RF-ET': lambda: VotingClassifier(voting='soft', estimators=[
        ('rf', RandomForestClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=1000, max_features=0.5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)),
    ]),
}

WINDOW_CONFIGS = [(60, 30), (90, 45), (120, 60), (150, 75)]

results = []
log_lines = []

def log(s):
    print(s)
    log_lines.append(s)

t0 = time.time()
for win, stride in WINDOW_CONFIGS:
    df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=win, stride_sec=stride)
    n_before = len(df_feat)
    df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
    n_after = len(df_feat)
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat['label'].values.astype(int)
    groups = df_feat['subject'].values
    n_subj = df_feat['subject'].nunique()
    log(f'\n{"="*70}\nWindow={win}s stride={stride}s  N={n_after} (removed {n_before-n_after} phase3)  subj={n_subj}\n{"="*70}')

    for mname, factory in MODEL_FACTORIES.items():
        m, b, ba, auc = loso_eval(X, y, groups, factory, use_calibration=False)
        results.append({'win': win, 'model': mname, 'calib': None, 'macro_f1': m, 'binary_f1': b, 'bal_acc': ba, 'auc': auc})
        log(f'  {mname:<22} macro={m:.4f}  binary={b:.4f}  bal_acc={ba:.4f}  auc={auc:.4f}')

elapsed = time.time() - t0
log(f'\nSweep took {elapsed/60:.1f} min')

# ── Best config so far ────────────────────────────────────────────────────
results_sorted = sorted(results, key=lambda r: r['macro_f1'], reverse=True)
log('\n' + '='*70)
log('TOP 10 (no calibration)')
log('='*70)
for r in results_sorted[:10]:
    log(f"  win={r['win']:>3}s  {r['model']:<22} macro={r['macro_f1']:.4f}  auc={r['auc']:.4f}")

with open(RESULTS_PATH, 'w') as f:
    f.write('\n'.join(log_lines))
print(f'\nSaved intermediate results to {RESULTS_PATH}')
