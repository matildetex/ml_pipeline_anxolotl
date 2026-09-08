"""
Leakage sanity check for the Watch-only and Watch+ECG @ 90s results.

Test: label permutation. For each subject, randomly shuffle that subject's
OWN labels among their own windows (preserving per-subject class balance),
then re-run the exact same pipeline (BL-norm computed from the shuffled
labels, median imputation, RF-1000, LOSO). If the pipeline has a structural
leak (e.g. some feature secretly encodes row order / elapsed time / subject
identity independent of true label), permuted labels would still be
predictable. If it collapses to chance, the real result depends on genuine
label-feature correspondence.

Also reports Balanced Accuracy alongside MacroF1 for the real (unpermuted)
results, and checks for duplicate BIOPAC recordings across participants
(a distinct leakage vector: two "different" subjects sharing one recording).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from demographics_analysis import baseline_relative_scale

with open('../processed_data/anxolotl/processed_data_anxolotl_ecg_90s.pkl', 'rb') as f:
    d = pickle.load(f)

X_watch = np.array(d['X_watch'], dtype=float)
y_all = np.array(d['y'])
groups_all = np.array(d['groups'])
stress_phase = np.array(d['stress_phase'])
ecg_cols = d['ecg_cols']
X_ecg = d['X_ecg'].astype(float).copy()
X_ecg[X_ecg[:, ecg_cols.index('ecg_rmssd')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_rr_std')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_lf_hf')] > 50, ecg_cols.index('ecg_lf_hf')] = np.nan
X_combined = np.hstack([X_watch, X_ecg])

mask = stress_phase != 3
X_watch_f = X_watch[mask]; X_comb_f = X_combined[mask]
y_f = y_all[mask]; g_f = groups_all[mask]


def loso_eval(X, y, groups, clf_factory):
    subjects = np.unique(groups)
    imp = SimpleImputer(strategy='median')
    m_scores, ba_scores = [], []
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n); X_te_n = imp.transform(X_te_n)
        clf = clf_factory(); clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        m_scores.append(f1_score(y[te], pred, average='macro', zero_division=0))
        ba_scores.append(balanced_accuracy_score(y[te], pred))
        prob = clf.predict_proba(X_te_n)[:, 1] if hasattr(clf, 'predict_proba') else pred.astype(float)
        y_true_all.extend(y[te]); y_prob_all.extend(prob)
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auc = float('nan')
    return float(np.mean(m_scores)), float(np.mean(ba_scores)), auc


def shuffle_within_subject(y, groups, rng):
    y_shuf = y.copy()
    for subj in np.unique(groups):
        idx = np.where(groups == subj)[0]
        y_shuf[idx] = rng.permutation(y[idx])
    return y_shuf


ET90 = lambda: ExtraTreesClassifier(n_estimators=300, max_features=0.6, min_samples_leaf=1,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
RF90 = lambda: RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42, n_jobs=-1)

print('='*70); print('REAL (unpermuted) results — Watch-only vs Watch+ECG @ 90s'); print('='*70)
m_w, ba_w, auc_w = loso_eval(X_watch_f, y_f, g_f, ET90)
print(f'  Watch-only  ET-300-mf0.6   macro={m_w:.4f}  bal_acc={ba_w:.4f}  auc={auc_w:.4f}')
m_c, ba_c, auc_c = loso_eval(X_comb_f, y_f, g_f, RF90)
print(f'  Watch+ECG   RF-1000        macro={m_c:.4f}  bal_acc={ba_c:.4f}  auc={auc_c:.4f}')

N_PERM = 10
print(f'\n' + '='*70); print(f'PERMUTATION TEST ({N_PERM} shuffles, within-subject label shuffle)'); print('='*70)

for label, X, factory in [('Watch-only', X_watch_f, ET90), ('Watch+ECG', X_comb_f, RF90)]:
    perm_macro, perm_ba, perm_auc = [], [], []
    for seed in range(N_PERM):
        rng = np.random.default_rng(1000 + seed)
        y_shuf = shuffle_within_subject(y_f, g_f, rng)
        m, ba, auc = loso_eval(X, y_shuf, g_f, factory)
        perm_macro.append(m); perm_ba.append(ba); perm_auc.append(auc)
    print(f'\n{label}:')
    print(f'  Real            : macro={m_w if label=="Watch-only" else m_c:.4f}')
    print(f'  Permuted (mean) : macro={np.mean(perm_macro):.4f}  sd={np.std(perm_macro):.4f}  '
          f'bal_acc={np.mean(perm_ba):.4f}  auc={np.mean(perm_auc):.4f}')
    print(f'  Permuted range  : macro=[{min(perm_macro):.4f}, {max(perm_macro):.4f}]')

print('\nDone.')
