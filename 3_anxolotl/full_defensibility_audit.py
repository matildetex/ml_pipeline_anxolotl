"""
Full defensibility audit for the Watch-only result at 60s AND 90s, covering
every factor a rigorous thesis jury would probe:
  1. Window/stride construction sanity (counts, overlap, boundary handling).
  2. BL-norm leakage — permutation test, explicit for both windows.
  3. Hyperparameter-selection optimism bias — tuned (best-of-grid) vs a fixed,
     not-tuned-on-this-data default, at both windows.
  4. Seed sensitivity — same config, multiple random_state values.
"""
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
    n_before = len(df_feat)
    df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
    X = df_feat[feat_cols].values.astype(float)
    y = df_feat['label'].values.astype(int)
    groups = df_feat['subject'].values
    return X, y, groups, n_before, len(df_feat)


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
    return float(np.mean(m_scores)), float(np.std(m_scores)), float(np.mean(ba_scores)), auc


def shuffle_within_subject(y, groups, rng):
    y_shuf = y.copy()
    for subj in np.unique(groups):
        idx = np.where(groups == subj)[0]
        y_shuf[idx] = rng.permutation(y[idx])
    return y_shuf


DEFAULT_ET = lambda seed=42: ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                                   min_samples_leaf=1, class_weight='balanced',
                                                   random_state=seed, n_jobs=-1)
TUNED_ET_90 = lambda seed=42: ExtraTreesClassifier(n_estimators=300, max_features=0.6,
                                                    min_samples_leaf=1, class_weight='balanced',
                                                    random_state=seed, n_jobs=-1)

results = {}

for win, stride in [(60, 30), (90, 45)]:
    print('\n' + '='*70)
    print(f'WINDOW = {win}s, stride = {stride}s')
    print('='*70)
    X, y, g, n_before, n_after = build(win, stride)
    n_subj = len(np.unique(g))
    per_subj_counts = [ (g==s).sum() for s in np.unique(g) ]
    print(f'  Windows: {n_before} -> {n_after} after excluding phase3 ({n_before-n_after} removed)')
    print(f'  Subjects: {n_subj}, windows/subj: min={min(per_subj_counts)} '
          f'median={np.median(per_subj_counts):.1f} max={max(per_subj_counts)}')
    print(f'  Class balance: baseline={int((y==0).sum())} ({(y==0).mean()*100:.1f}%)  '
          f'stress={int((y==1).sum())} ({(y==1).mean()*100:.1f}%)')
    print(f'  Overlap: window={win}s, stride={stride}s -> {100*(1-stride/win):.0f}% overlap between adjacent windows')

    # 1. DEFAULT (untuned) config -- not selected by looking at this LOSO result
    m_def, sd_def, ba_def, auc_def = loso_eval(X, y, g, DEFAULT_ET)
    print(f'\n  [DEFAULT, untuned] ET-1000 mf=0.5 msl=1: macro={m_def:.4f} sd={sd_def:.4f} '
          f'bal_acc={ba_def:.4f} auc={auc_def:.4f}')

    # 2. TUNED config (only meaningful at 90s, where the grid search ran)
    if win == 90:
        m_tun, sd_tun, ba_tun, auc_tun = loso_eval(X, y, g, TUNED_ET_90)
        print(f'  [TUNED on this LOSO] ET-300 mf=0.6 msl=1: macro={m_tun:.4f} sd={sd_tun:.4f} '
              f'bal_acc={ba_tun:.4f} auc={auc_tun:.4f}')
        print(f'  Optimism from tuning (best-of-42 grid vs untuned default): '
              f'{m_tun - m_def:+.4f}')

    # 3. Permutation leakage test (10 shuffles) on the DEFAULT config
    perm_scores = []
    for seed in range(10):
        rng = np.random.default_rng(2000 + seed)
        y_shuf = shuffle_within_subject(y, g, rng)
        m, sd, ba, auc = loso_eval(X, y_shuf, g, DEFAULT_ET)
        perm_scores.append(m)
    print(f'\n  [PERMUTATION CHECK] real={m_def:.4f}  '
          f'shuffled mean={np.mean(perm_scores):.4f} sd={np.std(perm_scores):.4f}  '
          f'range=[{min(perm_scores):.4f}, {max(perm_scores):.4f}]')

    # 4. Seed sensitivity (5 seeds) on the DEFAULT config
    seed_scores = []
    for seed in [0, 1, 42, 123, 2024]:
        m, sd, ba, auc = loso_eval(X, y, g, lambda seed=seed: DEFAULT_ET(seed))
        seed_scores.append(m)
        print(f'  [SEED {seed:>5}] macro={m:.4f}')
    print(f'  Seed sensitivity: mean={np.mean(seed_scores):.4f}  sd={np.std(seed_scores):.4f}  '
          f'range=[{min(seed_scores):.4f}, {max(seed_scores):.4f}]')

    results[win] = dict(m_def=m_def, sd_def=sd_def, ba_def=ba_def, auc_def=auc_def,
                         perm_mean=np.mean(perm_scores), perm_sd=np.std(perm_scores),
                         seed_mean=np.mean(seed_scores), seed_sd=np.std(seed_scores),
                         n_after=n_after, n_subj=n_subj)

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
for win in [60, 90]:
    r = results[win]
    print(f'{win}s: default macro={r["m_def"]:.4f} (sd={r["sd_def"]:.4f}), '
          f'permuted={r["perm_mean"]:.4f}, seed-range sd={r["seed_sd"]:.4f}')
