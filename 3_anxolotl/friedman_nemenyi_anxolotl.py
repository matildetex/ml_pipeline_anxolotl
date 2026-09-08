"""
Friedman + Nemenyi on per-subject MacroF1, Anxolotl, canonical configuration:
60s window / 30s stride, boundary fix applied, stress_phase==3 excluded,
55 wrist-only features (acc_wrist, hr, ibi, rmssd, sdnn x 11 stats), BL-norm
(leak-fixed 2026-07-19), LOSO-CV, 37 subjects.

Models = the 7 in the canonical "per-model comparison" table
(3_anxolotl/recompute_macro_f1.py, Part B) — same factories, same hyperparameters,
so this is a faithful re-export of the thesis table, not a new sweep.

Per-subject fold_f1s were NOT previously saved to disk for the phase3-excluded
variant (only the aggregate mean was reported), so this script re-runs the LOSO
loop to capture them — these are fast sklearn/lightgbm fits (seconds per fold),
not a "retrain" in the expensive DL sense; no deep learning model is refit here.

PEasyPI CNN (the only Anxolotl DL architecture in the repo) is NOT included:
it evaluates on 1s-stride, near-continuous sequence windows (hundreds of test
instances per subject), not the 60s/30s-stride tabular pipeline this comparison
uses, and pooling it in would compare apples to oranges. No other DL model
(CNN-GRU/1D-CNN/Transformer/etc.) has been trained on Anxolotl's tabular data —
those architectures are WESAD-only in this repo (see friedman_nemenyi.py).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'df_all_1hz_cache.pkl')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'results',
                         'friedman_nemenyi_anxolotl_wrist_loso.txt')

with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)

df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=60, stride_sec=30)
n_before = len(df_feat)
df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
n_after = len(df_feat)

X_all = df_feat[feat_cols].values.astype(float)
y_all = df_feat['label'].values.astype(int)
groups_all = df_feat['subject'].values
subjects = np.unique(groups_all)

print(f'Canonical config: 60s/30s stride, boundary-fixed, phase3 excluded.')
print(f'Windows: {n_before} -> {n_after} ({n_before-n_after} phase3 removed)')
print(f'Subjects: {len(subjects)}, features: {len(feat_cols)}')

MODELS = [
    ('ET-1000', lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                         class_weight='balanced', random_state=42, n_jobs=-1)),
    ('ET-1000 mf=0.3', lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.3,
                         class_weight='balanced', random_state=42, n_jobs=-1)),
    ('RF-1000', lambda: RandomForestClassifier(n_estimators=1000,
                         class_weight='balanced', random_state=42, n_jobs=-1)),
    ('HGB', lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                         learning_rate=0.1, random_state=42)),
    ('Ensemble(ET+RF+HGB)', lambda: VotingClassifier(voting='hard', estimators=[
        ('et', ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ])),
    ('LGBM-dart', lambda: LGBMClassifier(boosting_type='dart', n_estimators=300,
                         learning_rate=0.05, class_weight='balanced', random_state=42,
                         n_jobs=-1, verbose=-1)),
    ('LGBM-gbdt', lambda: LGBMClassifier(boosting_type='gbdt', n_estimators=300,
                         class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
]


def loso_per_subject_f1(clf_factory):
    imp = SimpleImputer(strategy='median')
    scores = {}
    for subj in subjects:
        te = groups_all == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_all[tr], y_all[tr], groups_all[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_all[te], y_all[te], groups_all[te]))
        X_tr_n = imp.fit_transform(X_tr_n); X_te_n = imp.transform(X_te_n)
        clf = clf_factory(); clf.fit(X_tr_n, y_all[tr])
        pred = clf.predict(X_te_n)
        scores[subj] = f1_score(y_all[te], pred, average='macro', zero_division=0)
    return scores


print('\nRunning LOSO per model (fast sklearn/lightgbm fits, no DL retraining)...')
cols = {}
for name, factory in MODELS:
    scores = loso_per_subject_f1(factory)
    cols[name] = [scores[s] for s in subjects]
    print(f'  {name:<22} mean={np.mean(cols[name]):.4f}  sd={np.std(cols[name]):.4f}')

df = pd.DataFrame(cols, index=subjects)
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'results',
                        'anxolotl_wrist_loso_per_subject.csv'))

stat, p = friedmanchisquare(*[df[c].values for c in df.columns])
ranks = df.rank(axis=1, ascending=False).mean().sort_values()
mean_f1 = df.mean().reindex(ranks.index)

nemenyi = sp.posthoc_nemenyi_friedman(df.values)
nemenyi.columns = df.columns
nemenyi.index = df.columns

# significant pairs
sig_pairs = []
cols_list = list(df.columns)
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        a, b = cols_list[i], cols_list[j]
        pv = nemenyi.loc[a, b]
        if pv < 0.05:
            sig_pairs.append((a, b, pv))
sig_pairs.sort(key=lambda t: t[2])

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*78)
w('Friedman + Nemenyi — Anxolotl, wrist-only, canonical LOSO configuration')
w('='*78)
w(f'Config: 60s window / 30s stride, boundary fix applied, stress_phase==3 excluded,')
w(f'55 wrist features (acc_wrist, hr, ibi, rmssd, sdnn x 11 stats), BL-norm (leak-fixed')
w(f'2026-07-19), LOSO-CV, {len(subjects)} subjects, {n_after} windows '
  f'({n_before-n_after} phase3 windows removed of {n_before}).')
w()
w('DL architectures NOT included: no CNN-GRU/1D-CNN/BiGRU/Transformer/SSM model has')
w('been trained on Anxolotl tabular data in this repo (those are WESAD-only, see')
w('scripts/friedman_nemenyi.py). PEasyPI CNN exists for Anxolotl but evaluates on')
w('1s-stride near-continuous sequence windows (hundreds of test instances/subject),')
w('not this 60s/30s-stride tabular pipeline -- excluded to avoid comparing incompatible')
w('evaluation granularities under one Friedman test.')
w()
w(f'Friedman chi-square = {stat:.4f}   p-value = {p:.6f}   '
  f'(n_models={len(df.columns)}, n_subjects={len(df)})')
w('--> ' + ('SIGNIFICANT differences among models (p<0.05).' if p < 0.05
            else 'NOT significant at p<0.05 -- pairwise Nemenyi results below should not be over-interpreted.'))
w()
w('-'*78)
w('Mean rank and mean MacroF1 per model (rank 1 = best)')
w('-'*78)
w(f'{"Model":<22}{"Mean rank":>12}{"Mean MacroF1":>16}{"SD":>10}')
for name in ranks.index:
    w(f'{name:<22}{ranks[name]:>12.2f}{mean_f1[name]:>16.4f}{df[name].std():>10.4f}')
w()
w('-'*78)
w('Nemenyi pairwise p-values (full matrix)')
w('-'*78)
w(nemenyi.round(4).to_string())
w()
w('-'*78)
w('Pairs reaching significance (p < 0.05), sorted by p-value')
w('-'*78)
if sig_pairs:
    for a, b, pv in sig_pairs:
        w(f'  {a:<22} vs {b:<22} p={pv:.4f}')
else:
    w('  None. No pairwise comparison reached p<0.05 in the Nemenyi post-hoc test.')
w()
w('-'*78)
w('Note on statistical power at n=37')
w('-'*78)
w('The Friedman test itself (omnibus, all models at once) has reasonable power at')
w('n=37 subjects -- clearly better powered than the WESAD equivalent (n=15, see')
w('friedman_nemenyi.py), which is why a global effect can be detected here even when')
w('individual differences between models are modest. The Nemenyi post-hoc test is')
w('the weaker link: it corrects for all pairwise comparisons among 7 models '
  f'({len(cols_list)*(len(cols_list)-1)//2} pairs), which is conservative, and with')
w('MacroF1 differences between the top models typically under 0.02-0.03, n=37 paired')
w('subjects is still on the low side for detecting small-to-moderate pairwise effect')
w('sizes after that correction. A significant Nemenyi pair here is a comparatively')
w('strong claim; a non-significant pair does not prove the two models are equivalent,')
w('only that this sample could not separate them at the corrected alpha.')

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines))
print(f'\nSaved: {OUT_PATH}')
print(f'Saved: results/anxolotl_wrist_loso_per_subject.csv')
