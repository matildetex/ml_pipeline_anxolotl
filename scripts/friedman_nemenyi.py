"""
Friedman + Nemenyi on per-subject MacroF1, WESAD, using existing results/*.json
(fold_f1s already saved from prior runs — no retraining). Two separate tests:
wrist-only models and all-channel models, since they use different feature
availability and aren't a fair single comparison pooled together.

Subject order verified identical across all files (LeaveOneGroupOut sorts by
np.unique(groups), and both the tabular and sequence WESAD pkls have the same
15 subjects: S10,S11,S13,S14,S15,S16,S17,S2,S3,S4,S5,S6,S7,S8,S9).
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

# (json file, index within the file's 2-entry list, display name)
WRIST_MODELS = [
    ('wesad_et.json', 0, 'ET'),
    ('wesad_rf.json', 0, 'RF'),
    ('wesad_logreg.json', 0, 'LogReg'),
    ('wesad_svm.json', 0, 'SVM'),
    ('wesad_ensemble.json', 0, 'Ensemble'),
    ('wesad_1dcnn.json', 0, '1D-CNN'),
    ('wesad_cnngru.json', 0, 'CNN-GRU'),
    ('wesad_rnn.json', 0, 'BiGRU'),
    ('wesad_transformer.json', 0, 'Transformer'),
    ('wesad_mamba.json', 0, 'SSM (Mamba)'),
]
ALL_MODELS = [(f, 1, n) for f, _, n in WRIST_MODELS]


def load_fold_f1s(fname, idx):
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    entry = d[idx]
    ff = entry.get('fold_f1s')
    if ff is None:
        return None
    return np.array(ff), entry.get('label', fname)


def run_test(model_list, title):
    print('\n' + '='*70)
    print(title)
    print('='*70)
    cols = {}
    n_folds = None
    for fname, idx, name in model_list:
        res = load_fold_f1s(fname, idx)
        if res is None:
            print(f'  [SKIP] {name} — no fold_f1s in {fname}')
            continue
        ff, label = res
        if n_folds is None:
            n_folds = len(ff)
        elif len(ff) != n_folds:
            print(f'  [SKIP] {name} — {len(ff)} folds, expected {n_folds} ({label})')
            continue
        cols[name] = ff
        print(f'  {name:<14} mean={ff.mean():.4f}  sd={ff.std():.4f}   ({label})')

    if len(cols) < 3:
        print('  Not enough models with matching fold counts for a Friedman test.')
        return None

    df = pd.DataFrame(cols)  # rows = subjects (folds), cols = models
    stat, p = friedmanchisquare(*[df[c].values for c in df.columns])
    print(f'\n  Friedman chi2={stat:.4f}  p={p:.6f}  (n_models={len(df.columns)}, n_subjects={len(df)})')

    if p < 0.05:
        print('  --> significant differences among models (p<0.05); running Nemenyi post-hoc...')
    else:
        print('  --> NOT significant at p<0.05 — Nemenyi still shown below for reference, '
              'but pairwise differences should not be over-interpreted.')

    nemenyi = sp.posthoc_nemenyi_friedman(df.values)
    nemenyi.columns = df.columns
    nemenyi.index = df.columns
    print('\n  Nemenyi pairwise p-values:')
    print(nemenyi.round(4).to_string())

    ranks = df.rank(axis=1, ascending=False).mean().sort_values()
    print('\n  Average rank (1 = best) per model:')
    print(ranks.round(2).to_string())

    return df, stat, p, nemenyi, ranks


wrist_result = run_test(WRIST_MODELS, 'WRIST-ONLY MODELS (5-7 channels, no chest ECG)')
all_result = run_test(ALL_MODELS, 'ALL-CHANNEL MODELS (includes chest ECG + demographics)')

# Save
out_path = os.path.join(RESULTS, 'friedman_nemenyi.txt')
with open(out_path, 'w') as f:
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_test(WRIST_MODELS, 'WRIST-ONLY MODELS (5-7 channels, no chest ECG)')
        run_test(ALL_MODELS, 'ALL-CHANNEL MODELS (includes chest ECG + demographics)')
    f.write(buf.getvalue())
print(f'\nSaved {out_path}')
