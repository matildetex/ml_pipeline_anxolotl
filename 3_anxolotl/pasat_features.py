"""
Extract PASAT (Paced Auditory Serial Addition Test) performance features —
the cognitive-stress-induction task administered during the BIOPAC session.
3 blocks per subject, decreasing interval (3.5s -> 2.5s -> 1.5s = increasing
pace/difficulty). These are subject-level (session-constant) covariates, like
demographics — not per-window signals (no timestamps to align to 60s windows).

Adds to the winning watch+ecg+demo (88-feat) set and re-evaluates with the
final tuned config (ET-1000, max_features=0.3, calibration=0.5, no threshold).

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, json, zipfile, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import load_demographics, baseline_relative_scale
from ecg_biopac_features import PARTICIPANT_TO_ACQ

PASAT_COLS = [
    'pasat_acc_t1', 'pasat_acc_t2', 'pasat_acc_t3',
    'pasat_acc_overall', 'pasat_degradation',
    'pasat_omission_rate', 'pasat_rt_mean', 'pasat_rt_std',
]

def pasat_features_from_json(data):
    correct = incorrect = omission = total = 0
    accs = []
    rts = []
    for key in ('test1', 'test2', 'test3'):
        block = data[key]
        c, i, o, t = block['correct'], block['incorrect'], block['omissions'], block['total_trials']
        correct += c; incorrect += i; omission += o; total += t
        accs.append(c / t if t else np.nan)
        for r in block['responses']:
            if r is not None and r.get('responseTime') is not None:
                rts.append(r['responseTime'])
    acc_overall = correct / total if total else np.nan
    degradation = accs[0] - accs[2] if len(accs) == 3 else np.nan
    omission_rate = omission / total if total else np.nan
    rt_mean = float(np.mean(rts)) if rts else np.nan
    rt_std  = float(np.std(rts))  if rts else np.nan
    return [accs[0], accs[1], accs[2], acc_overall, degradation, omission_rate, rt_mean, rt_std]


def session_dir_for(acq_path):
    # acq_path = '.../sessions/real_idNN_TS/biopac/xxx.acq' -> strip 'biopac/xxx.acq'
    return os.path.dirname(os.path.dirname(acq_path))


if __name__ == '__main__':
    print('Loading wearable data...')
    with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
        meta = pickle.load(f)
    X_watch    = np.array(meta['X'], dtype=float)
    y_all      = np.array(meta['y'])
    groups_all = np.array(meta['groups'])

    with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
        ecg_data = pickle.load(f)
    X_ecg_raw = ecg_data['X_ecg'].astype(float)
    ecg_cols  = ecg_data['ecg_cols']
    X_ecg = X_ecg_raw.copy()
    rmssd_idx = ecg_cols.index('ecg_rmssd')
    std_idx   = ecg_cols.index('ecg_rr_std')
    lfhf_idx  = ecg_cols.index('ecg_lf_hf')
    X_ecg[X_ecg[:, rmssd_idx] > 300, :] = np.nan
    X_ecg[X_ecg[:, std_idx]   > 300, :] = np.nan
    X_ecg[X_ecg[:, lfhf_idx]  > 50, lfhf_idx] = np.nan

    df_demo = load_demographics('Data_tese.xlsx')
    demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
                 'respiratory', 'medication', 'exercises', 'exercise_freq',
                 'athlete', 'stimulant']

    def expand(cols, src=df_demo):
        rows = []
        for g in groups_all:
            rows.append(src.loc[g, cols].values.astype(float)
                        if g in src.index else np.zeros(len(cols)))
        return np.array(rows)

    X_demo = expand(demo_cols)

    print('Extracting PASAT features from BIOPAC sessions...')
    zf = zipfile.ZipFile('dataset.zip')
    pasat_by_subject = {}
    for subj, acq_path in PARTICIPANT_TO_ACQ.items():
        sdir = session_dir_for(acq_path)
        pjson_path = f'{sdir}/pasat_results.json'
        try:
            with zf.open(pjson_path) as f:
                data = json.load(f)
            pasat_by_subject[subj] = pasat_features_from_json(data)
        except KeyError:
            print(f'  {subj}: no pasat_results.json at {pjson_path}')

    print(f'PASAT extracted for {len(pasat_by_subject)}/{len(PARTICIPANT_TO_ACQ)} subjects '
          f'(+ {37 - len(PARTICIPANT_TO_ACQ)} subjects with no BIOPAC at all -> NaN)')

    df_pasat = pd.DataFrame.from_dict(pasat_by_subject, orient='index', columns=PASAT_COLS)
    df_pasat.to_csv('results/pasat_features_by_subject.csv')
    print('\nPASAT features (first 10 subjects):')
    print(df_pasat.head(10))
    print('\nPASAT feature stats:')
    print(df_pasat.describe())

    X_pasat = np.full((len(groups_all), len(PASAT_COLS)), np.nan)
    for i, g in enumerate(groups_all):
        if g in df_pasat.index:
            X_pasat[i] = df_pasat.loc[g].values

    valid = (~np.isnan(X_pasat[:, 0])).sum()
    print(f'\nPASAT feature matrix: {X_pasat.shape}, valid windows: {valid}/{len(X_pasat)}')

    X_WIN       = np.hstack([X_watch, X_ecg, X_demo])             # winning set, 88 feat
    X_WIN_PASAT = np.hstack([X_watch, X_ecg, X_demo, X_pasat])    # + PASAT, 96 feat

    # ── Re-run final tuned config with and without PASAT ─────────────────────

    def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
        base_idx = np.where(y_te == 0)[0]
        if len(base_idx) < 4:
            return X_tr_n, y_tr, X_te_n, y_te
        n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
        cal_idx  = base_idx[:n_cal]
        eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
        return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
                np.concatenate([y_tr, y_te[cal_idx]]),
                X_te_n[eval_idx], y_te[eval_idx])

    def run_final(X, label):
        subjects = np.unique(groups_all)
        imp = SimpleImputer(strategy='median')
        f1s = []
        for subj in subjects:
            test_mask, train_mask = groups_all == subj, groups_all != subj
            X_tr, y_tr = X[train_mask], y_all[train_mask]
            X_te, y_te = X[test_mask],  y_all[test_mask]
            X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups_all[train_mask]))
            X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups_all[test_mask]))
            X_tr_n = imp.fit_transform(X_tr_n)
            X_te_n = imp.transform(X_te_n)
            X_tr_n2, y_tr2, X_te_n2, y_te2 = calibration_split(X_tr_n, y_tr, X_te_n, y_te, 0.5)
            clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.3, min_samples_leaf=1,
                                        class_weight='balanced', random_state=42, n_jobs=-1)
            clf.fit(X_tr_n2, y_tr2)
            y_pred = clf.predict(X_te_n2)
            f1s.append(f1_score(y_te2, y_pred, average='binary', zero_division=0))
        mf1 = np.mean(f1s)
        print(f'{label}: mean per-subject F1 = {mf1:.4f}')
        return mf1

    print('\n' + '='*60)
    print('FINAL CONFIG: watch+ECG+demo  vs.  watch+ECG+demo+PASAT')
    print('='*60)
    f1_base  = run_final(X_WIN,       'watch+ecg+demo        (no PASAT)')
    f1_pasat = run_final(X_WIN_PASAT, 'watch+ecg+demo+PASAT  (+PASAT)  ')
    print(f'\nDelta from adding PASAT: {f1_pasat - f1_base:+.4f}')
    print('DONE')
