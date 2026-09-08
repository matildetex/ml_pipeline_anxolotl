"""
Re-extract ECG features at a 90s/45s window (matching the best watch-only
config), reusing the logic in ecg_biopac_features.py via monkey-patched
WINDOW_SEC/STRIDE_SEC. Needed bioread + neurokit2, installed into the ml env.
"""
import sys, os, pickle, zipfile, glob
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import ecg_biopac_features as ebf
import anxolotl_preprocessing as ap

WINDOW_SEC, STRIDE_SEC = 90, 45
ebf.WINDOW_SEC = WINDOW_SEC
ebf.STRIDE_SEC = STRIDE_SEC

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'df_all_1hz_cache.pkl')
with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)

df_tab, feat_cols = ap.build_features_tabular(df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)
X_all = df_tab[feat_cols].values.astype(float)
y_all = df_tab['label'].values
groups_all = df_tab['subject'].values
print(f'Watch tabular @ {WINDOW_SEC}s: {X_all.shape[0]} windows, {X_all.shape[1]} features, '
      f'{len(np.unique(groups_all))} subjects')

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')
DATASET_ZIP = os.path.join(os.path.dirname(__file__), '..', 'dataset.zip')

csv_map = {}
for fp in sorted(glob.glob(f'{SESSIONS_DIR}/session_*.csv')):
    sid = 'p' + os.path.basename(fp).replace('session_0', '').replace('session_', '').replace('.csv', '').lstrip('0')
    csv_map[sid] = fp

subjects = sorted(np.unique(groups_all), key=lambda s: int(s[1:]))
X_ecg_all = np.full((len(X_all), ebf.N_ECG_FEAT), np.nan, dtype=np.float32)

with zipfile.ZipFile(DATASET_ZIP) as zf:
    for subj in subjects:
        mask = groups_all == subj
        n_windows = int(mask.sum())
        row_idx = np.where(mask)[0]

        if subj not in ebf.PARTICIPANT_TO_ACQ:
            print(f'-- {subj} -- NO BIOPAC (skipping, {n_windows} windows -> NaN)')
            continue
        if subj not in csv_map:
            print(f'-- {subj} -- no session CSV, skipping')
            continue
        try:
            rows = ebf.process_participant(subj, csv_map[subj], ebf.PARTICIPANT_TO_ACQ[subj], zf)
            if len(rows) != n_windows:
                print(f'  WARNING window count mismatch: ECG={len(rows)} vs tab={n_windows}')
            n_fill = min(len(rows), n_windows)
            X_ecg_all[row_idx[:n_fill]] = np.array(rows[:n_fill])
        except Exception as e:
            print(f'  ERROR {subj}: {e}')

out_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'anxolotl',
                         'processed_data_anxolotl_ecg_90s.pkl')
with open(out_path, 'wb') as f:
    pickle.dump({
        'X_watch': X_all, 'feat_cols': feat_cols, 'stress_phase': df_tab['stress_phase'].values,
        'X_ecg': X_ecg_all, 'ecg_cols': ebf.ECG_FEAT_COLS,
        'y': y_all, 'groups': groups_all,
    }, f)

valid = np.sum(~np.isnan(X_ecg_all[:, 0]))
print(f'\n{"="*60}')
print(f'ECG @ 90s computed: {valid}/{len(X_all)} windows valid')
print(f'Saved -> {out_path}')
