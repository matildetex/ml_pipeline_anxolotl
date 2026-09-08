"""
ECG feature extraction from BIOPAC recordings (dataset.zip).

For each Anxolotl participant with a BIOPAC recording:
  1. Re-run preprocessing to get the 1Hz labelled signal + t_sec
  2. Build the same 60s/30s stride windows as build_features_tabular
  3. For each window, extract the corresponding ECG segment (2000 Hz)
  4. Compute ECG / HRV features with neurokit2
  5. Save as processed_data_anxolotl_ecg.pkl (same window order as main pkl)

Mapping (confirmed by timestamp matching):
  p7 → real_id21  ...  p44 → real_id59
  p41 has no BIOPAC recording (ECG features set to NaN).
  p23 → real_id39 contains a duplicate of id38's file (flagged, skipped).
  p6, p24 excluded from analysis.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, io, zipfile, calendar, warnings, pickle, re
import numpy as np
import pandas as pd
import bioread
import neurokit2 as nk
from scipy.signal import welch as sp_welch

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

sys.path.insert(0, '3_anxolotl')
from anxolotl_preprocessing import preprocess_session

DATASET_ZIP  = 'dataset.zip'
SESSIONS_DIR = 'sessions'
FS_ECG       = 2000   # BIOPAC ECG sampling rate (Hz)
TZ_OFFSET    = 1      # Portugal UTC+1 (WEST, in effect for all sessions Mar-May 2026)
WINDOW_SEC   = 60
STRIDE_SEC   = 30

# ── Participant → BIOPAC .acq path in dataset.zip ────────────────────────────
# Derived by matching session CSV start times to BIOPAC folder names (±2 min).
# p25/p26 (real_id25/26) have no watch data → excluded at source.
# p23 (real_id39) has a corrupt .acq (duplicate of real_id38) → skipped.
# p41 has no BIOPAC recording → ECG features will be NaN.

PARTICIPANT_TO_ACQ = {
    'p7':  'dataset/sessions/real_id21_20260401_112759/biopac/real_id21_20260401_112759_biopac.acq',
    'p8':  'dataset/sessions/real_id22_20260401_141800/biopac/real_id22_20260401_141800_biopac.acq',
    'p9':  'dataset/sessions/real_id23_20260408_153220/biopac/real_id23_20260408_153220_biopac.acq',
    'p10': 'dataset/sessions/real_id24_20260408_162426/biopac/real_id24_20260408_162426_biopac.acq',
    'p11': 'dataset/sessions/real_id27_20260415_112703/biopac/real_id27_20260415_112703_biopac.acq',
    'p12': 'dataset/sessions/real_id28_20260415_142352/biopac/real_id28_20260415_142352_biopac.acq',
    'p13': 'dataset/sessions/real_id29_20260415_151808/biopac/real_id29_20260415_151808_biopac.acq',
    'p14': 'dataset/sessions/real_id30_20260415_165356/biopac/real_id30_20260415_165356_biopac.acq',
    'p15': 'dataset/sessions/real_id31_20260421_143658/biopac/real_id31_20260421_143658_biopac.acq',
    'p16': 'dataset/sessions/real_id32_20260422_111906/biopac/real_id32_20260422_111906_biopac.acq',
    'p17': 'dataset/sessions/real_id33_20260422_153421/biopac/real_id33_20260422_153421_biopac.acq',
    'p18': 'dataset/sessions/real_id34_20260423_142928/biopac/real_id34_20260423_142928_biopac.acq',
    'p19': 'dataset/sessions/real_id35_20260423_161331/biopac/real_id35_20260423_161331_biopac.acq',
    'p20': 'dataset/sessions/real_id36_20260424_114649/biopac/real_id36_20260424_114649_biopac.acq',
    'p21': 'dataset/sessions/real_id37_20260424_141932/biopac/real_id37_20260424_141932_biopac.acq',
    'p22': 'dataset/sessions/real_id38_20260424_150752/biopac/real_id38_20260424_150752_biopac.acq',
    # p23 → real_id39 folder contains real_id38-named file but different content (correct data, wrong filename)
    'p23': 'dataset/sessions/real_id39_20260424_154725/biopac/real_id38_20260424_150752_biopac.acq',
    'p25': 'dataset/sessions/real_id41_20260506_113659/biopac/real_id41_20260506_113659_biopac.acq',
    'p26': 'dataset/sessions/real_id42_20260506_184504/biopac/real_id42_20260506_184504_biopac.acq',
    'p27': 'dataset/sessions/real_id43_20260508_141900/biopac/real_id43_20260508_141900_biopac.acq',
    'p28': 'dataset/sessions/real_id44_20260508_150401/biopac/real_id44_20260508_150401_biopac.acq',
    'p29': 'dataset/sessions/real_id45_20260508_154519/biopac/real_id45_20260508_154519_biopac.acq',
    'p30': 'dataset/sessions/real_id46_20260513_143044/biopac/real_id46_20260513_143044_biopac.acq',
    'p31': 'dataset/sessions/real_id47_20260513_151703/biopac/real_id47_20260513_151703_biopac.acq',
    'p32': 'dataset/sessions/real_id48_20260513_162046/biopac/real_id48_20260513_162046_biopac.acq',
    'p33': 'dataset/sessions/real_id49_20260514_113334/biopac/real_id49_20260514_113334_biopac.acq',
    'p34': 'dataset/sessions/real_id50_20260514_143239/biopac/real_id50_20260514_143239_biopac.acq',
    'p35': 'dataset/sessions/real_id51_20260514_152946/biopac/real_id51_20260514_152946_biopac.acq',
    'p36': 'dataset/sessions/real_id52_20260514_162415/biopac/real_id52_20260514_162415_biopac.acq',
    'p37': 'dataset/sessions/real_id53_20260514_171958/biopac/real_id53_20260514_171958_biopac.acq',
    'p38': 'dataset/sessions/real_id54_20260519_142208/biopac/real_id54_20260519_142208_biopac.acq',
    'p39': 'dataset/sessions/real_id55_20260519_153016/biopac/real_id55_20260519_153016_biopac.acq',
    'p40': 'dataset/sessions/real_id56_20260520_142044/biopac/real_id56_20260520_142044_biopac.acq',
    # p41 → no BIOPAC recording
    'p42': 'dataset/sessions/real_id57_20260520_161755/biopac/real_id57_20260520_161755_biopac.acq',
    'p43': 'dataset/sessions/real_id58_20260521_145115/biopac/real_id58_20260521_145115_biopac.acq',
    'p44': 'dataset/sessions/real_id59_20260521_152105/biopac/real_id59_20260521_152105_biopac.acq',
}

ECG_FEAT_COLS = [
    # Time-domain HRV
    'ecg_rr_mean', 'ecg_rr_std', 'ecg_rmssd', 'ecg_pnn50', 'ecg_hr',
    # Frequency-domain HRV
    'ecg_lf_power', 'ecg_hf_power', 'ecg_lf_hf',
    # ECG amplitude
    'ecg_amp_mean', 'ecg_amp_std',
    # Respiration
    'ecg_resp_mean', 'ecg_resp_std',
    # Poincaré plot (nonlinear)
    'ecg_sd1', 'ecg_sd2', 'ecg_sd1_sd2',
    # DFA α1 (short-range fractal scaling, most stress-sensitive)
    'ecg_dfa_alpha1',
    # Sample entropy (signal regularity)
    'ecg_sampen',
]
N_ECG_FEAT = len(ECG_FEAT_COLS)


# ── Utility: BIOPAC start time from .acq path ────────────────────────────────

def acq_path_to_unix(acq_path):
    """
    Parse BIOPAC start Unix timestamp from folder name inside the zip path.
    e.g. 'dataset/sessions/real_id21_20260401_112759/biopac/...'
         → 2026-04-01 11:27:59 local (UTC+1) → Unix UTC
    """
    m = re.search(r'real_id\d+_(\d{8})_(\d{6})', acq_path)
    date_s, time_s = m.group(1), m.group(2)
    y, mo, d  = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:])
    h, mi, s  = int(time_s[:2]), int(time_s[2:4]), int(time_s[4:])
    # Local time → UTC: subtract TZ_OFFSET hours
    h_utc = h - TZ_OFFSET
    if h_utc < 0:
        h_utc += 24
        d -= 1
    from datetime import datetime
    dt_utc = datetime(y, mo, d, h_utc, mi, s)
    return calendar.timegm(dt_utc.timetuple())


# ── ECG feature extraction for one 60s segment ───────────────────────────────

def ecg_features_from_segment(ecg_seg, resp_seg, fs=2000):
    """
    Extracts HRV + amplitude + respiration features from a 60s ECG segment.
    Returns a 1D array of length N_ECG_FEAT, or NaN array on failure.
    """
    nan_row = np.full(N_ECG_FEAT, np.nan)

    try:
        # Clean ECG
        ecg_clean = nk.ecg_clean(ecg_seg, sampling_rate=fs, method='pantompkins1985')

        # Detect R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method='pantompkins1985')
        r_idx = rpeaks['ECG_R_Peaks']

        if len(r_idx) < 5:
            return nan_row

        # R-R intervals in ms
        rr_ms = np.diff(r_idx) / fs * 1000.0

        if len(rr_ms) < 4:
            return nan_row

        # Time-domain HRV
        rr_mean  = float(np.mean(rr_ms))
        rr_std   = float(np.std(rr_ms))
        diffs    = np.diff(rr_ms)
        rmssd    = float(np.sqrt(np.mean(diffs**2)))
        pnn50    = float(np.mean(np.abs(diffs) > 50.0))
        hr_ecg   = float(60000.0 / rr_mean) if rr_mean > 0 else np.nan

        # Frequency-domain HRV via Welch on RR series
        # Resample RR to 4 Hz for frequency analysis
        t_rr   = np.cumsum(rr_ms) / 1000.0  # seconds
        t_uni  = np.arange(0, t_rr[-1], 0.25)  # 4 Hz uniform grid
        rr_uni = np.interp(t_uni, t_rr, rr_ms)
        if len(rr_uni) >= 8:
            freqs, psd = sp_welch(rr_uni, fs=4.0, nperseg=min(len(rr_uni), 64))
            lf = float(np.trapezoid(psd[(freqs >= 0.04) & (freqs <= 0.15)], freqs[(freqs >= 0.04) & (freqs <= 0.15)]) if np.any((freqs >= 0.04) & (freqs <= 0.15)) else np.nan)
            hf = float(np.trapezoid(psd[(freqs >= 0.15) & (freqs <= 0.40)], freqs[(freqs >= 0.15) & (freqs <= 0.40)]) if np.any((freqs >= 0.15) & (freqs <= 0.40)) else np.nan)
            lf_hf = lf / (hf + 1e-10) if (hf is not np.nan and hf > 0) else np.nan
        else:
            lf = hf = lf_hf = np.nan

        # ECG amplitude statistics (cleaned signal)
        amp_mean = float(np.mean(ecg_clean))
        amp_std  = float(np.std(ecg_clean))

        # Respiration
        resp_mean = float(np.mean(resp_seg)) if len(resp_seg) > 0 else np.nan
        resp_std  = float(np.std(resp_seg))  if len(resp_seg) > 0 else np.nan

        # ── Nonlinear HRV ────────────────────────────────────────────────────
        # Poincaré plot: SD1 (short-term), SD2 (long-term)
        diff_rr = np.diff(rr_ms)
        sd1 = float(np.std(diff_rr) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.var(rr_ms) - np.var(diff_rr) / 2))
        sd1_sd2 = sd1 / (sd2 + 1e-10)

        # DFA alpha1 (short-range, n=4..16) using nk.hrv_nonlinear if enough beats
        dfa_alpha1 = np.nan
        sampen     = np.nan
        if len(rr_ms) >= 20:
            try:
                peaks_df = pd.DataFrame({'ECG_R_Peaks': r_idx})
                hrv_nl = nk.hrv_nonlinear(peaks_df, sampling_rate=fs, show=False)
                if 'HRV_DFA_alpha1' in hrv_nl.columns:
                    dfa_alpha1 = float(hrv_nl['HRV_DFA_alpha1'].iloc[0])
                if 'HRV_SampEn' in hrv_nl.columns:
                    sampen = float(hrv_nl['HRV_SampEn'].iloc[0])
            except Exception:
                pass

        return np.array([
            rr_mean, rr_std, rmssd, pnn50, hr_ecg,
            lf, hf, lf_hf,
            amp_mean, amp_std,
            resp_mean, resp_std,
            sd1, sd2, sd1_sd2,
            dfa_alpha1, sampen,
        ], dtype=np.float32)

    except Exception:
        return nan_row


# ── Per-participant processing ────────────────────────────────────────────────

def process_participant(subject, session_csv, acq_zip_path, zf):
    """
    Returns a list of (label, stress_phase, ecg_feat_row) for each valid window.
    """
    print(f'\n── {subject} ──')

    # 1. Preprocess session CSV to get 1Hz data with t_sec
    result = preprocess_session(session_csv, subject, verbose=False)
    df_1hz = result['df']   # filtered rows, has t_sec column
    t0_unix = result['signals_norm']   # need t0 from session CSV

    # Re-read t0 directly from session CSV
    raw = pd.read_csv(session_csv)
    t0 = float(raw['timestamp_unix'].iloc[0])

    # 2. Load BIOPAC from zip
    t_biopac_start = acq_path_to_unix(acq_zip_path)
    offset_sec = t0 - t_biopac_start
    print(f'   t0={t0:.0f}  biopac_start={t_biopac_start:.0f}  '
          f'offset={offset_sec:.1f}s (BIOPAC started {offset_sec:.0f}s before watch)')

    with zf.open(acq_zip_path) as f:
        raw_bytes = io.BytesIO(f.read())
    acq = bioread.read(raw_bytes)
    ecg_full  = acq.channels[0].data.astype(np.float32)
    resp_full = acq.channels[1].data.astype(np.float32) if len(acq.channels) > 1 else np.array([])
    n_biopac  = len(ecg_full)
    print(f'   ECG: {n_biopac} samples ({n_biopac/FS_ECG/60:.1f} min)')

    # 3. Build windows (same logic as build_features_tabular)
    grp = df_1hz.copy().sort_values('t_sec').reset_index(drop=True)
    n   = len(grp)
    rows = []

    for i in range(WINDOW_SEC, n, STRIDE_SEC):
        label_first = int(grp['label'].iloc[i - WINDOW_SEC])
        label_last  = int(grp['label'].iloc[i - 1])
        if label_first != label_last:
            continue  # boundary window — skip
        label = label_first
        if label not in [0, 1]:
            continue

        # t_sec of the window: rows [i-WINDOW_SEC : i]
        t_start_rel = float(grp['t_sec'].iloc[i - WINDOW_SEC])   # relative to t0
        t_end_rel   = float(grp['t_sec'].iloc[i - 1]) + 1.0      # +1s to include last second

        # Absolute time (Unix UTC)
        t_start_abs = t0 + t_start_rel
        t_end_abs   = t0 + t_end_rel

        # BIOPAC sample indices
        s_start = int((t_start_abs - t_biopac_start) * FS_ECG)
        s_end   = int((t_end_abs   - t_biopac_start) * FS_ECG)

        # Guard: indices must be within BIOPAC recording
        if s_start < 0 or s_end > n_biopac or (s_end - s_start) < WINDOW_SEC * FS_ECG * 0.8:
            rows.append(np.full(N_ECG_FEAT, np.nan, dtype=np.float32))
            continue

        ecg_seg  = ecg_full[s_start:s_end]
        resp_seg = resp_full[s_start:s_end] if len(resp_full) > 0 else np.array([])

        feat = ecg_features_from_segment(ecg_seg, resp_seg, fs=FS_ECG)
        rows.append(feat)

    stress_phase = int(grp['stress_phase'].iloc[i]) if 'stress_phase' in grp.columns else -1
    print(f'   Windows processed: {len(rows)}')
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading existing tabular features...')
    with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
        meta = pickle.load(f)

    df_tab     = meta['df']           # DataFrame with all windows
    X_all      = np.array(meta['X'], dtype=float)
    y_all      = np.array(meta['y'])
    groups_all = np.array(meta['groups'])
    feat_cols  = meta['feature_cols']

    print(f'Existing: {X_all.shape[0]} windows, {X_all.shape[1]} features, '
          f'{len(np.unique(groups_all))} subjects')

    # Session CSV map
    import glob
    csv_map = {}
    for fp in sorted(glob.glob(f'{SESSIONS_DIR}/session_*.csv')):
        sid = 'p' + os.path.basename(fp).replace('session_0', '').replace('session_', '').replace('.csv','').lstrip('0')
        csv_map[sid] = fp

    # ECG feature matrix: same row order as X_all
    subjects = sorted(np.unique(groups_all), key=lambda s: int(s[1:]))
    X_ecg_all = np.full((len(X_all), N_ECG_FEAT), np.nan, dtype=np.float32)

    with zipfile.ZipFile(DATASET_ZIP) as zf:
        for subj in subjects:
            mask = groups_all == subj
            n_windows = mask.sum()
            row_idx = np.where(mask)[0]  # actual row positions for this subject (any order)

            if subj not in PARTICIPANT_TO_ACQ:
                print(f'\n── {subj} ── NO BIOPAC (skipping, {n_windows} windows → NaN)')
                continue

            if subj not in csv_map:
                print(f'\n── {subj} ── no session CSV, skipping')
                continue

            try:
                rows = process_participant(subj, csv_map[subj],
                                          PARTICIPANT_TO_ACQ[subj], zf)
                if len(rows) != n_windows:
                    print(f'  ⚠  window count mismatch: ECG={len(rows)} vs tab={n_windows}')
                n_fill = min(len(rows), n_windows)
                X_ecg_all[row_idx[:n_fill]] = np.array(rows[:n_fill])
            except Exception as e:
                print(f'  ERROR {subj}: {e}')

    # Save
    out_path = 'processed_data/anxolotl/processed_data_anxolotl_ecg.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'X_ecg'     : X_ecg_all,
            'ecg_cols'  : ECG_FEAT_COLS,
            'y'         : y_all,
            'groups'    : groups_all,
        }, f)

    # Report
    valid = np.sum(~np.isnan(X_ecg_all[:, 0]))
    print(f'\n{"="*55}')
    print(f'ECG features computed:  {valid}/{len(X_all)} windows')
    print(f'NaN windows (no BIOPAC): {len(X_all)-valid}')
    print(f'Saved → {out_path}')
    print(f'Shape  : {X_ecg_all.shape}')
    print(f'Columns: {ECG_FEAT_COLS}')

    # Quick stats
    df_ecg = pd.DataFrame(X_ecg_all, columns=ECG_FEAT_COLS)
    print('\nECG feature stats (valid windows):')
    print(df_ecg.describe().round(3).to_string())
