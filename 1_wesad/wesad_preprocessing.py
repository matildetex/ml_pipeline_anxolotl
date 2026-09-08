"""
WESAD preprocessing — 11 stats per channel, fully vectorised.

Reads S*.pkl files from wesad.zip and saves processed_data/wesad/processed_data.pkl.

Stats per channel (11): mean, std, min, max, p25, p75, skew, kurtosis, slope, lf_hf, delta_mean

Channels:
  Wrist : acc_wrist, eda_wrist, temp_wrist, hr, ibi, rmssd, sdnn
  Chest : acc_chest, eda_chest, temp_chest, emg_chest
  Demo  : age, gender, coffee_today, coffee_last_hour, sports, smoker, ill,
          stai_baseline, stai_stress

Window 60 s, stride 1 s.  WESAD labels: 1→baseline(0), 2→stress(1), others→skip.
"""

import zipfile, pickle, os
import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
from scipy.signal import find_peaks, butter, filtfilt

BASE     = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE, '..', 'wesad.zip')
OUT_PATH = os.path.join(BASE, '..', 'processed_data', 'wesad', 'processed_data.pkl')

WINDOW_SEC = 60
STRIDE_SEC = 1
LABEL_FS   = 700

DEMO = {
    'S2':  {'age': 27, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 15, 'stai_stress': 8},
    'S3':  {'age': 27, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 16, 'stai_stress': 8},
    'S4':  {'age': 25, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 10, 'stai_stress': 7},
    'S5':  {'age': 35, 'gender': 0, 'coffee_today': 1, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 16, 'stai_stress': 8},
    'S6':  {'age': 27, 'gender': 0, 'coffee_today': 1, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 1, 'ill': 0, 'stai_baseline': 18, 'stai_stress': 10},
    'S7':  {'age': 28, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 1, 'smoker': 0, 'ill': 0, 'stai_baseline': 14, 'stai_stress': 7},
    'S8':  {'age': 27, 'gender': 1, 'coffee_today': 1, 'coffee_last_hour': 0, 'sports': 1, 'smoker': 0, 'ill': 0, 'stai_baseline': 11, 'stai_stress': 8},
    'S9':  {'age': 26, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 1, 'stai_baseline': 17, 'stai_stress': 7},
    'S10': {'age': 28, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 14, 'stai_stress': 10},
    'S11': {'age': 26, 'gender': 1, 'coffee_today': 1, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 16, 'stai_stress': 7},
    'S13': {'age': 28, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 13, 'stai_stress': 10},
    'S14': {'age': 27, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 5,  'stai_stress': 8},
    'S15': {'age': 28, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 14, 'stai_stress': 9},
    'S16': {'age': 24, 'gender': 0, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 11, 'stai_stress': 9},
    'S17': {'age': 29, 'gender': 1, 'coffee_today': 0, 'coffee_last_hour': 0, 'sports': 0, 'smoker': 0, 'ill': 0, 'stai_baseline': 11, 'stai_stress': 9},
}


# ── Signal helpers ────────────────────────────────────────────────────────────

def bpf(sig, lo, hi, fs, order=3):
    sig = np.asarray(sig, dtype=float).ravel()
    bad = ~np.isfinite(sig)
    if bad.all():
        return sig
    if bad.any():
        x = np.arange(len(sig))
        sig[bad] = np.interp(x[bad], x[~bad], sig[~bad])
    nyq = fs / 2.0
    b, a = butter(order, [max(lo / nyq, 1e-4), min(hi / nyq, 1 - 1e-4)], btype='band')
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def downsample_mean(sig, fs_in, n_seconds):
    sig = np.asarray(sig, dtype=float).ravel()
    n   = n_seconds * fs_in
    if len(sig) < n:
        sig = np.concatenate([sig, np.full(n - len(sig), np.nan)])
    return np.nanmean(sig[:n].reshape(n_seconds, fs_in), axis=1)


def acc_mag_1hz(acc, acc_fs, n_seconds):
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        acc = acc[:, None]
    mag = np.sqrt((acc ** 2).sum(axis=1))
    return downsample_mean(mag, acc_fs, n_seconds)


def emg_rms_1hz(emg, fs_in, n_seconds):
    emg = np.asarray(emg, dtype=float).ravel()
    n   = n_seconds * fs_in
    if len(emg) < n:
        emg = np.concatenate([emg, np.full(n - len(emg), np.nan)])
    return np.sqrt(np.nanmean((emg[:n].reshape(n_seconds, fs_in)) ** 2, axis=1))


def bvp_to_1hz(bvp, fs=64, n_seconds=None):
    bvp = np.asarray(bvp, dtype=float).ravel()
    n_seconds = n_seconds or (len(bvp) // fs)
    t_out = np.arange(n_seconds) + 0.5
    bvp_f = bpf(bvp, 0.5, 4.0, fs)
    win   = max(1, int(fs * 0.1))
    bvp_s = np.convolve(bvp_f, np.ones(win) / win, mode='same')
    thr   = np.mean(bvp_s) + np.std(bvp_s)
    peaks, _ = find_peaks(bvp_s, distance=int(fs * 0.4),
                          height=thr, prominence=np.std(bvp_s) * 0.5)
    if len(peaks) < 3:
        return np.full(n_seconds, 75.0), np.full(n_seconds, 0.8)
    t_pk = peaks / fs
    ibi  = np.diff(t_pk)
    ok   = (ibi > 0.35) & (ibi < 1.5)
    ibi, t_ibi = ibi[ok], t_pk[1:][ok]
    if len(ibi) < 2:
        return np.full(n_seconds, 75.0), np.full(n_seconds, 0.8)
    hr = np.clip(60.0 / ibi, 30, 200)
    hr_1hz  = np.interp(t_out, t_ibi, hr,  left=hr[0],  right=hr[-1])
    ibi_1hz = np.interp(t_out, t_ibi, ibi, left=ibi[0], right=ibi[-1])
    return hr_1hz, ibi_1hz


def rolling_hrv(ibi_1hz, win=5):
    s    = pd.Series(ibi_1hz)
    sdnn = s.rolling(win, min_periods=1).std().fillna(0).to_numpy()
    diff = s.diff().fillna(0)
    rmssd = np.sqrt((diff ** 2).rolling(win, min_periods=1).mean().fillna(0).to_numpy())
    return rmssd, sdnn


# ── Vectorised batch stats ───────────────────────────────────────────────────

def batch_stats(windows, prefix):
    """
    windows: (N, W) — N windows each of length W=60.
    Returns a dict of N-length arrays, one per statistic.
    """
    N, W = windows.shape
    t    = np.arange(W, dtype=float)

    # Basic stats
    mu  = np.nanmean(windows, axis=1)
    sd  = np.nanstd(windows, axis=1)
    mn  = np.nanmin(windows, axis=1)
    mx  = np.nanmax(windows, axis=1)
    p25 = np.nanpercentile(windows, 25, axis=1)
    p75 = np.nanpercentile(windows, 75, axis=1)

    # Normalised for skew/kurtosis
    w_norm = (windows - mu[:, None]) / (sd[:, None] + 1e-8)
    sk = sp_skew(w_norm, axis=1)
    ku = sp_kurtosis(w_norm, axis=1)

    # Delta mean (second half − first half)
    half  = W // 2
    delta = np.nanmean(windows[:, half:], axis=1) - np.nanmean(windows[:, :half], axis=1)

    # Slope via closed-form OLS (vectorised)
    sx  = t.sum()
    sxx = (t ** 2).sum()
    sxy = windows @ t          # (N,)
    sy  = windows.sum(axis=1)  # (N,)
    denom = N * sxx - sx ** 2  # wait, this should be W*sxx - sx^2
    denom = W * sxx - sx ** 2
    slope = (W * sxy - sx * sy) / (denom + 1e-8)

    # LF/HF via batch FFT (no loop)
    nperseg = min(W, 32)
    fft_mat = np.fft.rfft(windows, n=nperseg, axis=1)          # (N, nperseg//2+1)
    freqs   = np.fft.rfftfreq(nperseg, d=1.0)
    psd     = (np.abs(fft_mat) ** 2) * 2.0 / (nperseg * 1.0)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    lf      = psd[:, lf_mask].sum(axis=1) if lf_mask.any() else np.zeros(N)
    hf      = psd[:, hf_mask].sum(axis=1) if hf_mask.any() else np.zeros(N)
    lf_hf   = lf / (hf + 1e-8)

    return {
        f'{prefix}_mean'      : mu,
        f'{prefix}_std'       : sd,
        f'{prefix}_min'       : mn,
        f'{prefix}_max'       : mx,
        f'{prefix}_p25'       : p25,
        f'{prefix}_p75'       : p75,
        f'{prefix}_skew'      : sk,
        f'{prefix}_kurtosis'  : ku,
        f'{prefix}_slope'     : slope,
        f'{prefix}_lf_hf'     : lf_hf,
        f'{prefix}_delta_mean': delta,
    }


# ── Per-subject processing ────────────────────────────────────────────────────

def process_subject(subj_id, raw):
    n_sec   = len(raw['label']) // LABEL_FS
    lbl_raw = raw['label']
    label   = np.array([int(lbl_raw[min((s + 1) * LABEL_FS - 1, len(lbl_raw) - 1)])
                        for s in range(n_sec)])

    wrist = raw['signal']['wrist']
    chest = raw['signal']['chest']

    # 1-Hz signals
    acc_w           = acc_mag_1hz(wrist['ACC'], 32, n_sec)
    hr_w, ibi_w     = bvp_to_1hz(wrist['BVP'], fs=64, n_seconds=n_sec)
    rmssd_w, sdnn_w = rolling_hrv(ibi_w)
    eda_w           = downsample_mean(wrist['EDA'],  4, n_sec)
    temp_w          = downsample_mean(wrist['TEMP'], 4, n_sec)
    acc_c           = acc_mag_1hz(chest['ACC'], LABEL_FS, n_sec)
    eda_c           = downsample_mean(chest['EDA'],  LABEL_FS, n_sec)
    temp_c          = downsample_mean(chest['Temp'], LABEL_FS, n_sec)
    emg_c           = emg_rms_1hz(chest['EMG'],  LABEL_FS, n_sec)

    channels = [acc_w, eda_w, temp_w, hr_w, ibi_w, rmssd_w, sdnn_w,
                acc_c, eda_c, temp_c, emg_c]
    ch_names = ['acc_wrist', 'eda_wrist', 'temp_wrist', 'hr', 'ibi', 'rmssd', 'sdnn',
                'acc_chest', 'eda_chest', 'temp_chest', 'emg_chest']
    sig_mat  = np.stack(channels, axis=0)   # (n_channels, n_sec)

    # Find valid window end-indices
    valid_idx, valid_y = [], []
    for i in range(WINDOW_SEC, n_sec, STRIDE_SEC):
        l_end   = label[i - 1]
        l_start = label[i - WINDOW_SEC]
        if l_end != l_start:
            continue
        if l_end == 1:
            y = 0
        elif l_end == 2:
            y = 1
        else:
            continue
        valid_idx.append(i)
        valid_y.append(y)

    if not valid_idx:
        return []

    valid_idx = np.array(valid_idx)
    valid_y   = np.array(valid_y)
    N         = len(valid_idx)

    # Build batch arrays: (N, n_channels, W)
    batch = np.stack([sig_mat[:, i - WINDOW_SEC:i] for i in valid_idx], axis=0)
    # batch shape: (N, n_channels, W=60)

    # Compute stats for all channels at once
    all_stats = {}
    for ci, ch_name in enumerate(ch_names):
        all_stats.update(batch_stats(batch[:, ci, :], ch_name))

    # Build DataFrame
    rows = {'subject': np.full(N, subj_id), 'label': valid_y}
    rows.update(all_stats)
    demo = DEMO.get(subj_id, {})
    for k, v in demo.items():
        rows[k] = np.full(N, float(v))

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import re
    z = zipfile.ZipFile(ZIP_PATH)
    subjects = sorted(
        set(re.match(r'WESAD/(S\d+)/', n).group(1)
            for n in z.namelist() if re.match(r'WESAD/(S\d+)/', n)),
        key=lambda s: int(s[1:])
    )

    dfs = []
    for subj in subjects:
        print(f'  {subj} ...', end='  ', flush=True)
        with z.open(f'WESAD/{subj}/{subj}.pkl') as f:
            raw = pickle.load(f, encoding='latin1')
        df_s = process_subject(subj, raw)
        n = len(df_s) if df_s is not None and len(df_s) else 0
        print(f'{n} windows')
        if n:
            dfs.append(df_s)

    df = pd.concat(dfs, ignore_index=True)
    n_bl = int((df['label'] == 0).sum())
    n_st = int((df['label'] == 1).sum())
    print(f'\nTotal: {len(df)} windows  (baseline={n_bl}  stress={n_st})')

    demo_keys    = list(DEMO['S2'].keys())
    signal_cols  = [c for c in df.columns if c not in ['subject', 'label'] + demo_keys]
    feature_cols = signal_cols + demo_keys

    X      = df[feature_cols]
    y      = df['label']
    groups = df['subject']

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump({'df': df, 'X': X, 'y': y,
                     'groups': groups, 'feature_cols': feature_cols}, f)

    print(f'\nSaved → {OUT_PATH}  ({len(feature_cols)} features)')
    acc_cols = [c for c in feature_cols if c.startswith('acc_wrist_')]
    print(f'acc_wrist stats ({len(acc_cols)}): {acc_cols}')
    shared = [c for c in feature_cols if c.startswith(('acc_wrist_', 'hr_', 'ibi_', 'rmssd_', 'sdnn_'))]
    print(f'Shared-with-Anxolotl features: {len(shared)}  (5 channels × 11 stats = 55)')


if __name__ == '__main__':
    main()
