"""
Anxolotl Session Preprocessing Pipeline
========================================
Adapts the WDD preprocessing pipeline for Apple Watch CSV sessions.

CSV format (9 columns):
    timestamp_unix, hr_bpm, ibi_ms, steps, calories_kcal,
    distance_m, acc_x, acc_y, acc_z

Protocol structure (detected via wrist-raise gestures in ACC):
    setup (discard) → [spike1] baseline 5min →
    [spike2] breathing (discard) → [spike3] stress1 →
    [spike4] breathing (discard) → [spike5] stress2 →
    [spike6] breathing (discard) → [spike7] stress3

Labels:
    0 = rest/baseline
    1 = stress
   -1 = discard (setup, breathing transitions)

Usage:
    result = preprocess_session("session_0011.csv", subject_id="p01")
    # result contains: df_1hz, labels_1hz, marker_times, signals dict
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. SIGNAL FILTERS (same as WDD pipeline)
# ──────────────────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    if len(signal) < 30:
        return signal
    return filtfilt(b, a, signal, method='gust')


def lowpass_filter(signal, cutoff, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='low')
    if len(signal) < 20:
        return signal
    return filtfilt(b, a, signal, method='gust')


# ──────────────────────────────────────────────
# 2. LOAD & PARSE CSV
# ──────────────────────────────────────────────

def load_session_csv(filepath):
    """
    Loads an Anxolotl session CSV.
    Returns separate DataFrames for ACC (25 Hz) and HR/IBI (sparse).
    """
    df = pd.read_csv(filepath)
    t0 = df['timestamp_unix'].iloc[0]
    df['t_rel'] = df['timestamp_unix'] - t0  # seconds from session start

    df_acc = df[df['acc_x'].notna()].copy().reset_index(drop=True)
    df_hr  = df[df['hr_bpm'].notna()].copy().reset_index(drop=True)

    # Estimate ACC sampling rate from actual timestamps
    dt = np.diff(df_acc['timestamp_unix'].values[:200])
    fs_acc = round(1.0 / np.median(dt))

    print(f"  Session duration : {df['t_rel'].max():.1f}s "
          f"({df['t_rel'].max()/60:.1f} min)")
    print(f"  ACC samples      : {len(df_acc)}  (fs≈{fs_acc} Hz)")
    print(f"  HR/IBI readings  : {len(df_hr)}")
    print(f"  HR range         : {df_hr['hr_bpm'].min():.0f}–"
          f"{df_hr['hr_bpm'].max():.0f} bpm")
    print(f"  IBI range        : {df_hr['ibi_ms'].min():.0f}–"
          f"{df_hr['ibi_ms'].max():.0f} ms")

    return df_acc, df_hr, fs_acc, t0


# ──────────────────────────────────────────────
# 3. MARKER DETECTION (wrist-raise gesture)
# ──────────────────────────────────────────────

def _find_raised_segments(raised, t_1s, min_duration_sec, bridge_gap_sec=0):
    """
    Finds contiguous True-runs in a boolean array, bridging short gaps
    (≤ bridge_gap_sec) between two True-runs so brief signal dropouts
    inside a real gesture don't split it into two sub-threshold pieces.
    Returns the start time of each run lasting >= min_duration_sec.
    """
    raised = raised.copy()
    if bridge_gap_sec > 0:
        i = 0
        n = len(raised)
        while i < n:
            if not raised[i]:
                j = i
                while j < n and not raised[j]:
                    j += 1
                gap = j - i
                if gap <= bridge_gap_sec and i > 0 and j < n and raised[i-1] and raised[j]:
                    raised[i:j] = True
                i = j
            else:
                i += 1

    gesture_times = []
    in_raised = False
    seg_start = 0
    for i, r in enumerate(raised):
        if r and not in_raised:
            seg_start = i
            in_raised = True
        elif not r and in_raised:
            duration = i - seg_start
            if duration >= min_duration_sec:
                gesture_times.append(t_1s[seg_start])
            in_raised = False
    if in_raised:
        duration = len(raised) - seg_start
        if duration >= min_duration_sec:
            gesture_times.append(t_1s[seg_start])
    return gesture_times


def detect_markers(df_acc, fs_acc, n_markers=8,
                   z_percentile=85, min_duration_sec=8,
                   min_gap_sec=120, baseline_duration_sec=300,
                   bridge_gap_sec=0, relax_level_frac=0.45,
                   min_gesture_amplitude=0.15, relax_min_duration_sec=5,
                   relax_max_std_frac=0.35):
    """
    Detects wrist-raise protocol markers from the ACC z-axis.

    The protocol gesture is: arm raised + 3 deep breaths + 5s apnea.
    This produces a sustained period (≥8-20s) where ACC z shifts from
    the resting value (~-1.0, arm down) toward 0 (arm up/horizontal).

    Short movements (scratching nose, reaching for something) are filtered
    out by the minimum duration requirement.

    Strategy:
      1. Compute 1s median of ACC z to get a stable posture signal.
      2. Find contiguous segments where z_1s > z_threshold, bridging gaps
         ≤ bridge_gap_sec (brief dropouts inside a genuine sustained
         gesture, e.g. a noisy sample or sensor hiccup).
      3. Keep only segments lasting ≥ min_duration_sec (the gesture).
      4. Record the start time of each valid segment as the marker.
      5. Apply protocol spacing constraints via greedy selection:
           marker[0]→[1] gap ≥ baseline_duration_sec (300s, fixed baseline)
           all subsequent gaps ≥ min_gap_sec (120s, task + breathing block)
      6. If marker[0] is missing (baseline gesture not detected), fall back
         to t=0 as the session start and require n_markers-1 gestures.
      7. If markers are still missing after the strict pass, run a second,
         relaxed pass restricted to the unexplored tail of the session
         (after the last confirmed marker + min_gap_sec). The relaxed
         threshold is calibrated per-subject from the z-level their own
         confirmed gestures actually reached — NOT a percentile of the
         tail's local distribution, which is unreliable when the tail is
         mostly still (a flat region has no real arm-raise to separate
         from, so a relative percentile there just fires on noise).
         A candidate is only accepted if the subject showed a real
         resting→raised amplitude (>= min_gesture_amplitude) in their
         confirmed gestures, and the candidate reaches >= relax_level_frac
         of the way from resting to that confirmed raised level.

    Parameters
    ----------
    z_percentile      : percentile of the session z_1s distribution used as
                        arm-raised threshold (default 85; robust across subjects)
    min_duration_sec  : minimum seconds the arm must stay raised (default 8)
    min_gap_sec       : minimum gap between any two non-baseline markers (120s)
    baseline_duration_sec : expected baseline duration, enforced as minimum
                            gap between marker[0] and marker[1] (300s)
    bridge_gap_sec    : max gap (seconds) bridged between two raised runs (default 3)
    relax_level_frac  : fraction of (subject's raised - resting) level a
                        relaxed-pass candidate must reach to be accepted (default 0.6)
    min_gesture_amplitude : minimum resting→raised z amplitude required from
                        the subject's own confirmed gestures before attempting
                        the relaxed pass (default 0.15; guards against
                        calibrating off noise for subjects with weak signal)

    Returns: array of relative timestamps (seconds) for each detected marker.
    """
    z = df_acc['acc_z'].values
    t = df_acc['t_rel'].values

    # 1s median of z (robust to individual noisy samples)
    w = max(1, int(fs_acc * 1.0))
    n_windows = len(z) // w
    z_1s = np.array([np.median(z[i*w:(i+1)*w]) for i in range(n_windows)])
    t_1s = np.array([t[i*w] for i in range(n_windows)])

    # Adaptive threshold: percentile of the session's own z distribution.
    # Arm-raise gestures push z well above the resting baseline,
    # so p85 reliably separates raised-arm from resting/moving-arm states
    # regardless of how high the participant actually lifts their arm.
    z_threshold = np.percentile(z_1s, z_percentile)

    # Find contiguous raised-arm segments (bridging brief dropouts)
    raised = z_1s > z_threshold
    gesture_times = np.array(
        _find_raised_segments(raised, t_1s, min_duration_sec, bridge_gap_sec)
    )

    if len(gesture_times) == 0:
        raise ValueError(
            "No arm-raise gestures detected. Check z_threshold or min_duration_sec."
        )

    # If no gesture is detected in the first baseline_duration_sec,
    # the participant likely skipped the opening gesture.
    # Prepend t=0 so the baseline is still correctly anchored.
    early_threshold = 60  # seconds — if first gesture is later than this,
                          # assume baseline started at t=0 without a gesture
    if gesture_times[0] > early_threshold:
        gesture_times = np.concatenate([[0.0], gesture_times])
        print("  ℹ️  No gesture detected in first 60s — baseline anchored at t=0.")

    # Greedy protocol-aware selection:
    #   marker[0] → marker[1]: gap >= baseline_duration_sec (5 min baseline)
    #   all subsequent markers: gap >= min_gap_sec (task + breathing block)
    def greedy_filter(candidates):
        filtered = []
        for ct in candidates:
            if len(filtered) == 0:
                filtered.append(ct)
            elif len(filtered) == 1:
                if ct - filtered[-1] >= baseline_duration_sec:
                    filtered.append(ct)
            else:
                if ct - filtered[-1] >= min_gap_sec:
                    filtered.append(ct)
            if len(filtered) == n_markers:
                break
        return np.array(filtered)

    marker_times = greedy_filter(gesture_times)

    # Relaxed second pass: try to recover trailing markers the strict pass
    # missed, using a per-subject calibrated threshold (see docstring).
    n_recovered = 0
    if 2 <= len(marker_times) < n_markers:
        peak_zs = []
        for mt in marker_times:
            seg_mask = (t_1s >= mt) & (t_1s < mt + 15)
            if seg_mask.sum() > 0:
                peak_zs.append(np.percentile(z_1s[seg_mask], 90))
        raised_level = np.median(peak_zs)
        resting_level = np.percentile(z_1s, 50)
        amplitude = raised_level - resting_level

        if amplitude >= min_gesture_amplitude:
            relaxed_threshold = resting_level + relax_level_frac * amplitude
            cur = marker_times[-1]
            tail_mask = t_1s > cur + min_gap_sec
            if tail_mask.sum() > relax_min_duration_sec:
                z_tail = z_1s[tail_mask]
                t_tail = t_1s[tail_mask]
                candidate_starts = _find_raised_segments(
                    z_tail > relaxed_threshold, t_tail,
                    relax_min_duration_sec, bridge_gap_sec
                )
                # Plateau check: a genuine held gesture has low variance
                # throughout the segment, unlike a brief spike whose tail
                # is dragged across threshold by a single high sample.
                max_std = relax_max_std_frac * amplitude
                marker_times = list(marker_times)
                for ct in candidate_starts:
                    seg_end_idx = np.searchsorted(t_tail, ct)
                    # walk forward to find this segment's end (next point below threshold)
                    j = seg_end_idx
                    while j < len(z_tail) and z_tail[j] > relaxed_threshold:
                        j += 1
                    seg_vals = z_tail[seg_end_idx:j]
                    seg_std = seg_vals.std() if len(seg_vals) > 0 else np.inf
                    if seg_std > max_std:
                        print(f"  ℹ️  Rejected relaxed candidate at t={ct:.0f}s "
                              f"(std={seg_std:.3f} > {max_std:.3f} — looks like noise, not a held gesture).")
                        continue
                    if ct - cur >= min_gap_sec:
                        marker_times.append(ct)
                        n_recovered += 1
                        cur = ct
                    if len(marker_times) == n_markers:
                        break
                marker_times = np.array(marker_times)

        if n_recovered > 0:
            print(f"  ℹ️  Relaxed pass recovered {n_recovered} additional "
                  f"marker(s) in the session tail (calibrated threshold "
                  f"{relaxed_threshold:.3f}, subject amplitude {amplitude:.3f}).")

    # Minimum required: at least 4 markers (baseline + breathing1 + stress1 + breathing2)
    min_acceptable = 4
    if len(marker_times) < min_acceptable:
        raise ValueError(
            f"Only {len(marker_times)} valid arm-raise gestures detected "
            f"(expected {n_markers}, minimum {min_acceptable}). "
            f"Try lowering z_percentile (current: {z_percentile}) "
            f"or min_duration_sec (current: {min_duration_sec})."
        )

    # Structural coherence check: inter-marker spacings must be plausible.
    # Each task+breathing block is typically 120–400s.
    # A spacing > 400s means a block was missed or the marker is noise.
    if len(marker_times) >= 3:
        spacings = np.diff(marker_times[1:])  # gaps between post-baseline markers
        bad_spacings = [s for s in spacings if s > 400]
        all_bad = all(s > 400 or s < 100 for s in spacings)
        if bad_spacings and all_bad:
            raise ValueError(
                f"All inter-marker spacings are implausible "
                f"({[round(s,0) for s in spacings]}s). "
                f"Expected 100–400s per protocol block. "
                f"Session likely has no valid arm-raise gestures."
            )
        elif bad_spacings:
            print(f"  ⚠️  Suspicious spacing(s) detected: "
                  f"{[round(s,0) for s in bad_spacings]}s "
                  f"(expected ≤400s). Some markers may be misplaced.")

    if len(marker_times) < n_markers:
        print(f"  ⚠️  Only {len(marker_times)}/{n_markers} markers found — "
              f"session appears incomplete ({n_markers - len(marker_times)} "
              f"gesture(s) missing).")

    n_found = len(marker_times)
    print(f"\n  Detected {n_found} markers (wrist-raise gestures):")
    labels_desc = [
        "→ baseline start",
        "→ breathing 1 start (discard)",
        "→ stress 1 start",
        "→ breathing 2 start (discard)",
        "→ stress 2 start",
        "→ breathing 3 start (discard)",
        "→ stress 3 start",
        "→ baseline final start",
    ]
    for i, mt in enumerate(marker_times):
        desc = labels_desc[i] if i < len(labels_desc) else "→ ?"
        print(f"    Marker {i+1} at t={mt:.1f}s  {desc}")

    return marker_times


# ──────────────────────────────────────────────
# 4. BUILD SECOND-BY-SECOND LABELS
# ──────────────────────────────────────────────

def build_labels(marker_times, total_seconds, baseline_duration_sec=300):
    """
    Builds a label array at 1 Hz resolution. Handles any number of markers
    from 4 (partial session) to 8 (complete session) gracefully.

    Full 8-marker phase map:
        m[0] → m[0]+300s : label=0  (baseline initial, 5 min)
        m[1] → m[2]      : label=-1 (breathing 1, discard)
        m[2] → m[3]      : label=1  (stress 1)
        m[3] → m[4]      : label=-1 (breathing 2, discard)
        m[4] → m[5]      : label=1  (stress 2)
        m[5] → m[6]      : label=-1 (breathing 3, discard)
        m[6] → m[7]      : label=1  (stress 3)
        m[7] → m[7]+300s : label=0  (baseline final, up to 5 min or session end)

    If fewer markers are present, only the phases covered are labelled;
    the rest stays -1 (discard).
    """
    labels = np.full(total_seconds, -1, dtype=int)
    m = marker_times
    n = len(m)

    # Helper: safe slice avoiding out-of-bounds
    def label_range(start, end, lbl):
        s = max(0, min(int(start), total_seconds))
        e = max(0, min(int(end),   total_seconds))
        if e > s:
            labels[s:e] = lbl

    # Baseline initial (always present if n >= 1)
    if n >= 1:
        label_range(m[0], m[0] + baseline_duration_sec, 0)

    # Stress 1 (needs markers 2 and 3, indices 2 and 3)
    if n >= 4:
        label_range(m[2], m[3], 1)

    # Stress 2 (needs markers 4 and 5)
    if n >= 6:
        label_range(m[4], m[5], 1)

    # Stress 3 (needs marker 6; ends at marker 7 or session end)
    if n >= 7:
        end_stress3 = int(m[7]) if n >= 8 else total_seconds
        label_range(m[6], end_stress3, 1)

    # Baseline final (needs marker 7)
    if n >= 8:
        t_b2_end = min(int(m[7]) + baseline_duration_sec, total_seconds)
        label_range(m[7], t_b2_end, 0)
        b2_sec = t_b2_end - int(m[7])
        if b2_sec < baseline_duration_sec:
            print(f"  ⚠️  Baseline final truncated: {b2_sec}s "
                  f"(expected {baseline_duration_sec}s)")

    counts = {v: int(np.sum(labels == v)) for v in [-1, 0, 1]}
    print(f"\n  Label distribution (seconds):")
    print(f"    Rest (0)    : {counts[0]}s ({counts[0]/60:.1f} min)")
    print(f"    Stress (1)  : {counts[1]}s ({counts[1]/60:.1f} min)")
    print(f"    Discard (-1): {counts[-1]}s")

    return labels


# ──────────────────────────────────────────────
# 5. PREPROCESS ACC → 1 Hz energy
# ──────────────────────────────────────────────

def preprocess_acc(df_acc, fs_acc):
    """
    Bandpass filter ACC components, compute magnitude,
    then downsample to 1 Hz energy (mean squared per second).
    Returns: energy_1hz array
    """
    x = df_acc['acc_x'].values.astype(float)
    y = df_acc['acc_y'].values.astype(float)
    z = df_acc['acc_z'].values.astype(float)

    xf = bandpass_filter(x, 0.5, 10, fs_acc)
    yf = bandpass_filter(y, 0.5, 10, fs_acc)
    zf = bandpass_filter(z, 0.5, 10, fs_acc)

    magnitude = np.sqrt(xf**2 + yf**2 + zf**2)

    # Downsample to 1 Hz: mean squared energy per second
    samples_per_sec = fs_acc
    n_seconds = len(magnitude) // samples_per_sec
    energy_1hz = np.array([
        np.mean(magnitude[i*samples_per_sec:(i+1)*samples_per_sec]**2)
        for i in range(n_seconds)
    ])

    return energy_1hz


# ──────────────────────────────────────────────
# 6. PREPROCESS HR/IBI → 1 Hz via interpolation
# ──────────────────────────────────────────────

def preprocess_hr_ibi(df_hr, total_seconds):
    """
    The Apple Watch provides HR and IBI at irregular intervals (~every 5s).

    HR   → interpolated to 1 Hz (safe: slowly varying signal).
    IBI  → interpolated to 1 Hz (used as a proxy for beat-to-beat interval).
    RMSSD/SDNN → computed on the *sparse* IBI series using a rolling 120s
                 time window, then mapped back to a 1 Hz grid.
                 This avoids the artificial variability that interpolation
                 would introduce (interpolated IBI is almost constant between
                 two real readings, making RMSSD near-zero and SDNN inflated).

    IBI is in ms in the CSV → converted to seconds.
    """
    t_sparse   = df_hr['t_rel'].values
    hr_sparse  = df_hr['hr_bpm'].values.astype(float)
    ibi_sparse = df_hr['ibi_ms'].values.astype(float) / 1000.0  # ms → s

    # Outlier removal
    valid = (
        (hr_sparse >= 40) & (hr_sparse <= 180) &
        (ibi_sparse >= 0.3) & (ibi_sparse <= 1.5)
    )
    t_sparse   = t_sparse[valid]
    hr_sparse  = hr_sparse[valid]
    ibi_sparse = ibi_sparse[valid]

    t_uniform = np.arange(total_seconds, dtype=float)

    # HR and IBI: linear interpolation to 1 Hz
    hr_1hz  = np.interp(t_uniform, t_sparse, hr_sparse,
                        left=hr_sparse[0], right=hr_sparse[-1])
    ibi_1hz = np.interp(t_uniform, t_sparse, ibi_sparse,
                        left=ibi_sparse[0], right=ibi_sparse[-1])

    # RMSSD and SDNN: rolling window on the *sparse* IBI series
    # Window = 120s (captures ~24 IBI readings at ~1/5 Hz)
    hrv_window_sec = 120
    rmssd_sparse = np.full(len(t_sparse), np.nan)
    sdnn_sparse  = np.full(len(t_sparse), np.nan)

    for i in range(len(t_sparse)):
        t_now = t_sparse[i]
        mask  = (t_sparse >= t_now - hrv_window_sec) & (t_sparse <= t_now)
        window = ibi_sparse[mask]
        if len(window) >= 3:
            sdnn_sparse[i]  = np.std(window)
            diffs = np.diff(window)
            rmssd_sparse[i] = np.sqrt(np.mean(diffs**2))

    # Forward-fill NaNs (cold-start period), then fill any remaining with median
    for arr in [rmssd_sparse, sdnn_sparse]:
        first_valid = np.where(~np.isnan(arr))[0]
        if len(first_valid) > 0:
            arr[:first_valid[0]] = arr[first_valid[0]]  # back-fill start
        else:
            arr[:] = 0.0
        # any remaining NaN (shouldn't happen, but safety)
        arr[np.isnan(arr)] = np.nanmedian(arr)

    # Map sparse HRV back to 1 Hz grid
    rmssd_1hz = np.interp(t_uniform, t_sparse, rmssd_sparse,
                          left=rmssd_sparse[0], right=rmssd_sparse[-1])
    sdnn_1hz  = np.interp(t_uniform, t_sparse, sdnn_sparse,
                          left=sdnn_sparse[0], right=sdnn_sparse[-1])

    return hr_1hz, ibi_1hz, rmssd_1hz, sdnn_1hz


# ──────────────────────────────────────────────
# 7. PER-SUBJECT Z-SCORE NORMALISATION
# ──────────────────────────────────────────────

def normalize_per_subject(signals_dict):
    """
    Z-score normalise each signal using the subject's own mean and std.
    Same approach as WDD pipeline (avoids cross-subject leakage).
    """
    normalised = {}
    for key, sig in signals_dict.items():
        mu, sd = np.mean(sig), np.std(sig)
        normalised[key] = (sig - mu) / (sd + 1e-8)
    return normalised


# ──────────────────────────────────────────────
# 8. ASSEMBLE 1 Hz DATAFRAME
# ──────────────────────────────────────────────

def build_stress_phases(marker_times, total_seconds):
    """
    Returns a 1 Hz array with the stress phase index for each second.
    Values: -1=discard/baseline, 1=stress1, 2=stress2, 3=stress3
    """
    phases = np.full(total_seconds, -1, dtype=int)
    m = marker_times
    n = len(m)

    def set_range(start, end, val):
        s = max(0, min(int(start), total_seconds))
        e = max(0, min(int(end),   total_seconds))
        if e > s:
            phases[s:e] = val

    if n >= 4:
        set_range(m[2], m[3], 1)
    if n >= 6:
        set_range(m[4], m[5], 2)
    if n >= 7:
        end3 = int(m[7]) if n >= 8 else total_seconds
        set_range(m[6], end3, 3)

    return phases


def assemble_dataframe(signals, labels_1hz, subject_id, stress_phases=None):
    """
    Combines all 1 Hz signals and labels into a single DataFrame.
    Drops discard rows (label == -1).
    If stress_phases is provided, adds a 'stress_phase' column (1/2/3 for stress, -1 otherwise).

    NOTE: preprocess_session() always calls this with signals_raw, never
    signals_norm (its per-subject-normalised sibling) -- param renamed from
    signals_norm to signals 2026-07-25 to stop implying otherwise. See
    build_features_tabular()'s docstring for why this is fine (BL-norm is
    applied downstream, at the feature level, anchored to each subject's
    true resting baseline).
    """
    n = min(len(labels_1hz), *[len(v) for v in signals.values()])

    df_out = pd.DataFrame({
        'subject'   : subject_id,
        't_sec'     : np.arange(n),
        'label'     : labels_1hz[:n],
        'acc_wrist': signals['acc_wrist'][:n],
        'hr'        : signals['hr'][:n],
        'ibi'       : signals['ibi'][:n],
        'rmssd'     : signals['rmssd'][:n],
        'sdnn'      : signals['sdnn'][:n],
    })

    if stress_phases is not None:
        df_out['stress_phase'] = stress_phases[:n]

    # Keep only labelled rows
    df_out = df_out[df_out['label'] != -1].reset_index(drop=True)

    print(f"\n  Assembled DataFrame: {df_out.shape}")
    print(f"    Rest rows   : {(df_out['label']==0).sum()}")
    print(f"    Stress rows : {(df_out['label']==1).sum()}")
    if stress_phases is not None:
        for ph in [1, 2, 3]:
            cnt = (df_out.get('stress_phase', pd.Series()) == ph).sum()
            print(f"    Stress phase {ph}: {cnt} rows")

    return df_out


# ──────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ──────────────────────────────────────────────

def preprocess_session(filepath, subject_id,
                       n_markers=8,
                       baseline_duration_sec=300,
                       z_percentile=85,
                       min_duration_sec=8,
                       marker_min_gap_sec=120,
                       verbose=True):
    """
    Full preprocessing pipeline for a single Anxolotl session CSV.

    Parameters
    ----------
    filepath              : path to the session CSV
    subject_id            : string identifier (e.g. 'p01')
    n_markers             : expected number of wrist-raise gestures (default 8)
    baseline_duration_sec : fixed baseline duration in seconds (default 300)
    z_percentile          : z-axis percentile for arm-raised threshold (default 85)
    min_duration_sec      : minimum seconds arm must stay raised (default 8)
    marker_min_gap_sec    : minimum seconds between consecutive markers (default 120)

    Returns
    -------
    dict with keys:
        'df'          : labelled 1 Hz DataFrame (label -1 already dropped)
        'labels_full' : full label array (1 Hz, includes -1 for context)
        'marker_times': detected marker timestamps (relative, seconds)
        'signals_raw' : dict of raw 1 Hz signals before normalisation
        'signals_norm': dict of normalised 1 Hz signals
        'fs_acc'      : detected ACC sampling rate
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Preprocessing session: {filepath}")
        print(f"  Subject: {subject_id}")
        print(f"{'='*55}")

    # 1. Load
    df_acc, df_hr, fs_acc, t0 = load_session_csv(filepath)
    total_seconds = int(df_acc['t_rel'].max()) + 1

    # 2. Detect protocol markers
    marker_times = detect_markers(
        df_acc, fs_acc,
        n_markers=n_markers,
        z_percentile=z_percentile,
        min_duration_sec=min_duration_sec,
        min_gap_sec=marker_min_gap_sec,
        baseline_duration_sec=baseline_duration_sec,
    )

    # 3. Build labels and stress phases
    labels_1hz    = build_labels(marker_times, total_seconds, baseline_duration_sec)
    stress_phases = build_stress_phases(marker_times, total_seconds)

    # 4. Preprocess signals
    acc_energy_1hz = preprocess_acc(df_acc, fs_acc)
    hr_1hz, ibi_1hz, rmssd_1hz, sdnn_1hz = preprocess_hr_ibi(df_hr, total_seconds)

    signals_raw = {
        'acc_wrist': acc_energy_1hz,
        'hr'        : hr_1hz,
        'ibi'       : ibi_1hz,
        'rmssd'     : rmssd_1hz,
        'sdnn'      : sdnn_1hz,
    }

    # 5. Normalise
    signals_norm = normalize_per_subject(signals_raw)

    # 6. Assemble
    df_out = assemble_dataframe(signals_raw, labels_1hz, subject_id, stress_phases=stress_phases)

    return {
        'df'          : df_out,
        'labels_full' : labels_1hz,
        'marker_times': marker_times,
        'signals_raw' : signals_raw,
        'signals_norm': signals_norm,
        'fs_acc'      : fs_acc,
    }


# ──────────────────────────────────────────────
# 10. BATCH PROCESSING (multiple sessions)
# ──────────────────────────────────────────────

def preprocess_all_sessions(session_map, **kwargs):
    """
    Process multiple sessions and concatenate into one DataFrame.

    Parameters
    ----------
    session_map : dict {subject_id: filepath}
                  e.g. {'p01': 'data/session_0011.csv', 'p02': ...}

    Returns
    -------
    df_all    : concatenated DataFrame with all subjects
    results   : dict {subject_id: individual result dict}
    """
    results = {}
    failed = {}
    dfs = []

    for subject_id, filepath in session_map.items():
        try:
            res = preprocess_session(filepath, subject_id, **kwargs)
            results[subject_id] = res
            dfs.append(res['df'])
        except Exception as e:
            failed[subject_id] = str(e)
            print(f"\n  ✗  {subject_id} excluded: {e}")

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    print(f"\n{'='*55}")
    print(f"  All sessions combined: {df_all.shape}")
    print(f"  Subjects included : {df_all['subject'].nunique()} / {len(session_map)}")
    if failed:
        print(f"  Subjects excluded : {list(failed.keys())}")
    counts = df_all['label'].value_counts().to_dict() if len(df_all) > 0 else {}
    print(f"  Label dist        : rest={counts.get(0,0)}  stress={counts.get(1,0)}")
    print(f"{'='*55}")

    return df_all, results




# ──────────────────────────────────────────────
# 11. TABULAR FEATURES  (equivalente ao build_features_v2 do WESAD)
# ──────────────────────────────────────────────

def build_features_tabular(df_all, window_sec=60, stride_sec=30):
    """
    Extracts statistical features over sliding windows from the 1 Hz
    RAW signals (df_all's acc_wrist/hr/ibi/rmssd/sdnn columns are the
    session's signals_raw, not signals_norm -- assemble_dataframe() is
    called with signals_raw in preprocess_session(); normalize_per_subject()
    is computed but its output is never passed downstream. Confirmed
    2026-07-24/25, see results/acc_wrist_min_investigation_2026-07-24.txt.
    This is not a leak: baseline_relative_scale() (demographics_analysis.py)
    is applied downstream, at the feature level, to every canonical model
    and to SHAP alike -- it anchors each subject to their own true resting
    baseline, which is what actually matters here, and is arguably more
    correct than a whole-session z-score would be, since a whole-session
    z-score would fold stress-period statistics into the reference scale.

    Mirrors build_features_v2 from the WESAD pipeline so both datasets
    share the same feature space (minus EDA/TEMP which are not available
    from Apple Watch). WESAD/WDD's own pipelines also window raw signals
    and normalise only downstream -- this matches that design, it isn't a
    one-off deviation.

    Signals used (5 channels, all 1 Hz RAW, not per-subject normalised):
        acc_wrist, hr, ibi, rmssd, sdnn

    Features per channel (11 each = 55 total):
        mean, std, min, max, p25, p75, skew, kurtosis,
        slope, lf_hf (Welch), delta_mean

    Parameters
    ----------
    df_all      : concatenated 1 Hz DataFrame from preprocess_all_sessions
    window_sec  : window size in seconds (default 60)
    stride_sec  : stride between windows (default 30)

    Returns
    -------
    df_features : DataFrame with one row per window, columns:
                  subject, label, + feature columns
    """
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
    from scipy.signal import welch as sp_welch

    signal_cols = ['acc_wrist', 'hr', 'ibi', 'rmssd', 'sdnn']
    rows = []

    for subject, grp in df_all.groupby('subject'):
        grp = grp.sort_values('t_sec').reset_index(drop=True)
        n = len(grp)

        for i in range(window_sec, n, stride_sec):
            # Label by first sample of the window content; skip boundary windows
            # where first and last sample disagree (window spans a phase transition).
            label_first = int(grp['label'].iloc[i - window_sec])
            label_last  = int(grp['label'].iloc[i - 1])
            if label_first != label_last:
                continue
            label = label_first
            if label not in [0, 1]:
                continue

            row = {'subject': subject, 'label': label}
            if 'stress_phase' in grp.columns:
                row['stress_phase'] = int(grp['stress_phase'].iloc[i - 1])

            for col in signal_cols:
                w = grp[col].iloc[i - window_sec:i].values.astype(float)
                mu, sd = np.mean(w), np.std(w)
                w_norm = (w - mu) / (sd + 1e-8)
                sk = sp_skew(w_norm)   if sd > 1e-6 else 0.0
                ku = sp_kurtosis(w_norm) if sd > 1e-6 else 0.0

                freqs, psd = sp_welch(w, fs=1, nperseg=min(len(w), 32))
                lf = np.trapezoid(psd[(freqs >= 0.04) & (freqs <= 0.15)])
                hf = np.trapezoid(psd[(freqs >= 0.15) & (freqs <= 0.40)])
                lf_hf = lf / (hf + 1e-8)

                row.update({
                    f'{col}_mean'      : float(mu),
                    f'{col}_std'       : float(sd),
                    f'{col}_min'       : float(np.min(w)),
                    f'{col}_max'       : float(np.max(w)),
                    f'{col}_p25'       : float(np.percentile(w, 25)),
                    f'{col}_p75'       : float(np.percentile(w, 75)),
                    f'{col}_skew'      : float(sk),
                    f'{col}_kurtosis'  : float(ku),
                    f'{col}_slope'     : float(np.polyfit(np.arange(window_sec), w, 1)[0]),
                    f'{col}_lf_hf'     : float(lf_hf),
                    f'{col}_delta_mean': float(np.mean(w[window_sec//2:]) - np.mean(w[:window_sec//2])),
                })

            rows.append(row)

    df_feat = pd.DataFrame(rows)
    feat_cols = [c for c in df_feat.columns if c not in ['subject', 'label', 'stress_phase']]

    print(f"  Tabular features: {df_feat.shape}  "
          f"({len(feat_cols)} features × {len(df_feat)} windows)")
    print(f"  Label dist: {df_feat['label'].value_counts().to_dict()}")
    print(f"  Subjects  : {sorted(df_feat['subject'].unique())}")

    return df_feat, feat_cols


# ──────────────────────────────────────────────
# 12. SEQUENCE WINDOWS  (equivalente ao build_sequences do WESAD)
# ──────────────────────────────────────────────

def build_sequences(df_all, window_sec=60):
    """
    Builds 3D sequence windows for deep learning models (CNN, GRU, Transformer).

    Shape: (N_windows, n_channels, window_sec)  ← channels-first, same as WESAD

    Channels (5): acc_wrist, hr, ibi, rmssd, sdnn
    Stride     : 1 second (maximum overlap, same as WESAD build_sequences)

    Parameters
    ----------
    df_all     : concatenated 1 Hz DataFrame from preprocess_all_sessions
    window_sec : window length in seconds (default 60)

    Returns
    -------
    X_seq    : np.float32 array  (N, 5, window_sec)
    y_seq    : np.int64 array    (N,)
    groups   : np.array of str  (N,)  — subject id per window
    """
    signal_cols = ['acc_wrist', 'hr', 'ibi', 'rmssd', 'sdnn']
    X_list, y_list, g_list = [], [], []

    for subject, grp in df_all.groupby('subject'):
        grp = grp.sort_values('t_sec').reset_index(drop=True)
        signals = grp[signal_cols].values.astype(np.float32)  # (n, 5)
        labels  = grp['label'].values

        for i in range(window_sec, len(grp)):
            label = int(labels[i])
            if label not in [0, 1]:
                continue
            window = signals[i - window_sec:i]   # (window_sec, 5)
            X_list.append(window.T)               # (5, window_sec) — channels first
            y_list.append(label)
            g_list.append(subject)

    X_seq  = np.array(X_list, dtype=np.float32)
    y_seq  = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list)

    print(f"  Sequences: X={X_seq.shape}  y={y_seq.shape}")
    print(f"  Label dist: 0={int((y_seq==0).sum())}  1={int((y_seq==1).sum())}")
    print(f"  Subjects  : {sorted(set(groups))}")

    return X_seq, y_seq, groups


# ──────────────────────────────────────────────
# 13. PROCESS FOLDER  (main entry point for batch runs)
# ──────────────────────────────────────────────

def process_folder(folder_path, output_dir=None,
                   session_prefix='session_', subject_prefix='p',
                   window_sec=60, stride_sec=30,
                   exclude=None,
                   save_pickles=True, **session_kwargs):
    """
    Reads all session_XXXX.csv files from a folder, preprocesses each one,
    builds tabular and sequence datasets, and optionally saves two pickles:

        processed_data_anxolotl.pkl       — tabular features (sklearn-ready)
        processed_data_anxolotl_seq.pkl   — sequence windows (pytorch-ready)

    Subject id is derived from the file number:
        session_0011.csv  →  p11
        session_0020.csv  →  p20

    Parameters
    ----------
    folder_path    : path to folder containing session CSVs
    output_dir     : where to save pickles (defaults to folder_path)
    session_prefix : filename prefix to match (default 'session_')
    subject_prefix : prefix for subject ids (default 'p')
    window_sec     : window size for feature extraction (default 60)
    stride_sec     : stride for tabular features (default 30)
    exclude        : list of subject ids to exclude manually (e.g. ['p6', 'p24'])
    save_pickles   : whether to write .pkl files (default True)
    **session_kwargs : passed to preprocess_session (e.g. z_percentile=80)

    Returns
    -------
    df_tabular  : tabular feature DataFrame
    feat_cols   : list of feature column names
    X_seq       : sequence array (N, 5, window_sec)
    y_seq       : label array (N,)
    groups_seq  : subject array (N,)
    """
    import glob, os

    folder_path = os.path.abspath(folder_path)
    output_dir  = output_dir or folder_path

    # ── Discover session files ──────────────────
    pattern = os.path.join(folder_path, f'{session_prefix}*.csv')
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {folder_path}"
        )

    # Build subject map: session_0011.csv → p11
    session_map = {}
    for fp in files:
        basename = os.path.splitext(os.path.basename(fp))[0]  # 'session_0011'
        number   = basename.replace(session_prefix, '').lstrip('0') or '0'
        subject_id = f"{subject_prefix}{number}"
        session_map[subject_id] = fp

    # Apply manual exclusion list
    if exclude:
        for sid in list(exclude):
            if sid in session_map:
                del session_map[sid]
                print(f"  (manually excluded: {sid})")

    print(f"Found {len(session_map)} session(s):")
    for sid, fp in session_map.items():
        print(f"  {sid}  ←  {os.path.basename(fp)}")

    # ── Preprocess all sessions ─────────────────
    print(f"\n{'='*55}")
    print("  STEP 1 — Preprocessing sessions")
    print(f"{'='*55}")
    df_all, results = preprocess_all_sessions(session_map, **session_kwargs)

    if df_all.empty:
        raise RuntimeError("All sessions failed preprocessing.")

    # ── Build tabular features ──────────────────
    print(f"\n{'='*55}")
    print("  STEP 2 — Building tabular features")
    print(f"{'='*55}")
    df_tabular, feat_cols = build_features_tabular(
        df_all, window_sec=window_sec, stride_sec=stride_sec
    )

    # ── Build sequences ─────────────────────────
    print(f"\n{'='*55}")
    print("  STEP 3 — Building sequence windows")
    print(f"{'='*55}")
    X_seq, y_seq, groups_seq = build_sequences(df_all, window_sec=window_sec)

    # ── Save pickles ────────────────────────────
    if save_pickles:
        os.makedirs(output_dir, exist_ok=True)

        tabular_path = os.path.join(output_dir, 'processed_data_anxolotl.pkl')
        seq_path     = os.path.join(output_dir, 'processed_data_anxolotl_seq.pkl')

        import pickle
        with open(tabular_path, 'wb') as f:
            pickle.dump({
                'df'          : df_tabular,
                'X'           : df_tabular[feat_cols],
                'y'           : df_tabular['label'],
                'groups'      : df_tabular['subject'],
                'feature_cols': feat_cols,
            }, f)

        with open(seq_path, 'wb') as f:
            pickle.dump({
                'X_seq'     : X_seq,
                'y_seq'     : y_seq,
                'groups_seq': groups_seq,
            }, f)

        print(f"\n  Saved:")
        print(f"    {tabular_path}")
        print(f"    {seq_path}")

    print(f"\n{'='*55}")
    print("  DONE")
    print(f"  Tabular : {df_tabular.shape[0]} windows  ×  {len(feat_cols)} features")
    print(f"  Sequences: {X_seq.shape}")
    print(f"  Subjects : {df_all['subject'].nunique()}")
    print(f"{'='*55}")

    return df_tabular, feat_cols, X_seq, y_seq, groups_seq
# ──────────────────────────────────────────────
# EXAMPLE USAGE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        # Folder mode: process_folder <folder_path> [output_dir]
        folder   = sys.argv[1]
        out_dir  = sys.argv[2] if len(sys.argv) >= 3 else folder
        # p6 excluded: no valid arm-raise gestures detected (z signal too flat).
        # Add other subject ids here if manual exclusion is needed.
        process_folder(folder, output_dir=out_dir, exclude=['p6'])
    else:
        # Single session mode: script.py <csv_path> [subject_id]
        filepath   = sys.argv[1] if len(sys.argv) > 1 else "session_0011.csv"
        subject_id = sys.argv[2] if len(sys.argv) > 2 else "p01"
        result = preprocess_session(filepath, subject_id)
        df = result['df']
        print(f"\nFirst rows:\n{df.head()}")
        print(f"\nSignal stats:\n{df.describe().round(3)}")