"""
Gesture-detection validation, Task 4: per-session z-axis plots with detected
markers overlaid, for manual visual inspection. Subjects chosen (per user):
  p22 (8/8, only complete session)
  p12 (7/8, short tail -- 25s, "not performed")
  p21 (5/8, longest tail -- 373s, "possible miss")
  p24 (algorithm fails entirely -- 3 gestures, below the 4-marker minimum)
  p15 (5/8, shortest tail among the n_markers=5 group -- 321s)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anxolotl_preprocessing import load_session_csv, detect_markers, _find_raised_segments

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'gesture_validation_plots')
os.makedirs(OUT_DIR, exist_ok=True)

SUBJECTS = {
    'p22': 'session_0022.csv',
    'p12': 'session_0012.csv',
    'p21': 'session_0021.csv',
    'p24': 'session_0024.csv',
    'p15': 'session_0015.csv',
}

LABELS = ["baseline start", "breathing 1", "stress 1", "breathing 2",
          "stress 2", "breathing 3", "stress 3", "final baseline"]


def compute_z1s_and_threshold(df_acc, fs_acc, z_percentile=85):
    z = df_acc['acc_z'].values
    t = df_acc['t_rel'].values
    w = max(1, int(fs_acc * 1.0))
    n_windows = len(z) // w
    z_1s = np.array([np.median(z[i*w:(i+1)*w]) for i in range(n_windows)])
    t_1s = np.array([t[i*w] for i in range(n_windows)])
    z_threshold = np.percentile(z_1s, z_percentile)
    return t_1s, z_1s, z_threshold


for subj, fname in SUBJECTS.items():
    fp = os.path.join(SESSIONS_DIR, fname)
    df_acc, df_hr, fs_acc, t0 = load_session_csv(fp)
    t_1s, z_1s, z_threshold = compute_z1s_and_threshold(df_acc, fs_acc)

    try:
        marker_times = detect_markers(df_acc, fs_acc)
        status = f'{len(marker_times)}/8 markers accepted'
    except ValueError as e:
        marker_times = np.array([])
        status = f'DETECTION FAILED: {e}'

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t_1s, z_1s, color='steelblue', lw=0.8, label='ACC z-axis (1s median)')
    ax.axhline(z_threshold, color='gray', ls='--', lw=1,
               label=f'threshold (p85 of session) = {z_threshold:.3f}')

    raised = z_1s > z_threshold
    segs = _find_raised_segments(raised, t_1s, min_duration_sec=8, bridge_gap_sec=0)
    for i, r in enumerate(raised):
        pass
    # shade all raised runs >=8s for visual context, not just accepted markers
    in_run = False
    run_start = None
    for i in range(len(raised)):
        if raised[i] and not in_run:
            in_run = True; run_start = t_1s[i]
        elif not raised[i] and in_run:
            in_run = False
            if t_1s[i] - run_start >= 8:
                ax.axvspan(run_start, t_1s[i], color='orange', alpha=0.15)
    if in_run and (t_1s[-1] - run_start) >= 8:
        ax.axvspan(run_start, t_1s[-1], color='orange', alpha=0.15)

    for i, mt in enumerate(marker_times):
        lbl = LABELS[i] if i < len(LABELS) else f'marker {i+1}'
        ax.axvline(mt, color='red', lw=1.5)
        ax.text(mt, ax.get_ylim()[1], f' {i+1}: {lbl}', rotation=90,
                va='top', ha='left', fontsize=8, color='red')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ACC z-axis (median-filtered, 1s)')
    ax.set_title(f'{subj} — {status} — session duration {t_1s[-1]:.0f}s')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'{subj}_gesture_detection.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'{subj}: {status} -> saved {out_path}')

print(f'\nAll plots saved to {OUT_DIR}/')
