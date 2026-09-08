"""
Gesture-detection validation, part 1: parameters, ground-truth search, and
per-subject marker/tail-time report. No plots yet (those come after the user
reviews this output, per explicit instruction).
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from anxolotl_preprocessing import load_session_csv, detect_markers

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'gesture_validation_2026-07-24.txt')

lines = []
def w(s=''):
    print(s)
    lines.append(s)

w('='*78)
w('GESTURE-DETECTION VALIDATION — 2026-07-24')
w('='*78)
w()
w('-'*78)
w('1. ALGORITHM PARAMETERS (detect_markers, 3_anxolotl/anxolotl_preprocessing.py:132)')
w('-'*78)
w('Signature defaults actually used in production (preprocess_session never')
w('overrides bridge_gap_sec or relax_level_frac -- see call site at line 652):')
w()
w('  z_percentile          = 85     (adaptive threshold: 85th pctile of the')
w('                                   session\'s OWN 1s-median z-axis distribution)')
w('  min_duration_sec      = 8      (arm must stay above threshold >= 8s)')
w('  min_gap_sec           = 120    (min spacing between markers 2..8)')
w('  baseline_duration_sec = 300    (min spacing enforced between marker 1 and 2)')
w('  bridge_gap_sec        = 0      (ACTUAL default; docstring says 3 -- MISMATCH,')
w('                                   no gap-bridging happens in production runs)')
w('  relax_level_frac      = 0.45   (ACTUAL default; docstring says 0.6 -- MISMATCH)')
w('  min_gesture_amplitude = 0.15   (min resting->raised z amplitude to attempt relaxed pass)')
w('  relax_min_duration_sec= 5      (min duration for a relaxed-pass candidate)')
w('  relax_max_std_frac    = 0.35   (max variance, as frac of amplitude, for a')
w('                                   relaxed candidate to be accepted as a plateau)')
w()
w('DOCSTRING/CODE MISMATCH FLAGGED: the docstring for bridge_gap_sec and')
w('relax_level_frac states different default values than the function signature')
w('actually uses. Since preprocess_session() calls detect_markers() without')
w('overriding either, the REAL behaviour in every one of the 37 processed')
w('sessions is bridge_gap_sec=0 (no dropout-bridging at all) and')
w('relax_level_frac=0.45, not the values the docstring describes.')
w()
w('Core detection logic:')
w('  1. z_1s = 1-second median of raw ACC z-axis (robust to single-sample noise)')
w('  2. z_threshold = percentile(z_1s, 85)  -- adaptive per session, no fixed volts')
w('  3. raised = z_1s > z_threshold; find contiguous True-runs >= 8s long')
w('  4. Greedy protocol-aware selection: marker1->2 gap >= 300s, all others >= 120s')
w('  5. Relaxed second pass on the unexplored tail if < 8 markers found, using a')
w('     per-subject-calibrated threshold (45% of the subject\'s own resting->raised')
w('     amplitude), only if that subject\'s confirmed gestures show amplitude >= 0.15')
w('  6. Structural sanity check: inter-marker spacing must be 100-400s or the whole')
w('     session is rejected as having no valid gestures')
w('  7. Minimum 4 markers required to accept a session at all (else ValueError)')
w()
w('-'*78)
w('2. INDEPENDENT GROUND TRUTH — SEARCHED, NONE FOUND')
w('-'*78)
w('Checked and ruled out:')
w('  - Video/audio recordings: none found anywhere in the repo (searched for')
w('    .mp4/.mov/.avi)')
w('  - Investigator notes / session logs / timestamp files: none found')
w('    (searched for *notes*, *log_sessao*, *investigador*, *timestamp*)')
w('  - Manual annotation column in the watch session CSVs: does not exist --')
w('    columns are exactly [timestamp_unix, hr_bpm, ibi_ms, steps, calories_kcal,')
w('    distance_m, acc_x, acc_y, acc_z], no phase/event/marker column')
w('  - Data_tese.xlsx: only has Questionario/VAS/EADS-21 sheets (self-report')
w('    questionnaires), no session-timing sheet')
w('  - BIOPAC event markers (a real candidate -- BIOPAC AcqKnowledge supports a')
w('    manual marker button an investigator could press during acquisition):')
w('    checked via bioread .event_markers on all sampled participants -- every')
w('    single session has EXACTLY ONE marker, type_code \'apnd\' at sample_index=0,')
w('    which is the auto-generated "recording started" marker, not a manual')
w('    investigator annotation of any protocol phase transition.')
w()
w('CONCLUSION: there is no independent ground-truth record of real phase timing')
w('anywhere in this dataset. Task 3 (compare detected markers against ground')
w('truth) cannot be performed -- proceeding to Task 4 (visual-inspection plots)')
w('and Task 5 (marker distribution + tail-time analysis) as the fallback you')
w('specified.')
w()

# ── Task 5: per-subject marker counts + tail-time-after-last-marker ────────
w('-'*78)
w('5. PER-SUBJECT MARKER DISTRIBUTION AND TAIL-TIME ANALYSIS')
w('-'*78)

csv_map = {}
for fp in sorted(glob.glob(f'{SESSIONS_DIR}/session_*.csv')):
    sid = 'p' + os.path.basename(fp).replace('session_0', '').replace('session_', '').replace('.csv', '').lstrip('0')
    csv_map[sid] = fp
if 'p6' in csv_map:
    del csv_map['p6']

rows = []
for subj, fp in sorted(csv_map.items(), key=lambda kv: int(kv[0][1:])):
    try:
        df_acc, df_hr, fs_acc, t0 = load_session_csv(fp)
        total_seconds = float(df_acc['t_rel'].max())
        try:
            marker_times = detect_markers(df_acc, fs_acc)
        except ValueError as e:
            rows.append({'subject': subj, 'n_markers': 0, 'last_marker_t': np.nan,
                         'total_duration': total_seconds, 'tail_after_last': np.nan,
                         'note': f'FAILED: {e}'})
            continue
        n = len(marker_times)
        last_t = float(marker_times[-1])
        tail = total_seconds - last_t
        rows.append({'subject': subj, 'n_markers': n, 'last_marker_t': last_t,
                     'total_duration': total_seconds, 'tail_after_last': tail, 'note': ''})
    except Exception as e:
        rows.append({'subject': subj, 'n_markers': -1, 'last_marker_t': np.nan,
                     'total_duration': np.nan, 'tail_after_last': np.nan,
                     'note': f'LOAD ERROR: {e}'})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'results',
                        'gesture_validation_per_subject.csv'), index=False)

w()
w('Distribution of n_markers detected (excludes p6, manually excluded earlier):')
dist = df[df.n_markers >= 0].n_markers.value_counts().sort_index()
for n, c in dist.items():
    w(f'  n_markers = {n} : {c} subjects')
w(f'  Total subjects analysed: {len(df)}')
w()

w('Full per-subject table:')
w(df.to_string(index=False))
w()

w('-'*78)
w('Tail-time-after-last-marker for incomplete sessions (n_markers < 8)')
w('-'*78)
w('Distinguishes "gesture genuinely not performed" (short/no tail left in the')
w('recording after the last confirmed marker -- session simply ended) from')
w('"gesture performed but not detected" (long tail exists with plenty of time')
w('for another task+breathing block, yet no marker was found in it).')
w()
incomplete = df[(df.n_markers >= 4) & (df.n_markers < 8)].copy()
incomplete = incomplete.sort_values('tail_after_last')
w(incomplete[['subject', 'n_markers', 'last_marker_t', 'total_duration', 'tail_after_last']].to_string(index=False))
w()
w('Interpretation heuristic used here: min_gap_sec=120s is the minimum spacing')
w('the algorithm requires between markers, and a real task+breathing block runs')
w('120-400s per the structural sanity check in detect_markers(). So:')
w('  tail_after_last <  120s : very likely "gesture not performed" -- session')
w('                            ended before another block could physically fit')
w('  tail_after_last >= 120s : recording had room for another block -- worth')
w('                            flagging as a possible missed detection, not')
w('                            just a skipped gesture')
w()
short_tail = incomplete[incomplete.tail_after_last < 120]
long_tail = incomplete[incomplete.tail_after_last >= 120]
w(f'  Incomplete subjects with tail < 120s ("not performed"):  {len(short_tail)}  {list(short_tail.subject)}')
w(f'  Incomplete subjects with tail >= 120s ("possible miss") : {len(long_tail)}  {list(long_tail.subject)}')

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines))
print(f'\n\nSaved: {OUT_PATH}')
print(f'Saved: results/gesture_validation_per_subject.csv')
