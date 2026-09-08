"""
temporal_control.py — control experiments for the "does BL-norm just encode
temporal position?" question raised before the defense.

Reuses, UNCHANGED, the canonical ET-1000 / leak-fixed pipeline from
3_anxolotl/_canonical_sensitivity_common.py: same feature extraction
(build_canonical_xy, 60s/30s windows, phase3 excluded -> 686 windows/55
feat/37 subjects), same normalisation (baseline_relative_scale, leak-fixed
2026-07-19), same model (ExtraTrees n_estimators=1000, max_features=0.5,
class_weight=balanced, random_state=42), same LOSO loop (run_loso_et1000).
Nothing in 3_anxolotl/ is modified by this script.

Teste A (temporal confounder): replace the baseline/stress label with a
  purely positional one -- "first third of the subject's own chronological
  window sequence" (label 0) vs "last third" (label 1) -- discarding the
  middle third, then run the exact same LOSO pipeline (features + BL-norm +
  ET-1000). If the model reads temporal position rather than physiology, this
  should score close to the canonical ~0.90+; if it reads physiology, this
  synthetic task has no physiological signal to exploit and should land near
  chance (Macro F1 ~0.5, AUROC ~0.5).

Teste B (final-rest normalisation reference): SKIPPED. Checked directly
  against results/df_all_1hz_cache.pkl: of the 4 subjects who reach stress
  phase 3 (p12, p22, p31, p36), only p22 has any trailing label=0/
  stress_phase=-1 segment recorded (71s, t=1294-1365s) -- the other three
  subjects' recordings end still inside phase 3. Per
  results/canonical_numbers_for_thesis.txt sec.3 (marker coverage), only
  1/37 subjects has the marker-8 gesture confirmed, which is what would mark
  the true start of a final-rest period. There is no way to recover a
  final-rest reference for the other 36 subjects from the raw session CSVs
  either, since without marker 8 there is no known boundary between "still
  task 3" and "now resting" in the continuous recording. A LOSO-wide
  normalisation-reference swap needs a final-rest segment in every subject;
  with usable data in exactly 1/37, the test cannot be run. See this run's
  output file for this note recorded alongside the two runnable tests.

Teste C (marker-confirmation sensitivity, HI vs LO): not rerun -- this
  ablation already exists (results/marker_sensitivity_et1000_leakfixed.txt,
  same canonical config, same seed). This script reads that file and quotes
  its HI/LO/pooled numbers verbatim, so all three tests land side by side in
  one place. If that file's numbers should ever be regenerated, rerun
  3_anxolotl/marker_sensitivity_et1000_leakfixed.py directly -- this script
  does not recompute them.

Teste A2 (added, not in the original spec -- see below for why): Teste A as
  literally specified turned out to be uninformative. Checked directly: the
  synthetic "first third / last third" label agrees with the TRUE
  baseline/stress label on 99.77% of kept windows (429/430) -- because this
  protocol's stress phases start only after the initial ~5min baseline and
  there is essentially no final-rest to blur the picture (see Teste B), so
  "early in session" and "late in session" are almost the same partition as
  "baseline" and "stress" by construction. A high score on Teste A therefore
  cannot distinguish "model reads the clock" from "model reads physiology"
  -- both hypotheses predict the same labels almost everywhere. Teste A2
  isolates time from physiology properly: WITHIN each subject's TRUE label
  (baseline windows only, stress windows only), split chronologically into
  an early half and a late half (temporal pseudo-label), and test whether
  the model can tell early-vs-late apart *without the true label ever
  changing*. If it can (Macro F1 well above ~0.5), that is real evidence of
  a temporal/drift artifact independent of physiology; if it cannot
  (~0.5), the earlier high Teste-A score is fully explained by the label
  collinearity above, not by a temporal confound.

Output: results/temporal_control_2026-07-24.txt (refuses to overwrite if the
file already exists -- run once).
"""
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANX_DIR = os.path.join(REPO_ROOT, '3_anxolotl')
sys.path.insert(0, ANX_DIR)

import _canonical_sensitivity_common as csc  # noqa: E402
from demographics_analysis import baseline_relative_scale  # noqa: E402

OUT_PATH = os.path.join(REPO_ROOT, 'results', 'temporal_control_2026-07-24.txt')
MARKER_SENSITIVITY_PATH = os.path.join(REPO_ROOT, 'results',
                                        'marker_sensitivity_et1000_leakfixed.txt')

SEED = 42


def teste_a_temporal_confounder(X, y_true, groups, subjects):
    """
    Builds a purely positional pseudo-label per subject: first third of that
    subject's own chronological window run = 0, last third = 1, middle third
    discarded. y_true is only used to confirm we are NOT reusing the real
    label anywhere. Reuses run_loso_et1000 + baseline_relative_scale
    unchanged -- same model, same normalisation, same LOSO loop as canonical.
    """
    keep_mask = np.zeros(len(groups), dtype=bool)
    y_temporal = np.full(len(groups), -1, dtype=int)

    for subj in subjects:
        idx = np.where(groups == subj)[0]  # contiguous, chronological (see docstring)
        n = len(idx)
        third = n // 3
        if third < 1:
            raise RuntimeError(f'subject {subj} has too few windows ({n}) to split into thirds')
        first_idx = idx[:third]
        last_idx = idx[-third:]
        keep_mask[first_idx] = True
        keep_mask[last_idx] = True
        y_temporal[first_idx] = 0
        y_temporal[last_idx] = 1

    X_sub = X[keep_mask]
    y_sub = y_temporal[keep_mask]
    groups_sub = groups[keep_mask]
    assert (y_sub >= 0).all()

    n_first = int((y_sub == 0).sum())
    n_last = int((y_sub == 1).sum())

    fold_f1, auroc, _, _ = csc.run_loso_et1000(X_sub, y_sub, groups_sub, subjects,
                                                baseline_relative_scale)
    f1_vals = np.array(list(fold_f1.values()))

    agreement_with_true_label = float((y_true[keep_mask] == y_sub).mean())

    return {
        'n_windows_kept': int(keep_mask.sum()),
        'n_windows_total': len(groups),
        'n_first_third': n_first,
        'n_last_third': n_last,
        'macro_f1_mean': float(f1_vals.mean()),
        'macro_f1_sd': float(f1_vals.std(ddof=1)),
        'auroc': float(auroc),
        'agreement_with_true_label': agreement_with_true_label,
    }


def teste_a2_within_class_temporal_drift(X, y_true, groups, subjects):
    """
    Isolates temporal position from physiology: WITHIN each subject's own
    true label (baseline windows only, stress windows only), split
    chronologically into an early half (temporal=0) and late half
    (temporal=1). The true label never changes within either half, so any
    ability to tell early from late is a pure temporal/drift artifact, not
    a physiological difference. Subjects/classes with <2 windows in a class
    contribute nothing from that class (nothing to split).
    """
    keep_mask = np.zeros(len(groups), dtype=bool)
    y_temporal = np.full(len(groups), -1, dtype=int)

    for subj in subjects:
        for true_class in (0, 1):
            idx = np.where((groups == subj) & (y_true == true_class))[0]
            n = len(idx)
            half = n // 2
            if half < 1:
                continue  # not enough windows of this class for this subject to split
            early_idx = idx[:half]
            late_idx = idx[-half:]
            keep_mask[early_idx] = True
            keep_mask[late_idx] = True
            y_temporal[early_idx] = 0
            y_temporal[late_idx] = 1

    X_sub = X[keep_mask]
    y_sub = y_temporal[keep_mask]
    groups_sub = groups[keep_mask]
    true_label_sub = y_true[keep_mask]
    assert (y_sub >= 0).all()

    # sanity: true label distribution must be non-degenerate on both sides
    # of the temporal split (otherwise this reduces back to Teste A)
    cross = {}
    for tc in (0, 1):
        for tl in (0, 1):
            cross[(tc, tl)] = int(((y_sub == tc) & (true_label_sub == tl)).sum())

    fold_f1, auroc, _, _ = csc.run_loso_et1000(X_sub, y_sub, groups_sub, subjects,
                                                baseline_relative_scale)
    f1_vals = np.array(list(fold_f1.values()))
    return {
        'n_windows_kept': int(keep_mask.sum()),
        'n_windows_total': len(groups),
        'cross_tab_temporal_x_truelabel': cross,
        'macro_f1_mean': float(f1_vals.mean()),
        'macro_f1_sd': float(f1_vals.std(ddof=1)),
        'auroc': float(auroc),
    }


def teste_c_marker_sensitivity_readout():
    """Reads back the existing HI/LO marker-sensitivity ablation verbatim (not rerun)."""
    if not os.path.exists(MARKER_SENSITIVITY_PATH):
        return None
    with open(MARKER_SENSITIVITY_PATH) as f:
        text = f.read()
    rows = {}
    for label, pattern in [
        ('HI', r'HI \(n_markers>=6\)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'),
        ('LO', r'LO \(n_markers<=5\)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'),
    ]:
        m = re.search(pattern, text)
        if m:
            n_subj, n_testwin, macro_f1, sd, auroc = m.groups()
            rows[label] = {
                'n_subj': int(n_subj), 'n_testwin': int(n_testwin),
                'macro_f1': float(macro_f1), 'sd': float(sd), 'auroc': float(auroc),
            }
    m_pooled = re.search(r'MacroF1=([\d.]+)\s+SD=([\d.]+)\s+AUROC=([\d.]+)', text)
    if m_pooled:
        rows['pooled_reference'] = {
            'macro_f1': float(m_pooled.group(1)),
            'sd': float(m_pooled.group(2)),
            'auroc': float(m_pooled.group(3)),
        }
    return rows


def main():
    if os.path.exists(OUT_PATH):
        raise FileExistsError(
            f'{OUT_PATH} already exists -- refusing to overwrite. '
            f'Delete it manually first if you really want to rerun.'
        )

    print('Loading 1Hz cache and building canonical feature set (60s/30s, phase3 excluded)...')
    df_all = csc.load_cache()
    X, y, groups, subjects, feat_cols, n_before, n_after = csc.build_canonical_xy(
        df_all, window_sec=60, stride_sec=30, exclude_phase3=True)
    print(f'Canonical set: {n_before} windows before phase3 exclusion, '
          f'{n_after} after; {len(feat_cols)} features; {len(subjects)} subjects.')

    print('\n=== Teste A: temporal confounder (first third vs last third) ===')
    a = teste_a_temporal_confounder(X, y, groups, subjects)
    print(f"  Windows kept: {a['n_windows_kept']}/{a['n_windows_total']} "
          f"(first-third={a['n_first_third']}, last-third={a['n_last_third']}, middle discarded)")
    print(f"  Macro F1 = {a['macro_f1_mean']:.4f} (SD={a['macro_f1_sd']:.4f})   "
          f"AUROC = {a['auroc']:.4f}")
    print(f"  Agreement with TRUE label: {a['agreement_with_true_label']:.4f} "
          f"-- if this is near 1.0, Teste A alone cannot distinguish 'reads time' "
          f"from 'reads physiology' (see Teste A2 below)")

    print('\n=== Teste A2 (added): within-class temporal drift (early-vs-late, label held fixed) ===')
    a2 = teste_a2_within_class_temporal_drift(X, y, groups, subjects)
    print(f"  Windows kept: {a2['n_windows_kept']}/{a2['n_windows_total']}")
    print(f"  Cross-tab (temporal_label, true_label) -> count: {a2['cross_tab_temporal_x_truelabel']}")
    print(f"  Macro F1 = {a2['macro_f1_mean']:.4f} (SD={a2['macro_f1_sd']:.4f})   "
          f"AUROC = {a2['auroc']:.4f}")

    print('\n=== Teste B: final-rest normalisation reference ===')
    print('  SKIPPED -- final-rest segment usable for only 1/37 subjects (p22, 71s). See docstring.')

    print('\n=== Teste C: marker-confirmation sensitivity (HI vs LO), read from existing file ===')
    c = teste_c_marker_sensitivity_readout()
    if c is None:
        print(f'  Could not find {MARKER_SENSITIVITY_PATH}')
    else:
        for k in ('HI', 'LO', 'pooled_reference'):
            if k in c:
                print(f'  {k}: {c[k]}')

    with open(OUT_PATH, 'w') as f:
        f.write('=' * 100 + '\n')
        f.write('TEMPORAL CONTROL EXPERIMENTS -- does BL-norm/ET-1000 read temporal position\n')
        f.write('instead of physiological response?\n')
        f.write('Generated: 2026-07-24\n')
        f.write('=' * 100 + '\n\n')
        f.write('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, '
                'class_weight=balanced, random_state=42)\n')
        f.write('Normalisation : baseline_relative_scale -- LEAK-FIXED 2026-07-19 '
                '(3_anxolotl/demographics_analysis.py), unchanged\n')
        f.write('Features      : 55 (5 wrist channels x 11 stats), unchanged canonical extraction\n')
        f.write('CV            : LOSO, 37 subjects, unchanged canonical loop '
                '(3_anxolotl/_canonical_sensitivity_common.py)\n')
        f.write(f'Seed          : {SEED}\n')
        f.write(f'Canonical windows: {n_after} (of {n_before} before phase3 exclusion)\n\n')

        f.write('-' * 100 + '\n')
        f.write('TESTE A -- temporal confounder: label = first third vs last third of each\n')
        f.write("subject's own chronological window sequence (middle third discarded).\n")
        f.write('Real baseline/stress label is NOT used anywhere in this test.\n')
        f.write('Interpretation: MacroF1/AUROC near ~0.90-1.0 => model can read session position\n')
        f.write('alone; near ~0.5 => no positional signal for it to exploit.\n')
        f.write('-' * 100 + '\n')
        f.write(f"Windows kept        : {a['n_windows_kept']} / {a['n_windows_total']} "
                f"(first-third={a['n_first_third']}, last-third={a['n_last_third']})\n")
        f.write(f"Macro F1            : {a['macro_f1_mean']:.4f}  (SD={a['macro_f1_sd']:.4f})\n")
        f.write(f"AUROC               : {a['auroc']:.4f}\n")
        f.write(f"Agreement w/ TRUE label (first-third==baseline & last-third==stress): "
                f"{a['agreement_with_true_label']:.4f}\n\n")
        f.write("*** CAVEAT: this agreement is 99.77% (429/430 windows). Teste A's synthetic\n")
        f.write("label is almost identical to the TRUE baseline/stress label by construction\n")
        f.write("(stress phases start only after the initial baseline, and there is no\n")
        f.write("substantial final-rest to blur the boundary -- see Teste B). A high score here\n")
        f.write("is therefore expected regardless of whether the model reads time or physiology\n")
        f.write("-- it CANNOT be used alone to conclude a temporal confound exists. See Teste A2,\n")
        f.write("added below, which isolates the two properly. ***\n\n")

        f.write('-' * 100 + '\n')
        f.write("TESTE A2 (added, not in the original spec -- see script docstring for why Teste A\n")
        f.write("alone is uninformative here). Within each subject's TRUE label (baseline-only,\n")
        f.write("stress-only), splits chronologically into an early half (temporal=0) and late\n")
        f.write("half (temporal=1). True label is held fixed within each half -- so this isolates\n")
        f.write("pure temporal/drift signal from physiology. Interpretation: MacroF1/AUROC near\n")
        f.write("~0.5 => no temporal drift signal once physiology is held constant (reassuring);\n")
        f.write("well above ~0.5 => real evidence the model/normalisation carries a temporal\n")
        f.write("artifact independent of the true label.\n")
        f.write('-' * 100 + '\n')
        f.write(f"Windows kept        : {a2['n_windows_kept']} / {a2['n_windows_total']}\n")
        f.write(f"Cross-tab (temporal_label, true_label) -> count: "
                f"{a2['cross_tab_temporal_x_truelabel']}\n")
        f.write(f"Macro F1            : {a2['macro_f1_mean']:.4f}  (SD={a2['macro_f1_sd']:.4f})\n")
        f.write(f"AUROC               : {a2['auroc']:.4f}\n\n")

        f.write('-' * 100 + '\n')
        f.write('TESTE B -- final-rest as normalisation reference: SKIPPED, not executable.\n')
        f.write('-' * 100 + '\n')
        f.write('Checked results/df_all_1hz_cache.pkl directly: of the 4 subjects reaching\n')
        f.write('stress phase 3 (p12, p22, p31, p36), only p22 has any trailing label=0 /\n')
        f.write('stress_phase=-1 segment recorded (71s, t=1294-1365s); the other 3 recordings\n')
        f.write('end still inside phase 3. Per results/canonical_numbers_for_thesis.txt sec.3,\n')
        f.write('only 1/37 subjects has the marker-8 gesture confirmed (the marker that would\n')
        f.write('demarcate the true start of a final-rest period) -- not recoverable from raw\n')
        f.write('session CSVs for the other 36 subjects either, since without marker 8 there is\n')
        f.write('no known boundary between "still task 3" and "now resting" in the continuous\n')
        f.write('recording. A LOSO-wide reference swap needs a final-rest segment in every\n')
        f.write('subject; usable data exists in 1/37, so the test cannot be run as specified.\n\n')

        f.write('-' * 100 + '\n')
        f.write('TESTE C -- marker-confirmation sensitivity (HI n_markers>=6 vs LO n_markers<=5).\n')
        f.write(f'Not rerun here -- verbatim readout of {MARKER_SENSITIVITY_PATH}\n')
        f.write('(same canonical config/seed). Included for side-by-side comparison with A.\n')
        f.write('-' * 100 + '\n')
        if c is None:
            f.write('Source file not found.\n')
        else:
            if 'HI' in c:
                f.write(f"HI (n_markers>=6)   : n_subj={c['HI']['n_subj']}  "
                        f"n_testwin={c['HI']['n_testwin']}  MacroF1={c['HI']['macro_f1']:.4f}  "
                        f"SD={c['HI']['sd']:.4f}  AUROC={c['HI']['auroc']:.4f}\n")
            if 'LO' in c:
                f.write(f"LO (n_markers<=5)   : n_subj={c['LO']['n_subj']}  "
                        f"n_testwin={c['LO']['n_testwin']}  MacroF1={c['LO']['macro_f1']:.4f}  "
                        f"SD={c['LO']['sd']:.4f}  AUROC={c['LO']['auroc']:.4f}\n")
            if 'pooled_reference' in c:
                p = c['pooled_reference']
                f.write(f"All 37 (pooled ref) : MacroF1={p['macro_f1']:.4f}  "
                        f"SD={p['sd']:.4f}  AUROC={p['auroc']:.4f}\n")
        f.write('\n')

        f.write('=' * 100 + '\n')
        f.write('SUMMARY LINE (one row per test)\n')
        f.write('=' * 100 + '\n')
        f.write(f"{'Test':<10}{'Description':<45}{'MacroF1':>10}{'AUROC':>10}\n")
        f.write(f"{'A':<10}{'1st-vs-last-third (99.8% == true label; see caveat)':<45}"
                f"{a['macro_f1_mean']:>10.4f}{a['auroc']:>10.4f}\n")
        f.write(f"{'A2':<10}{'within-class early-vs-late (label held fixed)':<45}"
                f"{a2['macro_f1_mean']:>10.4f}{a2['auroc']:>10.4f}\n")
        f.write(f"{'B':<10}{'SKIPPED (final-rest not recoverable)':<45}{'--':>10}{'--':>10}\n")
        if c and 'HI' in c and 'LO' in c:
            f.write(f"{'C-HI':<10}{'marker HI (n_markers>=6)':<45}"
                    f"{c['HI']['macro_f1']:>10.4f}{c['HI']['auroc']:>10.4f}\n")
            f.write(f"{'C-LO':<10}{'marker LO (n_markers<=5)':<45}"
                    f"{c['LO']['macro_f1']:>10.4f}{c['LO']['auroc']:>10.4f}\n")

    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
