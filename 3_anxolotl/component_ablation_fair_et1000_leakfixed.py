"""
Fairer re-run of the calibration and demographics component ablations --
canonical ET-1000 configuration. Extends (does NOT replace or overwrite)
results/component_ablation_et1000_leakfixed.txt, which flagged two
methodological confounds worth controlling for:

  PART 1 -- ADAPTIVE CALIBRATION
    The original "with calibration" vs "without calibration" comparison
    scored the two conditions on different-sized eval sets (686 vs 537
    windows), because calibration_split removes the windows it moves to
    training from evaluation. This adds a THIRD row: "no calibration,
    matched eval" -- train exactly as in "without calibration" (no extra
    windows added to training) but evaluate ONLY on the same reduced
    eval-window set that "with calibration" uses for that subject. This
    isolates the eval-set-composition effect from the true effect of
    adding calibration windows to training:
      baseline          vs with-calibration        = original comparison (confounded)
      matched-eval       vs with-calibration        = TRUE calibration-training effect
      baseline           vs matched-eval            = pure eval-set-composition effect

  PART 2 -- DEMOGRAPHIC FEATURES
    The original "with demographics" row applied baseline_relative_scale
    (per-subject z-score against the subject's OWN baseline windows) to
    the demographic columns too -- but a demographic value is identical
    across all of a subject's own windows, so this always collapses each
    demographic column to std=0 for every subject (confirmed in the
    original run). This adds a fairer normalisation for the demographic
    block only: a StandardScaler fit on the TRAINING FOLD's demographic
    values (cross-subject variance, no leakage -- only training-fold
    subjects are used to fit it), applied to both train and test. The 55
    physiological columns still get the canonical leak-fixed BL-norm,
    unchanged. This gives demographics an actual chance to carry the
    cross-subject signal BL-norm was destroying.

Model         : ExtraTrees(n_estimators=1000, max_features=0.5,
                class_weight='balanced', random_state=42, n_jobs=-1)
Normalisation : baseline_relative_scale (physio) -- LEAK-FIXED 2026-07-19
                (demographics_analysis.py); StandardScaler (demo block only,
                fit on training-fold subjects) where noted
CV            : LOSO (Leave-One-Subject-Out), 37 subjects
Phase 3       : excluded (stress_phase == 3 removed) -- 686 windows
Metric        : Macro F1 (mean across 37 subjects), SD, pooled out-of-fold AUROC

Does NOT overwrite any existing result file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon

from _canonical_sensitivity_common import (
    load_cache, build_canonical_xy, et1000_factory, baseline_relative_scale
)
from demographics_analysis import load_demographics

OUT_TXT = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'component_ablation_fair_et1000_leakfixed.txt')
OUT_CSV_CAL = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'component_ablation_fair_et1000_leakfixed_calibration_per_subject.csv')
OUT_CSV_DEMO = os.path.join(os.path.dirname(__file__), '..', 'results',
                             'component_ablation_fair_et1000_leakfixed_demographics_per_subject.csv')
XLSX_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data_tese.xlsx')

WINDOW_SEC, STRIDE_SEC = 60, 30

df_all = load_cache()
X55, y, groups, subjects, feat_cols, n_before, n_after = build_canonical_xy(
    df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC, exclude_phase3=True)

lines = []
def w(s=''):
    print(s); lines.append(s)

w('='*100)
w('COMPONENT ABLATION -- FAIR RE-RUN (calibration + demographics), canonical ET-1000 configuration')
w('='*100)
w('Model         : ExtraTrees(n_estimators=1000, max_features=0.5, class_weight=balanced, random_state=42)')
w('Normalisation : baseline_relative_scale (physio) -- LEAK-FIXED 2026-07-19 (demographics_analysis.py);')
w('                StandardScaler (demographic block only, fit on training-fold subjects) where noted')
w(f'CV            : LOSO, 37 subjects. Phase 3 (stress_phase==3) excluded. Windows: {n_after} '
  f'(of {n_before} before exclusion).')
w('Metric        : Macro F1.')
w('Extends       : results/component_ablation_et1000_leakfixed.txt -- that file is NOT overwritten.')
w()

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 -- ADAPTIVE CALIBRATION, matched-eval control added
# ═══════════════════════════════════════════════════════════════════════════

def calibration_split(X_tr_n, y_tr, X_te_n, y_te, cal_frac=0.5):
    """Verbatim from 3_anxolotl/final_summary.py:84-93."""
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr_n, y_tr, X_te_n, y_te, np.arange(len(y_te))
    n_cal = max(1, min(int(round(len(base_idx) * cal_frac)), len(base_idx) - 1))
    cal_idx = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te == 1)[0]])
    return (np.vstack([X_tr_n, X_te_n[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te_n[eval_idx], y_te[eval_idx], eval_idx)


def run_loso_calibration_modes():
    """Single pass computing all three calibration-related rows per subject,
    so the eval_idx used to build 'matched eval' is guaranteed identical to
    the one 'with calibration' actually used for that subject."""
    imp = SimpleImputer(strategy='median')
    f1_baseline, f1_withcal, f1_matched = {}, {}, {}
    n_eval = {'baseline': {}, 'withcal': {}, 'matched': {}}
    yt_base, yp_base = [], []
    yt_cal, yp_cal = [], []
    yt_match, yp_match = [], []

    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X55[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X55[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        y_te_full = y[te]

        # -- baseline: train normally, evaluate on ALL test windows --
        clf_b = et1000_factory()
        clf_b.fit(X_tr_n, y[tr])
        pred_b = clf_b.predict(X_te_n)
        f1_baseline[subj] = f1_score(y_te_full, pred_b, average='macro', zero_division=0)
        n_eval['baseline'][subj] = len(y_te_full)
        proba_b = clf_b.predict_proba(X_te_n)
        pos_b = list(clf_b.classes_).index(1)
        yt_base.append(y_te_full); yp_base.append(proba_b[:, pos_b])

        # -- with calibration: train + calibration windows, evaluate on reduced eval set --
        Xtr_cal, ytr_cal, Xte_eval, yte_eval, eval_idx = calibration_split(
            X_tr_n, y[tr], X_te_n, y_te_full, cal_frac=0.5)
        clf_c = et1000_factory()
        clf_c.fit(Xtr_cal, ytr_cal)
        pred_c = clf_c.predict(Xte_eval)
        f1_withcal[subj] = f1_score(yte_eval, pred_c, average='macro', zero_division=0)
        n_eval['withcal'][subj] = len(yte_eval)
        proba_c = clf_c.predict_proba(Xte_eval)
        pos_c = list(clf_c.classes_).index(1)
        yt_cal.append(yte_eval); yp_cal.append(proba_c[:, pos_c])

        # -- matched eval: SAME baseline-trained classifier, but scored only on
        #    the identical reduced eval_idx subset "with calibration" used --
        pred_m = clf_b.predict(X_te_n[eval_idx])
        f1_matched[subj] = f1_score(y_te_full[eval_idx], pred_m, average='macro', zero_division=0)
        n_eval['matched'][subj] = len(eval_idx)
        proba_m = clf_b.predict_proba(X_te_n[eval_idx])
        yt_match.append(y_te_full[eval_idx]); yp_match.append(proba_m[:, pos_b])

    def pooled_auroc(yt, yp):
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        try:
            return roc_auc_score(yt, yp)
        except ValueError:
            return float('nan')

    return (f1_baseline, f1_withcal, f1_matched, n_eval,
             pooled_auroc(yt_base, yp_base), pooled_auroc(yt_cal, yp_cal), pooled_auroc(yt_match, yp_match))


w('-'*100)
w('PART 1 -- ADAPTIVE CALIBRATION, with matched-eval control')
w('-'*100)

f1_base, f1_cal, f1_match, n_eval, auroc_base, auroc_cal, auroc_match = run_loso_calibration_modes()

vals_base = np.array([f1_base[s] for s in subjects])
vals_cal = np.array([f1_cal[s] for s in subjects])
vals_match = np.array([f1_match[s] for s in subjects])
tot_base = sum(n_eval['baseline'].values())
tot_cal = sum(n_eval['withcal'].values())
tot_match = sum(n_eval['matched'].values())

w(f'{"Config":<32}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}  {"TotalEvalWindows":>17}')
w('-'*100)
w(f'{"baseline (full eval, ref)":<32}  {vals_base.mean():>8.4f}  {vals_base.std():>7.4f}  {auroc_base:>7.4f}  '
  f'{tot_base:>17}   <- should match canonical ET-1000 (0.9169)')
w(f'{"with calibration":<32}  {vals_cal.mean():>8.4f}  {vals_cal.std():>7.4f}  {auroc_cal:>7.4f}  {tot_cal:>17}')
w(f'{"no calibration, matched eval":<32}  {vals_match.mean():>8.4f}  {vals_match.std():>7.4f}  {auroc_match:>7.4f}  '
  f'{tot_match:>17}   <- same eval windows as row above, no cal.')
assert tot_cal == tot_match, 'matched-eval window count must equal with-calibration window count by construction'
w()
w('Three pairwise paired Wilcoxon signed-rank tests (two-sided, 37 subjects):')
stat1, p1 = wilcoxon(vals_cal, vals_base, alternative='two-sided')
w(f'  with-calibration    vs baseline (full eval)     : statistic={stat1:.2f}  p={p1:.6f}  '
  f'({"SIGNIFICANT" if p1 < 0.05 else "not significant"})  [confounded: eval sets differ]')
stat2, p2 = wilcoxon(vals_cal, vals_match, alternative='two-sided')
w(f'  with-calibration    vs matched-eval (no cal)     : statistic={stat2:.2f}  p={p2:.6f}  '
  f'({"SIGNIFICANT" if p2 < 0.05 else "not significant"})  [TRUE effect of calibration training data]')
stat3, p3 = wilcoxon(vals_match, vals_base, alternative='two-sided')
w(f'  matched-eval (no cal) vs baseline (full eval)    : statistic={stat3:.2f}  p={p3:.6f}  '
  f'({"SIGNIFICANT" if p3 < 0.05 else "not significant"})  [pure eval-set-composition effect]')
w()

pd.DataFrame({
    'subject': subjects,
    'macro_f1_baseline_full_eval': [f1_base[s] for s in subjects],
    'macro_f1_with_calibration': [f1_cal[s] for s in subjects],
    'macro_f1_matched_eval_no_cal': [f1_match[s] for s in subjects],
    'n_eval_baseline': [n_eval['baseline'][s] for s in subjects],
    'n_eval_with_cal': [n_eval['withcal'][s] for s in subjects],
    'n_eval_matched': [n_eval['matched'][s] for s in subjects],
}).to_csv(OUT_CSV_CAL, index=False)

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 -- DEMOGRAPHIC FEATURES, fair (non-destructive) normalisation added
# ═══════════════════════════════════════════════════════════════════════════

w('-'*100)
w('PART 2 -- DEMOGRAPHIC FEATURES, fair normalisation for the demographic block')
w('-'*100)

df_demo = load_demographics(XLSX_PATH)
demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
             'respiratory', 'medication', 'exercises', 'exercise_freq',
             'athlete', 'stimulant']

def expand_demo(groups_arr):
    rows = []
    for g in groups_arr:
        rows.append(df_demo.loc[g, demo_cols].values.astype(float)
                    if g in df_demo.index else np.zeros(len(demo_cols)))
    return np.array(rows)

X_demo11 = expand_demo(groups)
missing_subjects = [s for s in subjects if s not in df_demo.index]


def run_loso_demo(mode):
    """mode: 'none' (55 feat only), 'blnorm_all' (66 feat, BL-norm on everything,
    original/destructive), 'fair' (55 feat BL-norm + demo block StandardScaler
    fit on training-fold subjects only)."""
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        Xphys_tr_n = baseline_relative_scale(X55[tr], y[tr], groups[tr])
        Xphys_te_n = baseline_relative_scale(X55[te], y[te], groups[te])

        if mode == 'none':
            X_tr_n = np.nan_to_num(Xphys_tr_n)
            X_te_n = np.nan_to_num(Xphys_te_n)
        elif mode == 'blnorm_all':
            Xdemo_tr_n = baseline_relative_scale(X_demo11[tr], y[tr], groups[tr])
            Xdemo_te_n = baseline_relative_scale(X_demo11[te], y[te], groups[te])
            X_tr_n = np.nan_to_num(np.hstack([Xphys_tr_n, Xdemo_tr_n]))
            X_te_n = np.nan_to_num(np.hstack([Xphys_te_n, Xdemo_te_n]))
        elif mode == 'fair':
            sc = StandardScaler().fit(np.nan_to_num(X_demo11[tr]))
            Xdemo_tr_n = sc.transform(np.nan_to_num(X_demo11[tr]))
            Xdemo_te_n = sc.transform(np.nan_to_num(X_demo11[te]))
            X_tr_n = np.nan_to_num(np.hstack([Xphys_tr_n, Xdemo_tr_n]))
            X_te_n = np.nan_to_num(np.hstack([Xphys_te_n, Xdemo_te_n]))
        else:
            raise ValueError(mode)

        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = et1000_factory()
        clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y[te], pred, average='macro', zero_division=0)
        proba = clf.predict_proba(X_te_n)
        pos_idx = list(clf.classes_).index(1)
        y_true_all.append(y[te])
        y_prob_all.append(proba[:, pos_idx])
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    try:
        auroc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auroc = float('nan')
    return fold_f1, auroc


w(f'Subjects missing from Data_tese.xlsx (zero-filled demo row): {missing_subjects if missing_subjects else "none"}')
w()

f1_none, auroc_none = run_loso_demo('none')
f1_blall, auroc_blall = run_loso_demo('blnorm_all')
f1_fair, auroc_fair = run_loso_demo('fair')

vals_none = np.array([f1_none[s] for s in subjects])
vals_blall = np.array([f1_blall[s] for s in subjects])
vals_fair = np.array([f1_fair[s] for s in subjects])

w(f'{"Config":<38}  {"MacroF1":>8}  {"SD":>7}  {"AUROC":>7}')
w('-'*100)
w(f'{"without demographics (55, ref)":<38}  {vals_none.mean():>8.4f}  {vals_none.std():>7.4f}  {auroc_none:>7.4f}'
  '   <- should match canonical ET-1000 (0.9169)')
w(f'{"with demographics, BL-norm-all (orig)":<38}  {vals_blall.mean():>8.4f}  {vals_blall.std():>7.4f}  '
  f'{auroc_blall:>7.4f}   <- reproduces the confound (demo collapsed to 0)')
w(f'{"with demographics, fair norm (new)":<38}  {vals_fair.mean():>8.4f}  {vals_fair.std():>7.4f}  {auroc_fair:>7.4f}')
w()
w('Three pairwise paired Wilcoxon signed-rank tests (two-sided, 37 subjects):')
statA, pA = wilcoxon(vals_blall, vals_none, alternative='two-sided')
w(f'  BL-norm-all demographics vs without demographics : statistic={statA:.2f}  p={pA:.6f}  '
  f'({"SIGNIFICANT" if pA < 0.05 else "not significant"})  [reproduces confounded original result]')
statB, pB = wilcoxon(vals_fair, vals_none, alternative='two-sided')
w(f'  fair-norm demographics   vs without demographics : statistic={statB:.2f}  p={pB:.6f}  '
  f'({"SIGNIFICANT" if pB < 0.05 else "not significant"})  [TRUE effect, demo signal preserved]')
statC, pC = wilcoxon(vals_fair, vals_blall, alternative='two-sided')
w(f'  fair-norm demographics   vs BL-norm-all demographics: statistic={statC:.2f}  p={pC:.6f}  '
  f'({"SIGNIFICANT" if pC < 0.05 else "not significant"})  [does fixing the normalisation change the outcome?]')

with open(OUT_TXT, 'w') as f:
    f.write('\n'.join(lines))

pd.DataFrame({
    'subject': subjects,
    'macro_f1_without_demographics': [f1_none[s] for s in subjects],
    'macro_f1_demographics_blnorm_all': [f1_blall[s] for s in subjects],
    'macro_f1_demographics_fair_norm': [f1_fair[s] for s in subjects],
}).to_csv(OUT_CSV_DEMO, index=False)

print(f'\nSaved: {OUT_TXT}')
print(f'Saved: {OUT_CSV_CAL}')
print(f'Saved: {OUT_CSV_DEMO}')
