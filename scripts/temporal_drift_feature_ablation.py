"""
temporal_drift_feature_ablation.py -- the corrected "Ponto 3" for the
temporal-confounding control (Sec 1.5.1 restructure, 2026-07-26).

BACKGROUND / WHY THE ORIGINAL "Ponto 3" WAS DROPPED:
The originally-proposed test ("shuffle window order within each phase, check
if stress-classification performance holds") is a mathematical no-op for this
pipeline and was NOT run: ExtraTrees/RandomForest-style ensembles treat rows
as i.i.d. samples (row order never affects fit/predict), and
baseline_relative_scale's reference is defined by SET membership (the
leading run of label==0 windows) not by the order of windows WITHIN that
set -- reordering rows changes nothing, bit-for-bit, given a fixed
random_state. Running it would have produced an identical 0.9169 not because
the model is robust to temporal order, but because temporal order was never
a possible input to begin with. Reporting that as evidence would be
misleading (a careful examiner would ask "did the model ever see order?" and
the honest answer -- no -- kills the argument).

THE CORRECTED TEST:
Instead of asking "does the model use window order" (it structurally can't),
this asks "does the model rely on the specific FEATURE DIMENSIONS that carry
the temporal-drift signal found by Teste A2 (scripts/temporal_control.py,
Macro F1=0.7328 for early-vs-late within a fixed true class)?"

Candidate drift-carrying features are defined A PRIORI, by construction, NOT
by measuring feature importance on the early-vs-late task and then removing
whatever ranks highest (that would be selecting features based on the very
result being tested, then testing on a different target -- a subtle
circularity). Of the 11 statistics computed per channel
(3_anxolotl/anxolotl_preprocessing.py:756-820), 8 are permutation-invariant
functions of the window's sample values (mean, std, min, max, p25, p75,
skew, kurtosis -- reordering the 60 samples within a window changes none of
these). The remaining 3 are NOT permutation-invariant -- their value depends
on which samples come first/last within the window:
  - slope       : linear regression against the time index
  - delta_mean  : second-half mean minus first-half mean
  - lf_hf       : Welch PSD ratio (needs the sample sequence to define
                  frequency content)
This is the a priori criterion used to define TWO ablation tiers:
  CONSERVATIVE : drop {slope, delta_mean} only, 5 channels x 2 = 10 features
                 (the two statistics that are unambiguously about
                 within-window TREND, per the user's own framing)
  AGGRESSIVE   : drop {slope, delta_mean, lf_hf}, 5 channels x 3 = 15 features
                 (the full set of order-dependent statistics)

The early-vs-late classifier (Teste A2's construction, reused from
scripts/temporal_control.py) is used ONLY to CONFIRM after the fact that
these a-priori-defined features do carry the drift signal (via feature
importance on the SAME early-vs-late task, single pooled fit, in-sample --
not to select which features to drop; the drop set was fixed above before
this check ever runs).

PRE-REGISTERED INTERPRETATION (fixed BEFORE seeing any result, per explicit
instruction not to rationalize after the fact):
  |delta F1| <= 0.01 AND Wilcoxon p >= 0.05
      -> "MAINTAINS": the canonical stress classifier does not need the
         drift-carrying dimensions -- strong evidence the 0.9169 result is
         not explained by temporal drift. (Same threshold convention as the
         acc_wrist_min ablation already in the repo: delta=-0.0010 called
         negligible there.)
  0.01 < |delta F1| <= 0.05
      -> "AMBIGUOUS": a real but moderate drop. This does NOT by itself
         prove the model depends on a temporal artifact -- slope/delta_mean/
         lf_hf are not PURELY artifactual; e.g. LF/HF ratio is a standard
         sympathovagal-balance marker and HRV trend within a window can be a
         genuine physiological response to stress, not just sensor drift or
         habituation. A drop in this range must be reported as "these
         features carry information relevant to stress classification that
         may be partly physiological and partly temporal -- this test cannot
         separate the two" -- NOT as "the model relies on a temporal
         artifact".
  |delta F1| > 0.05
      -> "LARGE DROP": the model leans heavily on these dimensions. Still
         subject to the same physiological-vs-artifact ambiguity as above
         (do not claim artifact-dependence from this alone), but worth
         flagging as a limitation requiring further discussion rather than a
         closed case either way.
These thresholds and their exact wording are fixed here, before running
STEP 3 below, and are not to be adjusted after seeing the numbers.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANX_DIR = os.path.join(REPO_ROOT, '3_anxolotl')
sys.path.insert(0, ANX_DIR)

import _canonical_sensitivity_common as csc  # noqa: E402
from demographics_analysis import baseline_relative_scale  # noqa: E402

OUT_PATH = os.path.join(REPO_ROOT, 'results', 'temporal_drift_feature_ablation_2026-07-26.txt')
SEED = 42

CONSERVATIVE_SUFFIXES = ('_slope', '_delta_mean')
AGGRESSIVE_SUFFIXES = ('_slope', '_delta_mean', '_lf_hf')


def run_loso(X_in, y, groups, subjects, norm_fn):
    """Identical convention to 3_anxolotl/acc_wrist_min_investigation.py Task 4
    and 3_anxolotl/_canonical_sensitivity_common.py::run_loso_et1000."""
    imp = SimpleImputer(strategy='median')
    fold_f1 = {}
    y_true_all, y_prob_all = [], []
    for subj in subjects:
        te = groups == subj
        tr = ~te
        X_tr_n = np.nan_to_num(norm_fn(X_in[tr], y[tr], groups[tr]))
        X_te_n = np.nan_to_num(norm_fn(X_in[te], y[te], groups[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = csc.et1000_factory()
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


def interpret(delta, p):
    if abs(delta) <= 0.01 and p >= 0.05:
        return 'MAINTAINS'
    elif abs(delta) <= 0.05:
        return 'AMBIGUOUS'
    else:
        return 'LARGE DROP'


def confirm_drift_features_via_early_late(X, y_true, groups, subjects, feat_cols):
    """
    CONFIRMATION ONLY (not selection -- the drop sets above are already
    fixed). Reuses the exact early/late construction from
    scripts/temporal_control.py::teste_a2_within_class_temporal_drift, but
    here just fits ONE pooled ET-1000 (in-sample, like the canonical SHAP
    script -- not LOSO, since this is a feature-ranking check, not a
    headline metric) and reports feature_importances_ for the a-priori
    candidate features vs the rest.
    """
    keep_mask = np.zeros(len(groups), dtype=bool)
    y_temporal = np.full(len(groups), -1, dtype=int)
    for subj in subjects:
        for true_class in (0, 1):
            idx = np.where((groups == subj) & (y_true == true_class))[0]
            n = len(idx)
            half = n // 2
            if half < 1:
                continue
            early_idx, late_idx = idx[:half], idx[-half:]
            keep_mask[early_idx] = True
            keep_mask[late_idx] = True
            y_temporal[early_idx] = 0
            y_temporal[late_idx] = 1

    X_sub = X[keep_mask]
    y_sub = y_temporal[keep_mask]
    groups_sub = groups[keep_mask]

    X_norm = np.nan_to_num(baseline_relative_scale(X_sub, y_sub, groups_sub))
    imp = SimpleImputer(strategy='median')
    X_norm = imp.fit_transform(X_norm)

    clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    clf.fit(X_norm, y_sub)
    importances = dict(zip(feat_cols, clf.feature_importances_))
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def main():
    if os.path.exists(OUT_PATH):
        raise FileExistsError(f'{OUT_PATH} already exists -- refusing to overwrite.')

    print('Loading canonical feature set (60s/30s, phase3 excluded)...')
    df_all = csc.load_cache()
    X, y, groups, subjects, feat_cols, n_before, n_after = csc.build_canonical_xy(
        df_all, window_sec=60, stride_sec=30, exclude_phase3=True)
    print(f'{n_after} windows, {len(feat_cols)} features, {len(subjects)} subjects.')

    conservative_drop = [c for c in feat_cols if c.endswith(CONSERVATIVE_SUFFIXES)]
    aggressive_drop = [c for c in feat_cols if c.endswith(AGGRESSIVE_SUFFIXES)]
    assert len(conservative_drop) == 10, conservative_drop
    assert len(aggressive_drop) == 15, aggressive_drop
    print(f'Conservative drop ({len(conservative_drop)} feat): {conservative_drop}')
    print(f'Aggressive drop ({len(aggressive_drop)} feat): {aggressive_drop}')

    conservative_keep_idx = [i for i, c in enumerate(feat_cols) if c not in conservative_drop]
    aggressive_keep_idx = [i for i, c in enumerate(feat_cols) if c not in aggressive_drop]

    print('\n=== Full (55 feat, canonical) ===')
    fold_f1_full, auroc_full = run_loso(X, y, groups, subjects, baseline_relative_scale)
    f1_full = float(np.mean(list(fold_f1_full.values())))
    print(f'  Macro F1 = {f1_full:.4f}  AUROC = {auroc_full:.4f}')

    print('\n=== Conservative drop (45 feat: -slope, -delta_mean) ===')
    X_cons = X[:, conservative_keep_idx]
    fold_f1_cons, auroc_cons = run_loso(X_cons, y, groups, subjects, baseline_relative_scale)
    f1_cons = float(np.mean(list(fold_f1_cons.values())))
    print(f'  Macro F1 = {f1_cons:.4f}  AUROC = {auroc_cons:.4f}')

    print('\n=== Aggressive drop (40 feat: -slope, -delta_mean, -lf_hf) ===')
    X_agg = X[:, aggressive_keep_idx]
    fold_f1_agg, auroc_agg = run_loso(X_agg, y, groups, subjects, baseline_relative_scale)
    f1_agg = float(np.mean(list(fold_f1_agg.values())))
    print(f'  Macro F1 = {f1_agg:.4f}  AUROC = {auroc_agg:.4f}')

    common = list(subjects)
    f1_full_arr = np.array([fold_f1_full[s] for s in common])
    f1_cons_arr = np.array([fold_f1_cons[s] for s in common])
    f1_agg_arr = np.array([fold_f1_agg[s] for s in common])

    w_cons, p_cons = stats.wilcoxon(f1_full_arr, f1_cons_arr)
    w_agg, p_agg = stats.wilcoxon(f1_full_arr, f1_agg_arr)

    delta_cons = f1_full - f1_cons
    delta_agg = f1_full - f1_agg

    verdict_cons = interpret(delta_cons, p_cons)
    verdict_agg = interpret(delta_agg, p_agg)

    print(f'\nConservative: delta={delta_cons:+.4f}  Wilcoxon p={p_cons:.3e}  -> {verdict_cons}')
    print(f'Aggressive  : delta={delta_agg:+.4f}  Wilcoxon p={p_agg:.3e}  -> {verdict_agg}')

    print('\n=== Confirmation: do the a-priori drift features rank highly on early-vs-late? ===')
    ranked = confirm_drift_features_via_early_late(X, y, groups, subjects, feat_cols)
    top15 = ranked[:15]
    drift_in_top15 = [c for c, _ in top15 if c in aggressive_drop]
    print(f'Top 15 features by importance on early-vs-late (in-sample, pooled fit):')
    for c, imp_val in top15:
        flag = '  <- a-priori drift feature' if c in aggressive_drop else ''
        print(f'  {c:<20s} {imp_val:.4f}{flag}')
    print(f'{len(drift_in_top15)}/15 a-priori drift features appear in the top 15 '
          f'(of 15 total drift features / 55): {drift_in_top15}')

    with open(OUT_PATH, 'w') as f:
        def fw(s=''):
            f.write(str(s) + '\n')

        fw('=' * 100)
        fw('TEMPORAL-DRIFT FEATURE ABLATION -- corrected "Ponto 3" for the')
        fw('temporal-confounding control (Sec 1.5.1)')
        fw('Generated: 2026-07-26')
        fw('=' * 100)
        fw()
        fw('Replaces the originally-proposed "shuffle window order" test, which is a')
        fw('mathematical no-op for this pipeline (row-order-invariant model + a')
        fw('set-membership-based, not order-based, normalisation reference -- see this')
        fw("script's module docstring for the full argument). Instead: are the specific")
        fw('feature dimensions that carry the temporal-drift signal found by Teste A2')
        fw('(scripts/temporal_control.py, Macro F1=0.7328 early-vs-late within a fixed')
        fw('true class) dispensable for the canonical stress classifier?')
        fw()
        fw('A PRIORI drop criterion (fixed before running, not selected by measured')
        fw('importance): of the 11 stats/channel, 8 are permutation-invariant (mean,')
        fw('std, min, max, p25, p75, skew, kurtosis); 3 are NOT (slope, delta_mean,')
        fw('lf_hf -- their value depends on within-window sample order).')
        fw(f'  CONSERVATIVE drop (10 feat): {conservative_drop}')
        fw(f'  AGGRESSIVE drop (15 feat)  : {aggressive_drop}')
        fw()
        fw('PRE-REGISTERED interpretation (fixed before seeing results):')
        fw('  |delta F1|<=0.01 AND p>=0.05  -> MAINTAINS (model does not need these dims)')
        fw('  0.01 < |delta F1| <=0.05       -> AMBIGUOUS (real but moderate; these features')
        fw('                                    are not purely artifactual -- e.g. lf_hf is a')
        fw('                                    standard sympathovagal marker, so a drop here')
        fw('                                    does NOT prove artifact-dependence)')
        fw('  |delta F1| > 0.05              -> LARGE DROP (model leans on these dims; still')
        fw('                                    does not by itself prove artifact-dependence,')
        fw('                                    flag as a limitation for discussion)')
        fw()
        fw('Model/CV: identical to canonical (ET-1000 mf=0.5, leak-fixed BL-norm, LOSO,')
        fw(f'37 subjects, {n_after} windows).')
        fw()
        fw('-' * 100)
        fw(f'{"Config":<32s}{"Macro F1":>12s}{"AUROC":>10s}{"Delta vs full":>16s}'
           f'{"Wilcoxon p":>13s}{"Verdict":>14s}')
        fw('-' * 100)
        fw(f'{"Full (55 feat, canonical)":<32s}{f1_full:>12.4f}{auroc_full:>10.4f}'
           f'{"--":>16s}{"--":>13s}{"--":>14s}')
        fw(f'{"Conservative (45 feat)":<32s}{f1_cons:>12.4f}{auroc_cons:>10.4f}'
           f'{delta_cons:>+16.4f}{p_cons:>13.3e}{verdict_cons:>14s}')
        fw(f'{"Aggressive (40 feat)":<32s}{f1_agg:>12.4f}{auroc_agg:>10.4f}'
           f'{delta_agg:>+16.4f}{p_agg:>13.3e}{verdict_agg:>14s}')
        fw('-' * 100)
        fw()
        fw('CONFIRMATION (not selection) -- a-priori drift features vs early-vs-late')
        fw('feature importance (in-sample pooled fit, same construction as Teste A2):')
        for c, imp_val in top15:
            flag = '  <- a-priori drift feature' if c in aggressive_drop else ''
            fw(f'  {c:<20s} {imp_val:.4f}{flag}')
        fw(f'{len(drift_in_top15)}/15 a-priori drift features appear in the top 15 of 55 '
           f'ranked by early-vs-late importance: {drift_in_top15}')
        fw()
        fw('=' * 100)
        fw('SUMMARY')
        fw('=' * 100)
        fw(f'Conservative drop (slope, delta_mean only): {verdict_cons} '
           f'(delta={delta_cons:+.4f}, p={p_cons:.3e})')
        fw(f'Aggressive drop (+ lf_hf)                  : {verdict_agg} '
           f'(delta={delta_agg:+.4f}, p={p_agg:.3e})')

    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
