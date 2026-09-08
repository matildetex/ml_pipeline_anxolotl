"""
Investigation of acc_wrist_min -- the single most important SHAP feature in
both the canonical Anxolotl ET-1000 config (686 windows, 55 features) and
the WESAD wrist-only model, despite very different protocols. The thesis
currently explains this with "reduced spontaneous movement under stress",
which is speculative and contested in the literature. This script checks,
on real data, whether that story holds up -- or whether the SHAP signal is
actually a marker-gesture proximity artifact.

Four tasks, entirely on local files (sessions/*.csv, results/df_all_1hz_cache.pkl):

  1. Distribution of acc_wrist_min by class (baseline vs stress): mean, std,
     paired per-subject test + pooled test, direction of the effect.
  2. SHAP beeswarm direction: does a high acc_wrist_min value push the
     canonical ET-1000 toward "stress" or "baseline"?
  3. Marker-proximity confound test: are windows with extreme acc_wrist_min
     values systematically closer to a detected wrist-raise marker (gesture
     onset) than typical windows? Tested both on the window's own arg-min
     timestamp and stratified by class (label correlates with time-since-
     marker structurally, since every phase starts right at a marker, so
     an unstratified test would be confounded by label itself).
  4. Robustness: LOSO ET-1000 macro F1 with vs without acc_wrist_min in the
     55-feature set.

Model / pipeline: byte-for-byte identical to
3_anxolotl/_canonical_sensitivity_common.py and shap_canonical_et1000_leakfixed.py
  - ExtraTrees(n_estimators=1000, max_features=0.5, class_weight='balanced',
    random_state=42, n_jobs=-1)
  - baseline_relative_scale (leak-fixed 2026-07-19), LOSO
  - window/stride = 60s/30s, boundary-fixed labelling, phase3 excluded
  - 686 windows, 37 subjects, 55 features

detect_markers() is re-run on the raw sessions (not cached anywhere) using
the exact default params preprocess_session() uses
(n_markers=8, z_percentile=85, min_duration_sec=8, marker_min_gap_sec=120,
baseline_duration_sec=300) -- same call gesture_validation.py and
marker_sensitivity_et1000_leakfixed.py rely on for n_markers per subject.
"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

import anxolotl_preprocessing as ap
from demographics_analysis import baseline_relative_scale

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH   = os.path.join(ROOT, 'results', 'df_all_1hz_cache.pkl')
SESSIONS_DIR = os.path.join(ROOT, 'sessions')
OUT_PATH     = os.path.join(ROOT, 'results', 'acc_wrist_min_investigation_2026-07-24.txt')

assert not os.path.exists(OUT_PATH), f'{OUT_PATH} already exists -- not overwriting.'

WINDOW_SEC, STRIDE_SEC = 60, 30

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))


# ─────────────────────────────────────────────────────────────────
# 0. Build the canonical 686-window dataset (identical to shap_canonical_et1000_leakfixed.py)
# ─────────────────────────────────────────────────────────────────

with open(CACHE_PATH, 'rb') as f:
    df_all = pickle.load(f)

df_feat, feat_cols = ap.build_features_tabular(df_all, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC)
n_before = len(df_feat)
df_feat = df_feat[df_feat['stress_phase'] != 3].reset_index(drop=True)
n_after = len(df_feat)

assert len(feat_cols) == 55, f'expected 55 features, got {len(feat_cols)}'
assert n_after == 686, f'expected 686 windows after phase3 exclusion, got {n_after}'

X_raw    = df_feat[feat_cols].values.astype(float)
y        = df_feat['label'].values.astype(int)
groups   = df_feat['subject'].values
subjects = np.unique(groups)
idx_min  = feat_cols.index('acc_wrist_min')

w('='*78)
w('acc_wrist_min INVESTIGATION -- Anxolotl canonical ET-1000 (686 windows, 55 feat)')
w('='*78)
w(f'Windows: {n_before} -> {n_after} ({n_before-n_after} phase3 removed), subjects: {len(subjects)}')
w()

# ─────────────────────────────────────────────────────────────────
# TASK 1 -- distribution of acc_wrist_min by class
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('1. DISTRIBUTION OF acc_wrist_min BY CLASS')
w('-'*78)
w('Feature space: X_raw BEFORE baseline_relative_scale -- this is the "real measured" '
  'feature space, not yet biased toward baseline≈0 by BL-norm. NOTE: despite '
  'normalize_per_subject() being called in preprocess_session(), assemble_dataframe() is '
  'invoked with signals_raw, not signals_norm -- so acc_wrist (and acc_wrist_min) in the '
  'cache is RAW bandpass-filtered (0.5-10Hz) squared-magnitude energy, in native sensor '
  'units, not per-subject z-scored. Values are therefore very small (~1e-5) -- printed in '
  'scientific notation below.')
w()

acc_min = X_raw[:, idx_min]
base_vals   = acc_min[y == 0]
stress_vals = acc_min[y == 1]

w(f'  Baseline (n={len(base_vals)}): mean={base_vals.mean():.4e}  std={base_vals.std(ddof=1):.4e}  '
  f'median={np.median(base_vals):.4e}')
w(f'  Stress   (n={len(stress_vals)}): mean={stress_vals.mean():.4e}  std={stress_vals.std(ddof=1):.4e}  '
  f'median={np.median(stress_vals):.4e}')
diff = stress_vals.mean() - base_vals.mean()
w(f'  Mean difference (stress - baseline): {diff:+.4e}')
direction = 'HIGHER' if diff > 0 else 'LOWER'
w(f'  => stress windows have systematically {direction} acc_wrist_min than baseline windows (pooled means).')
w()

# Pooled test (crude: windows overlap 50% within-subject, not independent -- reported
# for completeness, per-subject paired test below is the trustworthy one)
u_stat, p_mwu = stats.mannwhitneyu(stress_vals, base_vals, alternative='two-sided')
w(f'  [Pooled, non-independence caveat] Mann-Whitney U={u_stat:.1f}, p={p_mwu:.3e}')

# Per-subject paired test: mean acc_wrist_min per subject per class, then paired test
per_subj = []
for s in subjects:
    m = groups == s
    b = acc_min[m & (y == 0)]
    st = acc_min[m & (y == 1)]
    if len(b) > 0 and len(st) > 0:
        per_subj.append((s, b.mean(), st.mean()))
per_subj_df = pd.DataFrame(per_subj, columns=['subject', 'baseline_mean', 'stress_mean'])
per_subj_df['delta'] = per_subj_df['stress_mean'] - per_subj_df['baseline_mean']

n_paired = len(per_subj_df)
n_higher = (per_subj_df['delta'] > 0).sum()
n_lower  = (per_subj_df['delta'] < 0).sum()
w()
w(f'  Per-subject paired comparison (n={n_paired}/{len(subjects)} subjects with both classes present):')
w(f'    Subjects with stress_mean > baseline_mean : {n_higher}')
w(f'    Subjects with stress_mean < baseline_mean : {n_lower}')
w(f'    Mean per-subject delta (stress-baseline)  : {per_subj_df["delta"].mean():+.4e}  '
  f'(std={per_subj_df["delta"].std(ddof=1):.4e})')

t_stat, p_ttest = stats.ttest_rel(per_subj_df['stress_mean'], per_subj_df['baseline_mean'])
wstat, p_wilcoxon = stats.wilcoxon(per_subj_df['stress_mean'], per_subj_df['baseline_mean'])
w(f'    Paired t-test        : t={t_stat:.3f}, p={p_ttest:.3e}')
w(f'    Wilcoxon signed-rank : W={wstat:.1f}, p={p_wilcoxon:.3e}')
w()
sig_paired = p_wilcoxon < 0.05
w(f'  CONCLUSION 1: stress windows have {"significantly " if sig_paired else "NOT significantly "}'
  f'{direction.lower()} acc_wrist_min than baseline (per-subject paired Wilcoxon '
  f'p={p_wilcoxon:.3e}).')
w()


# ─────────────────────────────────────────────────────────────────
# TASK 2 -- SHAP direction for acc_wrist_min
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('2. SHAP DIRECTION FOR acc_wrist_min (canonical ET-1000, pooled fit)')
w('-'*78)

X_bl = baseline_relative_scale(X_raw, y, groups)
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X_bl)

clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                            class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X, y)

w('Computing TreeSHAP (exact, full 686-row matrix)...')
explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(X)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
imp_vals = np.abs(sv).mean(axis=0)
rank = int((np.argsort(imp_vals)[::-1] == idx_min).nonzero()[0][0]) + 1
w(f'  acc_wrist_min mean|SHAP| = {imp_vals[idx_min]:.6f}  (rank {rank}/{len(feat_cols)}, cross-checks '
  f'results/shap_canonical_et1000_leakfixed.txt)')

shap_min = sv[:, idx_min]
feat_val_min = X[:, idx_min]   # BL-normalised, imputed -- same space the model/SHAP see

rho, p_rho = stats.spearmanr(feat_val_min, shap_min)
w(f'  Spearman(feature value, SHAP value) = {rho:+.4f}  (p={p_rho:.3e})')

# Quintile readout, like reading the beeswarm by eye
qs = pd.qcut(feat_val_min, 5, labels=False, duplicates='drop')
w('  Mean SHAP value by feature-value quintile (Q1=lowest acc_wrist_min ... Q5=highest):')
for q in sorted(np.unique(qs)):
    w(f'    Q{q+1}: mean SHAP(class=stress) = {shap_min[qs==q].mean():+.4f}  (n={int((qs==q).sum())})')

if rho > 0:
    shap_dir = 'HIGH acc_wrist_min values push the model TOWARD stress (positive SHAP); LOW values push toward baseline.'
else:
    shap_dir = 'HIGH acc_wrist_min values push the model TOWARD baseline (negative SHAP); LOW values push toward stress.'
w()
w(f'  CONCLUSION 2: {shap_dir}')
w('  Consistency check vs "less movement under stress" narrative: that story predicts stress windows have '
  'LOWER acc_wrist_min (already tested in Task 1) AND that low acc_wrist_min should carry positive SHAP '
  'toward stress (negative correlation feature-vs-SHAP). Compare the sign above to Task 1\'s direction.')
w()


# ─────────────────────────────────────────────────────────────────
# TASK 3 -- marker-proximity confound test
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('3. MARKER-PROXIMITY CONFOUND TEST')
w('-'*78)
w('Re-running detect_markers() on raw sessions (same defaults preprocess_session() uses: '
  'n_markers=8, z_percentile=85, min_duration_sec=8, marker_min_gap_sec=120, baseline_duration_sec=300).')
w()

csv_map = {}
for fp in sorted(glob.glob(os.path.join(SESSIONS_DIR, 'session_*.csv'))):
    basename = os.path.splitext(os.path.basename(fp))[0]
    number = basename.replace('session_', '').lstrip('0') or '0'
    sid = f'p{number}'
    csv_map[sid] = fp
if 'p6' in csv_map:
    del csv_map['p6']

marker_times_by_subj = {}
marker_fail = []
for s in subjects:
    fp = csv_map.get(s)
    if fp is None:
        marker_fail.append(s)
        continue
    try:
        df_acc, df_hr, fs_acc, t0 = ap.load_session_csv(fp)
        mt = ap.detect_markers(df_acc, fs_acc, n_markers=8, z_percentile=85,
                                min_duration_sec=8, min_gap_sec=120,
                                baseline_duration_sec=300)
        marker_times_by_subj[s] = np.array(mt, dtype=float)
    except Exception as e:
        marker_fail.append(s)

w(f'  Markers recovered for {len(marker_times_by_subj)}/{len(subjects)} subjects '
  f'(failed/missing: {marker_fail if marker_fail else "none"})')
w()

# Rebuild the window loop with per-window metadata (t_start, t_end, t_argmin),
# mirroring ap.build_features_tabular's control flow exactly so rows align
# positionally with the pre-phase3-filter df_feat (same groupby, same order,
# same skip conditions), then apply the identical phase3 filter.
meta_rows = []
for subject, grp in df_all.groupby('subject'):
    grp = grp.sort_values('t_sec').reset_index(drop=True)
    n = len(grp)
    for i in range(WINDOW_SEC, n, STRIDE_SEC):
        label_first = int(grp['label'].iloc[i - WINDOW_SEC])
        label_last  = int(grp['label'].iloc[i - 1])
        if label_first != label_last:
            continue
        label = label_first
        if label not in [0, 1]:
            continue
        stress_phase = int(grp['stress_phase'].iloc[i - 1]) if 'stress_phase' in grp.columns else -1
        t_window = grp['t_sec'].iloc[i - WINDOW_SEC:i].values
        w_acc = grp['acc_wrist'].iloc[i - WINDOW_SEC:i].values.astype(float)
        t_argmin = float(t_window[np.argmin(w_acc)])
        meta_rows.append({
            'subject': subject, 'label': label, 'stress_phase': stress_phase,
            't_start': float(t_window[0]), 't_end': float(t_window[-1]),
            't_argmin': t_argmin, 'w_min_check': float(np.min(w_acc)),
        })

df_meta = pd.DataFrame(meta_rows)
assert len(df_meta) == n_before, f'meta loop produced {len(df_meta)} rows, expected {n_before}'
df_meta = df_meta[df_meta['stress_phase'] != 3].reset_index(drop=True)
assert len(df_meta) == n_after == 686, f'meta post-filter {len(df_meta)} rows, expected 686'

# Alignment sanity check: df_meta and df_feat/X_raw must be row-for-row the same windows
assert np.allclose(df_meta['w_min_check'].values, acc_min, atol=1e-8), \
    'df_meta / df_feat row misalignment -- acc_wrist_min mismatch'
assert (df_meta['subject'].values == groups).all() and (df_meta['label'].values == y).all(), \
    'df_meta / df_feat row misalignment -- subject/label mismatch'
w('  Alignment check OK: meta rows match df_feat rows 1:1 (acc_wrist_min, subject, label).')
w()

def dist_to_nearest_marker(t, marker_arr):
    if marker_arr is None or len(marker_arr) == 0:
        return np.nan
    return float(np.min(np.abs(marker_arr - t)))

def dist_window_to_marker(t_start, t_end, marker_arr):
    """0 if a marker falls inside [t_start, t_end], else distance to the nearest edge."""
    if marker_arr is None or len(marker_arr) == 0:
        return np.nan
    inside = (marker_arr >= t_start) & (marker_arr <= t_end)
    if inside.any():
        return 0.0
    d_before = t_start - marker_arr[marker_arr < t_start]
    d_after  = marker_arr[marker_arr > t_end] - t_end
    cands = np.concatenate([d_before, d_after])
    return float(cands.min()) if len(cands) else np.nan

df_meta['dist_argmin_marker']  = [
    dist_to_nearest_marker(r.t_argmin, marker_times_by_subj.get(r.subject))
    for r in df_meta.itertuples()
]
df_meta['dist_window_marker'] = [
    dist_window_to_marker(r.t_start, r.t_end, marker_times_by_subj.get(r.subject))
    for r in df_meta.itertuples()
]

valid = df_meta['dist_argmin_marker'].notna()
w(f'  Windows with a marker time available: {valid.sum()}/{len(df_meta)}')
w()

dist_argmin = df_meta.loc[valid, 'dist_argmin_marker'].values
acc_min_v   = acc_min[valid.values]
y_v         = y[valid.values]

rho_d, p_d = stats.spearmanr(acc_min_v, dist_argmin)
w(f'  Spearman(acc_wrist_min, distance from window\'s own arg-min to nearest marker) = '
  f'{rho_d:+.4f}  (p={p_d:.3e})')
w('  Positive rho would mean "closer to marker => lower min" (supports gesture-aftermath '
  'contamination). Negative rho means the opposite: closer to marker => HIGHER min, i.e. '
  'the gesture itself (elevated motion) keeps nearby windows\' minimums elevated, and the '
  'lowest minimums occur far from any marker -- the opposite of the contamination hypothesis.')
w()

# Extreme (bottom/top 10% pooled) vs typical (middle 80%), stratified by class to avoid
# the structural confound that label itself correlates with time-since-marker
# (every phase starts right at a marker).
for cls, cls_name in [(0, 'BASELINE'), (1, 'STRESS')]:
    m = y_v == cls
    vals = acc_min_v[m]
    dists = dist_argmin[m]
    if len(vals) < 10:
        continue
    lo_thr, hi_thr = np.percentile(vals, [10, 90])
    extreme_mask = (vals <= lo_thr) | (vals >= hi_thr)
    typical_mask = ~extreme_mask
    d_extreme = dists[extreme_mask]
    d_typical = dists[typical_mask]
    try:
        u2, p2 = stats.mannwhitneyu(d_extreme, d_typical, alternative='two-sided')
    except ValueError:
        u2, p2 = np.nan, np.nan
    w(f'  [{cls_name}] extreme acc_wrist_min windows (bottom/top 10%, n={extreme_mask.sum()}): '
      f'median dist-to-marker={np.median(d_extreme):.1f}s')
    w(f'  [{cls_name}] typical windows (middle 80%, n={typical_mask.sum()}): '
      f'median dist-to-marker={np.median(d_typical):.1f}s')
    w(f'  [{cls_name}] Mann-Whitney U={u2:.1f}, p={p2:.3e}')
    w()

# Same Spearman test, stratified, to isolate the effect from the label confound directly
for cls, cls_name in [(0, 'BASELINE'), (1, 'STRESS')]:
    m = y_v == cls
    if m.sum() < 10:
        continue
    rho_c, p_c = stats.spearmanr(acc_min_v[m], dist_argmin[m])
    w(f'  [{cls_name} only] Spearman(acc_wrist_min, dist-to-marker) = {rho_c:+.4f}  (p={p_c:.3e}, n={m.sum()})')
w()

confound_supported = (p_d < 0.05) and (rho_d > 0)
confound_rejected_opposite = (p_d < 0.05) and (rho_d < 0)
if confound_supported:
    verdict = 'SUPPORTED'
elif confound_rejected_opposite:
    verdict = 'REJECTED (significant, but in the opposite direction: near-marker windows have HIGHER minimums, not lower)'
else:
    verdict = 'NOT SUPPORTED (no significant link)'
w(f'  CONCLUSION 3: marker-proximity confound is {verdict} '
  f'(pooled rho={rho_d:+.3f}, p={p_d:.3e}; see class-stratified numbers above).')
w()


# ─────────────────────────────────────────────────────────────────
# TASK 4 -- robustness: LOSO F1 with vs without acc_wrist_min
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('4. ROBUSTNESS -- LOSO ET-1000 macro F1 with vs without acc_wrist_min')
w('-'*78)

def et1000_factory():
    return ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                 class_weight='balanced', random_state=42, n_jobs=-1)

def run_loso(X_in, y, groups, subjects, norm_fn):
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
        clf = et1000_factory()
        clf.fit(X_tr_n, y[tr])
        pred = clf.predict(X_te_n)
        fold_f1[subj] = f1_score(y[te], pred, average='macro', zero_division=0)
        proba = clf.predict_proba(X_te_n)
        pos_idx = list(clf.classes_).index(1)
        y_true_all.append(y[te]); y_prob_all.append(proba[:, pos_idx])
    y_true_all = np.concatenate(y_true_all); y_prob_all = np.concatenate(y_prob_all)
    try:
        auroc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        auroc = float('nan')
    return fold_f1, auroc

w('Running LOSO with all 55 features (canonical)...')
fold_f1_full, auroc_full = run_loso(X_raw, y, groups, subjects, baseline_relative_scale)
f1_full = np.mean(list(fold_f1_full.values()))
w(f'  Full (55 feat)         : Macro F1 = {f1_full:.4f}  AUROC = {auroc_full:.4f}')

keep_cols = [c for c in feat_cols if c != 'acc_wrist_min']
X_drop = df_feat[keep_cols].values.astype(float)
w('Running LOSO with acc_wrist_min removed (54 feat)...')
fold_f1_drop, auroc_drop = run_loso(X_drop, y, groups, subjects, baseline_relative_scale)
f1_drop = np.mean(list(fold_f1_drop.values()))
w(f'  Without acc_wrist_min (54 feat): Macro F1 = {f1_drop:.4f}  AUROC = {auroc_drop:.4f}')

delta = f1_full - f1_drop
w(f'  Delta (full - dropped)  : {delta:+.4f}')
w()

# Per-subject paired test on the F1 drop, for significance of the ablation
common_subjs = [s for s in subjects if s in fold_f1_full and s in fold_f1_drop]
f1_full_arr = np.array([fold_f1_full[s] for s in common_subjs])
f1_drop_arr = np.array([fold_f1_drop[s] for s in common_subjs])
w_stat, p_ablation = stats.wilcoxon(f1_full_arr, f1_drop_arr)
w(f'  Per-subject paired Wilcoxon on fold F1 (full vs dropped): W={w_stat:.1f}, p={p_ablation:.3e}')
w()
w(f'  CONCLUSION 4: removing acc_wrist_min changes Macro F1 by {delta:+.4f} '
  f'({"statistically significant" if p_ablation < 0.05 else "not statistically significant"} '
  f'at p={p_ablation:.3e}). {"The model is NOT decisively dependent on this single feature." if abs(delta) < 0.02 else "The model relies materially on this feature."}')
w()

w('='*78)
w('END OF INVESTIGATION')
w('='*78)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved: {OUT_PATH}')
