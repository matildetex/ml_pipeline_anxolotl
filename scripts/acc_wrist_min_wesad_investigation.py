"""
Direction check: does acc_wrist_min behave the same way in WESAD wrist-only
(thesis Section 3.1.6, ExtraTrees) as it does in the canonical Anxolotl
ET-1000 config? (see results/acc_wrist_min_investigation_2026-07-24.txt,
which found stress windows have ~2x HIGHER acc_wrist_min, and SHAP agrees:
high value -> pushes toward stress -- the opposite of the thesis's current
"reduced spontaneous movement under stress" narrative.)

Mirrors scripts/xai.py's WESAD_wrist / ET pipeline exactly, byte-for-byte:
  - Feature set : wrist_cols(feat_cols) on load_wesad_tabular() (11 acc_wrist_*
    + 11 eda_wrist_* + 11 temp_wrist_* + 11 hr_* + 11 ibi_* + 11 rmssd_* +
    11 sdnn_* = wrist-only feature set used for 'WESAD_wrist' in xai.py)
  - Normalisation: use_blnorm=False -> prep_std() = StandardScaler + SimpleImputer(mean)
    (xai.py:452-455, run_dataset_tabular('WESAD_wrist', ..., use_blnorm=False) at xai.py:634-635)
  - Model      : ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                 class_weight='balanced', random_state=42, n_jobs=-1)
                 (xai.py:293), fit on ALL data pooled (no LOSO/split -- same
                 pooled-fit design as the canonical Anxolotl SHAP script)
  - SHAP       : TreeExplainer on a stratified sample of 1200 rows (600/600,
                 rng seed 42) -- xai.py:158-166, stratified_sample() xai.py:99-104

Task 1 (raw distribution) uses the SAME raw acc_wrist_min values that feed
prep_std (X_df[feat_cols], before StandardScaler) -- WESAD's own preprocessing
(1_wesad/wesad_preprocessing.py) never per-subject normalises the 1Hz signal
before windowing either (same raw-then-downstream-norm design as Anxolotl,
confirmed 2026-07-25), so this is an apples-to-apples raw comparison with the
Anxolotl investigation's Task 1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import load_wesad_tabular, wrist_cols

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'results', 'acc_wrist_min_wesad_2026-07-24.txt')

assert not os.path.exists(OUT_PATH), f'{OUT_PATH} already exists -- not overwriting.'

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

w('='*78)
w('acc_wrist_min DIRECTION CHECK -- WESAD wrist-only (Section 3.1.6, ExtraTrees)')
w('='*78)

X_df, y, groups, feat_cols = load_wesad_tabular()
w_cols = wrist_cols(feat_cols)
idx_min = w_cols.index('acc_wrist_min')

w(f'Windows: {len(X_df)}, subjects: {len(np.unique(groups))}, wrist features: {len(w_cols)}')
w(f'Label dist: baseline(0)={int((y==0).sum())}  stress(1)={int((y==1).sum())}')
w()

X_raw = X_df[w_cols].values.astype(float)
acc_min = X_raw[:, idx_min]

# ─────────────────────────────────────────────────────────────────
# TASK 1 -- distribution of acc_wrist_min by class (raw units, same
# rationale as the Anxolotl investigation's Task 1)
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('1. DISTRIBUTION OF acc_wrist_min BY CLASS (raw units, pre-StandardScaler)')
w('-'*78)

base_vals   = acc_min[y == 0]
stress_vals = acc_min[y == 1]
base_vals   = base_vals[~np.isnan(base_vals)]
stress_vals = stress_vals[~np.isnan(stress_vals)]

w(f'  Baseline (n={len(base_vals)}): mean={base_vals.mean():.4e}  std={base_vals.std(ddof=1):.4e}  '
  f'median={np.median(base_vals):.4e}')
w(f'  Stress   (n={len(stress_vals)}): mean={stress_vals.mean():.4e}  std={stress_vals.std(ddof=1):.4e}  '
  f'median={np.median(stress_vals):.4e}')
diff = stress_vals.mean() - base_vals.mean()
w(f'  Mean difference (stress - baseline): {diff:+.4e}')
direction = 'HIGHER' if diff > 0 else 'LOWER'
w(f'  => stress windows have systematically {direction} acc_wrist_min than baseline windows (pooled means).')
w()

u_stat, p_mwu = stats.mannwhitneyu(stress_vals, base_vals, alternative='two-sided')
w(f'  [Pooled, non-independence caveat] Mann-Whitney U={u_stat:.1f}, p={p_mwu:.3e}')

subjects = np.unique(groups)
per_subj = []
for s in subjects:
    m = groups == s
    b = acc_min[m & (y == 0)]; b = b[~np.isnan(b)]
    st = acc_min[m & (y == 1)]; st = st[~np.isnan(st)]
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
w(f'  CONCLUSION 1: WESAD stress windows have {"significantly " if sig_paired else "NOT significantly "}'
  f'{direction.lower()} acc_wrist_min than baseline (per-subject paired Wilcoxon p={p_wilcoxon:.3e}).')
w()


# ─────────────────────────────────────────────────────────────────
# TASK 2 -- SHAP direction (exact xai.py WESAD_wrist / ET replication)
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('2. SHAP DIRECTION FOR acc_wrist_min (WESAD_wrist / ET, pooled fit, xai.py replica)')
w('-'*78)

sc = StandardScaler()
imp = SimpleImputer(strategy='mean')
Xn = imp.fit_transform(sc.fit_transform(X_raw))

clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                            class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(Xn, y)

def stratified_sample(X, y, n=1200):
    rng = np.random.default_rng(42)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx], idx

Xs, ys, idx = stratified_sample(Xn, y, n=1200)
w(f'  TreeSHAP on stratified sample: n={len(Xs)} (baseline={int((ys==0).sum())}, stress={int((ys==1).sum())})')

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
imp_vals = np.abs(sv).mean(axis=0)
rank = int((np.argsort(imp_vals)[::-1] == idx_min).nonzero()[0][0]) + 1
w(f'  acc_wrist_min mean|SHAP| = {imp_vals[idx_min]:.6f}  (rank {rank}/{len(w_cols)})')

shap_min = sv[:, idx_min]
feat_val_min = Xs[:, idx_min]   # StandardScaler-space, same as model/SHAP see

rho, p_rho = stats.spearmanr(feat_val_min, shap_min)
w(f'  Spearman(feature value, SHAP value) = {rho:+.4f}  (p={p_rho:.3e})')

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
w()

w('-'*78)
w('2b. CRITICAL CAVEAT -- figures/shap/WESAD_wrist_ET_bar.png IS STALE')
w('-'*78)
w('figures/shap/WESAD_wrist_ET_bar.png (existing figure showing acc_wrist_min as the '
  '#1 feature for WESAD_wrist/ET, mtime 2026-07-16 20:50) predates the next-day '
  'processed_data/wesad/processed_data.pkl regen (mtime 2026-07-17 16:31) -- already '
  'flagged, unresolved, in MISSING.md:74 / ANXOLOTL_AUDIT_FULL.txt:1016. This rerun on '
  'the CURRENT pkl gives a materially different ranking: '
  f'rank {rank}/{len(w_cols)}, mean|SHAP|={imp_vals[idx_min]:.4f}, vs the stale figure\'s '
  'rank 1 / ~0.044. acc_wrist_min is NOT important in the current WESAD_wrist/ET model. '
  'If Section 3.1.6 cites that figure, the claim is built on a superseded feature cache. '
  'Separately, WESAD\'s window/stride (60s/1s near-continuous, '
  '1_wesad/wesad_preprocessing.py:27-28) is not protocol-matched to Anxolotl\'s 60s/30s '
  '(MISSING.md:120) -- a direct WESAD/Anxolotl comparison here is weak evidence either way.')
w()


# ─────────────────────────────────────────────────────────────────
# FINAL COMPARISON WITH ANXOLOTL
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('3. DIRECTION COMPARISON: WESAD vs Anxolotl')
w('-'*78)
w('Anxolotl (results/acc_wrist_min_investigation_2026-07-24.txt): stress ~2x HIGHER '
  'acc_wrist_min than baseline (paired Wilcoxon p=1.56e-05); SHAP Spearman(feature,SHAP)=+0.51 '
  '(high value -> pushes toward stress).')
w()
dist_significant = p_wilcoxon < 0.05
shap_significant = p_rho < 0.05
w(f'  WESAD distribution direction : stress {direction} than baseline, '
  f'{"significant" if dist_significant else "NOT significant"} (paired Wilcoxon p={p_wilcoxon:.3e})')
w(f'  WESAD SHAP direction          : nominally {"high -> stress" if rho > 0 else "high -> baseline"}, '
  f'{"significant" if shap_significant else "NOT significant"} (Spearman p={p_rho:.3e})')
w()
if not dist_significant and not shap_significant:
    verdict = (f'Neither WESAD test reaches significance, and the feature is not meaningfully '
               f'important in WESAD to begin with (rank {rank}/{len(w_cols)}, mean|SHAP|={imp_vals[idx_min]:.4f}). '
               f'Calling the nominal directions "same" or "opposite" as Anxolotl would overinterpret noise '
               f'around a near-null effect -- WESAD provides NO reliable signal either way. Do not cite WESAD '
               f'as corroborating or contradicting the Anxolotl result; treat the Anxolotl finding as standing '
               f'on its own, and correct Section 3.1.6\'s cross-dataset framing accordingly. See 2b above for '
               f'the additional staleness caveat on the pre-existing WESAD_wrist_ET_bar.png figure.')
elif dist_significant and shap_significant and (direction == 'HIGHER') == (rho > 0):
    verdict = 'WESAD MATCHES Anxolotl on a significant effect in the same direction (see 2b for the figure-staleness caveat regardless).'
elif dist_significant and shap_significant:
    verdict = 'WESAD shows a significant effect but in the OPPOSITE direction from Anxolotl -- the two datasets disagree; do not generalise one explanation to both (see 2b for the figure-staleness caveat regardless).'
else:
    verdict = 'MIXED: distribution and SHAP significance/direction do not agree with each other within WESAD -- inspect the numbers above before writing any single narrative (see 2b for the figure-staleness caveat regardless).'
w(f'  VERDICT: {verdict}')
w()

w('='*78)
w('END')
w('='*78)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved: {OUT_PATH}')
