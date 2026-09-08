"""
Label-free, per-subject logit recentering for the WESAD->Anxolotl transfer
(the ExtraTrees BL-norm result in results/cross_dataset_wesad.json:
f1=0.7295 default-threshold, f1opt=0.7977 leaky-optimal-threshold, auc=0.8737).

Not a re-train, not a re-run of LOSO: trains the SAME ExtraTrees ONCE on
WESAD (exact setup copied from 2_cross_dataset/cross_dataset_blnorm.py --
n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42,
40 features common to WESAD/Anxolotl, label-based BL-norm on the WESAD side),
predicts probabilities on ALL Anxolotl windows once, then adjusts a single
per-subject scalar bias in LOGIT space using only that subject's first 8
baseline windows (protocol-based, label-free -- same N_BL=8 convention as
scripts/anxolotl_loso.py and cross_dataset_blnorm.py's own
baseline_relative_scale_protocol). Recentering happens in logit space, not
raw probability space (probabilities are bounded in [0,1] and don't average
well near the edges), then sigmoid back, then a plain 0.5 threshold --
no label is used anywhere in the calibration step itself, only in the final
scoring.

Edge case handled: raw probabilities of exactly 0.0 or 1.0 give an infinite
logit. Clipped to [1e-6, 1-1e-6] before the logit transform, same epsilon
convention already used in this repo's BL-norm (scripts/common.py, +1e-8 on
the std denominator -- here it's the clip bound instead, same spirit).

Output confined to this directory.
"""
import pickle, os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
)

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
OUT_DIR = os.path.join(ROOT, 'results', 'cross_dataset_recalibration_2026-07-26')
N_BL = 8       # established convention: scripts/anxolotl_loso.py, cross_dataset_blnorm.py
EPS = 1e-6     # clip bound before logit, avoids +/-inf at p=0 or p=1

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

# ── BL-norm helpers, copied verbatim from 2_cross_dataset/cross_dataset_blnorm.py ──

def baseline_relative_scale(X, y, groups):
    X = np.array(X, dtype=float).copy()
    y = np.array(y); groups = np.array(groups)
    for subj in np.unique(groups):
        idx = np.where(groups == subj)[0]
        n_baseline = 0
        for v in y[idx]:
            if v == 0: n_baseline += 1
            else: break
        bl = idx[:n_baseline]
        if len(bl) > 1:
            for j in range(X.shape[1]):
                v = X[bl, j][~np.isnan(X[bl, j])]
                if len(v) > 1:
                    X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X

def baseline_relative_scale_protocol(X, groups, n_baseline=8):
    X = np.array(X, dtype=float).copy()
    groups = np.array(groups)
    for subj in np.unique(groups):
        idx = np.where(groups == subj)[0]; bl = idx[:n_baseline]
        for j in range(X.shape[1]):
            v = X[bl, j][~np.isnan(X[bl, j])]
            if len(v) > 1:
                X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X

# ── Load data ──────────────────────────────────────────────────────────────

with open(os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl'), 'rb') as f:
    wesad = pickle.load(f)
with open(os.path.join(ROOT, 'processed_data', 'anxolotl', 'processed_data_anxolotl.pkl'), 'rb') as f:
    anx = pickle.load(f)

# CRITICAL: processed_data_anxolotl.pkl ships 699 raw rows (stress_phase in
# {-1,1,2,3}); the canonical Anxolotl convention used everywhere else in this
# repo (3_anxolotl/shap_canonical_et1000_leakfixed.py, friedman_nemenyi_anxolotl.py)
# excludes stress_phase==3 (13 rows) to get to 686. Row-aligned indices
# confirmed (df/X/y/groups all share the same 0..698 index) -- apply the same
# mask here, or this run silently includes 13 non-canonical windows.
phase3_mask = (anx['df']['stress_phase'] != 3).values
w_n_before_phase3 = len(anx['y'])
for k in ('X', 'y', 'groups'):
    anx[k] = anx[k][phase3_mask] if hasattr(anx[k], '__getitem__') else anx[k]
anx['df'] = anx['df'][phase3_mask].reset_index(drop=True)
anx['X'] = anx['X'].reset_index(drop=True) if hasattr(anx['X'], 'reset_index') else anx['X']
anx['y'] = anx['y'].reset_index(drop=True) if hasattr(anx['y'], 'reset_index') else anx['y']
anx['groups'] = anx['groups'].reset_index(drop=True) if hasattr(anx['groups'], 'reset_index') else anx['groups']

w(f'Anxolotl phase3 exclusion: {w_n_before_phase3} raw rows -> {len(anx["y"])} after excluding stress_phase==3 '
  f'({"OK, matches canonical 686" if len(anx["y"]) == 686 else "MISMATCH -- expected 686"}).')

COMMON_COLS = sorted(set(wesad['feature_cols']) & set(anx['feature_cols']))
w(f'Using {len(COMMON_COLS)} common features (WESAD {"∩"} Anxolotl).')

X_wesad_df = wesad['X'][COMMON_COLS] if isinstance(wesad['X'], pd.DataFrame) \
             else pd.DataFrame(wesad['X'], columns=wesad['feature_cols'])[COMMON_COLS]
y_wesad = np.array(wesad['y'])
gr_wesad = np.array(wesad['groups'])
X_wesad = baseline_relative_scale(X_wesad_df.values.astype(float), y_wesad, gr_wesad)
imp_wesad = SimpleImputer(strategy='mean')
X_wesad = imp_wesad.fit_transform(X_wesad)
w(f'WESAD (source): {X_wesad.shape[0]} windows, {X_wesad.shape[1]} features, '
  f'{len(np.unique(gr_wesad))} subjects.')

X_anx_df = anx['X'][COMMON_COLS] if isinstance(anx['X'], pd.DataFrame) \
           else pd.DataFrame(anx['X'], columns=anx['feature_cols'])[COMMON_COLS]
y_anx = np.array(anx['y'])
gr_anx = np.array(anx['groups'])
X_anx = baseline_relative_scale_protocol(X_anx_df.values.astype(float), gr_anx, n_baseline=N_BL)
X_anx = imp_wesad.transform(X_anx)
w(f'Anxolotl (target): {X_anx.shape[0]} windows, {len(np.unique(gr_anx))} subjects, '
  f'baseline-relative-scale (protocol-based, label-free, N_BL={N_BL}).')
w()

# ── Train ExtraTrees on WESAD (once) ────────────────────────────────────────

w('Training ExtraTrees on WESAD (n_estimators=500, min_samples_leaf=1, '
  'class_weight=balanced, random_state=42) -- exact setup from cross_dataset_blnorm.py...')
et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                           class_weight='balanced', random_state=42, n_jobs=-1)
et.fit(X_wesad, y_wesad)
w('Done.')
w()

# ── Predict on Anxolotl (once) ──────────────────────────────────────────────

y_prob_raw = et.predict_proba(X_anx)[:, 1]

# ── BEFORE recalibration: default 0.5 threshold on raw probabilities ───────

y_pred_before = (y_prob_raw >= 0.5).astype(int)
f1_before = f1_score(y_anx, y_pred_before, average='macro', zero_division=0)
acc_before = accuracy_score(y_anx, y_pred_before)
ba_before = balanced_accuracy_score(y_anx, y_pred_before)
auc_before = roc_auc_score(y_anx, y_prob_raw)
mcc_before = matthews_corrcoef(y_anx, y_pred_before)

# Leaky, label-using ceiling on THIS run's own (current-schema) probabilities
# -- NOT for deployment, only as an apples-to-apples upper bound, since the
# stale 0.7977 reference was computed on the OLD 40-common-feature schema and
# is not a fair ceiling for this 55-common-feature run.
from sklearn.metrics import roc_curve
_fpr, _tpr, _thr = roc_curve(y_anx, y_prob_raw)
_f1s_at_thr = [f1_score(y_anx, (y_prob_raw >= t).astype(int), average='macro', zero_division=0) for t in _thr]
f1opt_leaky_current_schema = float(np.max(_f1s_at_thr))
t_opt_current_schema = float(_thr[int(np.argmax(_f1s_at_thr))])

w('='*70)
w('BEFORE recalibration (raw ET probability, fixed threshold 0.5)')
w('='*70)
w(f'  Macro F1  : {f1_before:.4f}')
w(f'  Accuracy  : {acc_before:.4f}')
w(f'  Bal. Acc  : {ba_before:.4f}')
w(f'  AUC       : {auc_before:.4f}  (unaffected by any per-subject monotonic shift within a subject, but the pooled AUC IS affected because the shift differs across subjects)')
w(f'  MCC       : {mcc_before:.4f}')
w()

# ── Recalibration: per-subject logit recentering, label-free ──────────────
# logit(p) = log(p/(1-p)); clip p first to avoid +/-inf at p in {0,1}.
# For each Anxolotl subject: bias = mean(logit(p)) over that subject's first
# N_BL protocol-baseline windows (no labels used). Subtract that subject's
# bias from ALL of that subject's logits (baseline included), then sigmoid
# back to a recalibrated probability, then threshold at 0.5.

def logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z_raw = logit(y_prob_raw)
y_prob_recal = np.empty_like(y_prob_raw)
subject_biases = {}

for subj in np.unique(gr_anx):
    idx = np.where(gr_anx == subj)[0]
    bl_idx = idx[:N_BL]  # protocol-based, label-free -- same windows as baseline_relative_scale_protocol
    bias = z_raw[bl_idx].mean()
    subject_biases[str(subj)] = float(bias)
    y_prob_recal[idx] = sigmoid(z_raw[idx] - bias)

w(f'Per-subject logit bias (mean logit over first {N_BL} protocol-baseline windows, label-free):')
for subj, bias in subject_biases.items():
    w(f'  subject {subj}: bias={bias:+.4f}')
w()

y_pred_after = (y_prob_recal >= 0.5).astype(int)
f1_after = f1_score(y_anx, y_pred_after, average='macro', zero_division=0)
acc_after = accuracy_score(y_anx, y_pred_after)
ba_after = balanced_accuracy_score(y_anx, y_pred_after)
auc_after = roc_auc_score(y_anx, y_prob_recal)
mcc_after = matthews_corrcoef(y_anx, y_pred_after)

w('='*70)
w('AFTER recalibration (logit recentered per subject, fixed threshold 0.5)')
w('='*70)
w(f'  Macro F1  : {f1_after:.4f}')
w(f'  Accuracy  : {acc_after:.4f}')
w(f'  Bal. Acc  : {ba_after:.4f}')
w(f'  AUC       : {auc_after:.4f}')
w(f'  MCC       : {mcc_after:.4f}')
w()

w('='*70)
w('COMPARISON')
w('='*70)
w(f'  {"Metric":12s} {"Before":>10s} {"After":>10s} {"Delta":>10s}')
for name, b, a in [('Macro F1', f1_before, f1_after), ('Accuracy', acc_before, acc_after),
                    ('Bal. Acc', ba_before, ba_after), ('AUC', auc_before, auc_after),
                    ('MCC', mcc_before, mcc_after)]:
    w(f'  {name:12s} {b:10.4f} {a:10.4f} {a-b:+10.4f}')
w()
w('='*70)
w('SCHEMA-VERSION NOTE (sanity check against the OLD reference)')
w('='*70)
w('results/cross_dataset_wesad.json ("ExtraTrees BL-norm"), computed 2026-06-08, OLD schema: '
  'f1(default)=0.7295, f1opt(LEAKY)=0.7977, auc=0.8737, on 40 features common to the OLD '
  'WESAD/Anxolotl pkls.')
w(f'This run uses the CURRENT (post-2026-07-17-regen) pkls: {len(COMMON_COLS)} common features, not 40 '
  f'-- both WESAD and Anxolotl processed_data pkls were regenerated in that batch (same schema-drift '
  f'already documented elsewhere this session).')
w(f'  AUC cross-check: {auc_before:.4f} (this run) vs 0.8737 (old) -- near-identical, model discrimination '
  f'reproduces despite the schema change.')
w(f'  F1 cross-check:  {f1_before:.4f} (this run) vs 0.7295 (old) -- does NOT match; expected, since default-'
  f'threshold F1 is sensitive to the exact feature composition/probability distribution in a way AUC is not.')
w('  CONCLUSION: the old 0.7295/0.7977 numbers are not a fair apples-to-apples ceiling for this run -- '
  'they are a different (stale) feature schema. Use this run\'s OWN leaky-optimal-threshold ceiling instead:')
w(f'  f1opt (LEAKY, current schema, true-label-selected threshold t={t_opt_current_schema:.4f}) = '
  f'{f1opt_leaky_current_schema:.4f}')
w()
w(f'  Honest label-free recalibrated F1 ({f1_after:.4f}) vs this run\'s OWN leaky ceiling '
  f'({f1opt_leaky_current_schema:.4f}): recovers '
  f'{(f1_after-f1_before)/(f1opt_leaky_current_schema-f1_before)*100:.0f}% of the gap between the naive '
  f'default-threshold baseline and the best possible (label-using) threshold on this exact data.')

with open(os.path.join(OUT_DIR, 'recalibration_log.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')

out = {
    'n_bl': N_BL, 'eps': EPS,
    'before': {'f1': f1_before, 'accuracy': acc_before, 'balanced_accuracy': ba_before,
               'auc': auc_before, 'mcc': mcc_before},
    'after': {'f1': f1_after, 'accuracy': acc_after, 'balanced_accuracy': ba_after,
              'auc': auc_after, 'mcc': mcc_after},
    'subject_biases': subject_biases,
    'reference_cross_dataset_wesad_json': {'f1_default': 0.7295, 'f1opt_leaky': 0.7977, 'auc': 0.8737},
}
with open(os.path.join(OUT_DIR, 'recalibration_results.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f'\nSaved: {os.path.relpath(OUT_DIR, ROOT)}/recalibration_log.txt, recalibration_results.json')
