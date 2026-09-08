"""
ARCHIVED 2026-09-07: pre-leak-fix version (712 windows); superseded by the
686-window leak-fixed pipeline -- see
2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py.

This script's own results were never cited in results_master.csv as a
thesis number -- only its baseline_relative_scale() function was extracted
and copied (not imported) into 1_wesad/wrist_only/run_wesad_wrist55_blnorm.py
and run_ensemble_et_rf_hgb_blnorm.py (see their own headers). Its target-side
normalisation here (baseline_relative_scale_protocol, n_baseline=8) predates
the 2026-07-19 BL-norm leak fix; its final printed comparison
("LOSO ceiling (Anxolotl): AUC=0.725 F1=0.678") is a stale reference value
from before that fix, not a live/current number.

Cross-dataset experiment: WESAD (train) → Anxolotl (test)
Using baseline-relative normalization on both sides.
"""
import pickle, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score,
    matthews_corrcoef, classification_report, roc_curve
)

_RESULTS = []


def baseline_relative_scale(X, y, groups):
    """Uses the leading run of label=0 windows per subject as baseline
    reference (for WESAD training). Fixed 2026-07-19: previously used every
    y==0 row anywhere in the timeline, which is a self-referential leak when
    this function is ever applied to a fold using its own true labels — see
    3_anxolotl/demographics_analysis.py for the full writeup. Only used here
    for WESAD's training side, but fixed for consistency / to stop this
    pattern being copy-pasted into a leaky context again."""
    X = np.array(X, dtype=float).copy()
    y = np.array(y)
    groups = np.array(groups)
    for subj in np.unique(groups):
        subj_idx = np.where(groups == subj)[0]
        n_baseline = 0
        for v in y[subj_idx]:
            if v == 0:
                n_baseline += 1
            else:
                break
        baseline_idx = subj_idx[:n_baseline]
        if len(baseline_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[baseline_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X


def baseline_relative_scale_protocol(X, groups, n_baseline=8):
    """Uses first n_baseline windows per subject as reference — no labels needed.
    Valid for Anxolotl: protocol always starts with 5-min baseline (8 windows)."""
    X = np.array(X, dtype=float).copy()
    groups = np.array(groups)
    for subj in np.unique(groups):
        subj_idx = np.where(groups == subj)[0]
        bl_idx = subj_idx[:n_baseline]
        if len(bl_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[bl_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X


def find_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = np.argmax(tpr - fpr)
    return float(thresholds[best])


def print_metrics(y_true, y_pred, y_prob, label, threshold_used):
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    ba  = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    t_opt = float(thr[np.argmax(tpr - fpr)])
    f1opt = f1_score(y_true, (y_prob >= t_opt).astype(int), average='macro', zero_division=0)
    print(f"\n{'='*55}")
    print(f"  {label}  (threshold={threshold_used:.3f})")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=['baseline', 'stress'],
                                zero_division=0))
    print(f"Macro F1 : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Bal. Acc : {ba:.3f}")
    print(f"MCC      : {mcc:.3f}")
    _RESULTS.append({'label': label, 'f1': f1, 'f1opt': f1opt, 'auc': auc,
                     'ba': ba, 'mcc': mcc, 'dataset': 'WESAD→Anxolotl'})
    return f1, auc, mcc


# ── Load data ────────────────────────────────────────────────────────────────
# WESAD: the thesis's cross-dataset numbers (Section 4.4, 40-feature intersection) were
# computed on the OLD 137-feat/8-stat schema -- NOT the current 130-feat/11-stat
# processed_data/wesad/processed_data.pkl (regenerated 2026-07-17). See DATA.md.
with open('../processed_data/wesad_archive/processed_data_137feat_snapshot.pkl', 'rb') as f:
    wesad = pickle.load(f)
with open('../processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
    anx = pickle.load(f)

# 40 features common to both datasets
COMMON_COLS = sorted(set(wesad['feature_cols']) & set(anx['feature_cols']))
if len(COMMON_COLS) != 40:
    raise RuntimeError(
        f'Expected 40 WESAD∩Anxolotl common features (old 137-feat/8-stat WESAD '
        f'schema, per thesis Section 4.4), got {len(COMMON_COLS)}. This usually means '
        f'WESAD was loaded from the current 130-feat/11-stat schema instead of '
        f'processed_data/wesad_archive/processed_data_137feat_snapshot.pkl -- see DATA.md.'
    )
print(f"Using {len(COMMON_COLS)} common features")

# ── Prepare WESAD ────────────────────────────────────────────────────────────
X_wesad_df = wesad['X'][COMMON_COLS] if isinstance(wesad['X'], pd.DataFrame) \
             else pd.DataFrame(wesad['X'], columns=wesad['feature_cols'])[COMMON_COLS]
y_wesad  = np.array(wesad['y'])
gr_wesad = np.array(wesad['groups'])

X_wesad = X_wesad_df.values.astype(float)
X_wesad = baseline_relative_scale(X_wesad, y_wesad, gr_wesad)

imp_wesad = SimpleImputer(strategy='mean')
X_wesad   = imp_wesad.fit_transform(X_wesad)

# ── Prepare Anxolotl ─────────────────────────────────────────────────────────
X_anx_df = anx['X'][COMMON_COLS] if isinstance(anx['X'], pd.DataFrame) \
           else pd.DataFrame(anx['X'], columns=anx['feature_cols'])[COMMON_COLS]
y_anx  = np.array(anx['y'])
gr_anx = np.array(anx['groups'])

X_anx = X_anx_df.values.astype(float)
# Protocol-based: first 8 windows per subject = baseline (no labels used)
X_anx = baseline_relative_scale_protocol(X_anx, gr_anx, n_baseline=8)
X_anx = imp_wesad.transform(X_anx)   # use WESAD imputer (same feature space)

# ── Train ExtraTrees on full WESAD ───────────────────────────────────────────
print("\nTraining ExtraTrees on WESAD (baseline-relative)...")
et = ExtraTreesClassifier(
    n_estimators=500,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
et.fit(X_wesad, y_wesad)
print("Done.")

# ── Evaluate on Anxolotl — Scenario A ────────────────────────────────────────
# All windows evaluated. First 8 per subject used for unsupervised calibration
# (protocol guarantees baseline before stress — no labels needed).
y_anx_eval = y_anx
X_anx_eval = X_anx
print(f"\nEvaluation set: {len(y_anx_eval)} windows "
      f"(baseline={( y_anx_eval==0).sum()}, stress={(y_anx_eval==1).sum()})")

y_prob = et.predict_proba(X_anx_eval)[:, 1]
y_pred_05  = (y_prob >= 0.5).astype(int)
t_opt      = find_threshold(y_anx_eval, y_prob)
y_pred_opt = (y_prob >= t_opt).astype(int)

print("\n" + "="*55)
print("  CROSS-DATASET: WESAD (BL-norm) → Anxolotl")
print("  (unsupervised protocol-based calibration)")
print("="*55)

print("\n--- Default threshold (0.50) ---")
print_metrics(y_anx_eval, y_pred_05, y_prob, "ExtraTrees BL-norm | default", 0.50)

print("\n--- Youden optimal threshold (upper bound) ---")
print_metrics(y_anx_eval, y_pred_opt, y_prob, "ExtraTrees BL-norm | optimal", t_opt)

# ── Baseline comparison: standard scaler (reproduz o actual cross-dataset) ───
from sklearn.preprocessing import StandardScaler

print("\n\n" + "="*55)
print("  BASELINE: WESAD (standard scaler) → Anxolotl")
print("="*55)

X_wesad_std_df = wesad['X'][COMMON_COLS] if isinstance(wesad['X'], pd.DataFrame) \
                 else pd.DataFrame(wesad['X'], columns=wesad['feature_cols'])[COMMON_COLS]
X_wesad_std = X_wesad_std_df.values.astype(float)
imp_std = SimpleImputer(strategy='mean')
X_wesad_std = imp_std.fit_transform(X_wesad_std)
sc = StandardScaler()
X_wesad_std = sc.fit_transform(X_wesad_std)

X_anx_std_df = anx['X'][COMMON_COLS] if isinstance(anx['X'], pd.DataFrame) \
               else pd.DataFrame(anx['X'], columns=anx['feature_cols'])[COMMON_COLS]
X_anx_std = X_anx_std_df.values.astype(float)
X_anx_std = imp_std.transform(X_anx_std)
X_anx_std = sc.transform(X_anx_std)

# same eval mask: exclude first 8 windows per subject
X_anx_std_eval = X_anx_std

et_std = ExtraTreesClassifier(
    n_estimators=500, min_samples_leaf=1,
    class_weight='balanced', random_state=42, n_jobs=-1
)
et_std.fit(X_wesad_std, y_wesad)

y_prob_std     = et_std.predict_proba(X_anx_std_eval)[:, 1]
y_pred_std_05  = (y_prob_std >= 0.5).astype(int)
t_opt_std      = find_threshold(y_anx_eval, y_prob_std)
y_pred_std_opt = (y_prob_std >= t_opt_std).astype(int)

print("\n--- Default threshold (0.50) ---")
print_metrics(y_anx_eval, y_pred_std_05, y_prob_std, "ExtraTrees std-scaler | default", 0.50)

print("\n--- Youden optimal threshold (upper bound) ---")
print_metrics(y_anx_eval, y_pred_std_opt, y_prob_std, "ExtraTrees std-scaler | optimal", t_opt_std)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  SUMMARY")
print("="*65)
print(f"{'Método':<35} {'AUC':>6}  {'F1@0.5':>7}  {'F1@opt':>7}  {'MCC@0.5':>8}")
print("-"*65)

for label, y_p, y_pr in [
    ("ExtraTrees BL-norm (A)", y_pred_05, y_prob),
    ("ExtraTrees std-scaler",  y_pred_std_05, y_prob_std),
]:
    auc  = roc_auc_score(y_anx_eval, y_pr)
    f1d  = f1_score(y_anx_eval, y_p, average='macro', zero_division=0)
    t    = find_threshold(y_anx_eval, y_pr)
    f1o  = f1_score(y_anx_eval, (y_pr >= t).astype(int), average='macro', zero_division=0)
    mcc  = matthews_corrcoef(y_anx_eval, y_p)
    print(f"{label:<35} {auc:>6.3f}  {f1d:>7.3f}  {f1o:>7.3f}  {mcc:>8.3f}")

print(f"\n  LOSO ceiling (Anxolotl):    AUC=0.725  F1=0.678")
print(f"  Previous best cross-dataset: AUC=0.590  F1=0.582 (DANN target norm)")

_out = os.path.join(os.path.dirname(__file__), '..', 'results', 'cross_dataset_wesad.json')
os.makedirs(os.path.dirname(_out), exist_ok=True)
with open(_out, 'w') as _f:
    json.dump(_RESULTS, _f, indent=2)
print(f'\nResultados guardados em: {_out}')
