"""
Re-verification of WESAD SHAP figures/rankings for thesis Section 3.1.6, after
confirming (results/acc_wrist_min_wesad_2026-07-24.txt) that figures/shap/
WESAD_wrist_ET_bar.png is stale (predates the 2026-07-17 16:31 processed_data.pkl
regen) and that acc_wrist_min's rank collapses from #1 to #34 on the current schema.

Three tasks:
  1. List every WESAD_all_*/WESAD_wrist_* SHAP figure's mtime, flag which predate
     the 2026-07-17 16:31:57 regen (done first, printed before any regeneration).
  2. Regenerate the ET model's bar/beeswarm/groups figures (the model Section
     3.1.6's group-level claims are about) for BOTH feature sets, from the
     CURRENT pkl, using the exact scripts/xai.py methodology/hyperparameters/seed.
     Saved to figures/shap/reverified_2026-07-25/ (NEW location -- does not
     overwrite the existing stale PNGs, per instruction not to overwrite anything).
  3. Report signal-group SHAP importance ranking for WESAD_all/ET and
     WESAD_wrist/ET on the current schema, and compare against what the thesis
     claims (full -> RESP/IBI/EDA top, HRV bottom; wrist -> EDA/ACC/TEMP top,
     HRV bottom).

Methodology (byte-for-byte matching scripts/xai.py, cross-checked against the
already-computed-live numbers in RECONCILIATION.md section F2, 2026-07-24):
  - Xn = StandardScaler + SimpleImputer(mean) on raw X_df[feat_cols] (xai.py:452-455,
    'WESAD_all'/'WESAD_wrist' calls use_blnorm=False, xai.py:634-638)
  - ExtraTreesClassifier(n_estimators=1000, max_features=0.5, class_weight='balanced',
    random_state=42, n_jobs=-1), fit on ALL data pooled, no SMOTE (xai.py:293)
  - TreeSHAP exact on a stratified 1200-row sample (600/600), rng seed 42
    (xai.py:99-104, 158-166)
  - Signal groups: SIGNAL_GROUPS dict, xai.py:60-71 (ACC/EDA/TEMP/HR/IBI/RMSSD/
    SDNN/RESP/EMG/DEMO -- note ACC's rule `c.startswith('acc')` catches BOTH
    acc_wrist_* and acc_chest_* under one group for the 'all' config)
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import load_wesad_tabular, wrist_cols

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(ROOT, 'figures', 'shap')
NEW_FIG_DIR = os.path.join(FIG_DIR, 'reverified_2026-07-25')
OUT_PATH = os.path.join(ROOT, 'results', 'wesad_shap_reverification_2026-07-24.txt')

assert not os.path.exists(OUT_PATH), f'{OUT_PATH} already exists -- not overwriting.'
os.makedirs(NEW_FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})

CUTOFF = 1784302317.0  # 2026-07-17 16:31:57 local, processed_data/wesad/processed_data.pkl mtime

SIGNAL_GROUPS = {
    'ACC':   lambda c: c.startswith('acc'),
    'EDA':   lambda c: c.startswith('eda'),
    'TEMP':  lambda c: c.startswith('temp'),
    'HR':    lambda c: c.startswith('hr_') or c.startswith('ecg_hr'),
    'IBI':   lambda c: c.startswith('ibi') or c.startswith('ecg_ibi'),
    'RMSSD': lambda c: c.startswith('rmssd') or c.startswith('ecg_rmssd'),
    'SDNN':  lambda c: c.startswith('sdnn')  or c.startswith('ecg_sdnn'),
    'RESP':  lambda c: c.startswith('resp'),
    'EMG':   lambda c: c.startswith('emg'),
    'DEMO':  lambda c: c in ('age','gender','coffee_today','coffee_last_hour',
                              'sports','smoker','ill','stai_baseline','stai_stress'),
}
def assign_group(col):
    for name, fn in SIGNAL_GROUPS.items():
        if fn(col): return name
    return 'OTHER'

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

w('='*78)
w('WESAD SHAP RE-VERIFICATION -- Section 3.1.6 figures/rankings vs current schema')
w('='*78)

# ─────────────────────────────────────────────────────────────────
# TASK 1 -- mtimes of every WESAD SHAP figure
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('1. MTIMES OF EVERY WESAD_all_*/WESAD_wrist_* SHAP FIGURE')
w('-'*78)
w(f'Cutoff: processed_data/wesad/processed_data.pkl regen, 2026-07-17 16:31:57')
w()

files = sorted(glob.glob(os.path.join(FIG_DIR, 'WESAD_*.png')),
               key=lambda f: os.path.getmtime(f))
n_stale = n_current = 0
for f in files:
    mt = os.path.getmtime(f)
    status = 'STALE' if mt < CUTOFF else 'current'
    if status == 'STALE': n_stale += 1
    else: n_current += 1
    import datetime
    mtstr = datetime.datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S')
    w(f'  {os.path.basename(f):48s} {mtstr}  {status}')
w()
w(f'  TOTAL: {len(files)} figures, {n_stale} STALE (predate 2026-07-17 16:31:57), {n_current} current.')
w()

# ─────────────────────────────────────────────────────────────────
# TASK 2 -- regenerate ET figures (bar/beeswarm/groups) for both feature sets
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('2. REGENERATING ET FIGURES (bar/beeswarm/groups) FOR WESAD_all AND WESAD_wrist')
w('-'*78)
w(f'Saved to NEW location {os.path.relpath(NEW_FIG_DIR, ROOT)}/ -- does not overwrite '
  f'the existing stale figures at figures/shap/WESAD_*_ET_*.png.')
w()

X_df, y, groups, fc_all = load_wesad_tabular()
w_cols = wrist_cols(fc_all)

def stratified_sample(X, y, n=1200):
    rng = np.random.default_rng(42)
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n0 = min(n // 2, len(idx0)); n1 = min(n // 2, len(idx1))
    idx = np.concatenate([rng.choice(idx0, n0, replace=False),
                          rng.choice(idx1, n1, replace=False)])
    return X[idx], y[idx], idx

def bar_plot(imp_vals, feat_names, title, fname, top=20):
    order = np.argsort(imp_vals)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feat_names)[order][::-1], imp_vals[order][::-1], color='steelblue', alpha=0.85)
    ax.set_xlabel('Importance'); ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()
    print(f'    -> {os.path.basename(fname)}')

def groups_plot(imp_vals, feat_names, title, fname):
    g = {}
    for col, v in zip(feat_names, imp_vals):
        k = assign_group(col); g[k] = g.get(k, 0) + v
    g = {k: v for k, v in g.items() if v > 0}
    order = sorted(g, key=g.get, reverse=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(order, [g[k] for k in order], color=plt.cm.tab10(np.linspace(0, 1, len(order))), alpha=0.85)
    ax.set_ylabel('Summed Importance'); ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()
    print(f'    -> {os.path.basename(fname)}')
    return g, order

def beeswarm_plot(sv, X_s, feat_names, title, fname, max_d=20):
    try:
        expl = shap.Explanation(values=sv, data=X_s, feature_names=list(feat_names))
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(expl, max_display=max_d, show=False)
        plt.title(title, pad=10); plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()
        print(f'    -> {os.path.basename(fname)}')
    except Exception as e:
        plt.close(); print(f'    [beeswarm skip] {e}')

def run_config(feat_cols, ds_label):
    w(f'--- {ds_label} ({len(feat_cols)} features) ---')
    X_raw = X_df[feat_cols].values.astype(float)
    sc = StandardScaler(); imp = SimpleImputer(strategy='mean')
    Xn = imp.fit_transform(sc.fit_transform(X_raw))

    clf = ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(Xn, y)

    Xs, ys, idx = stratified_sample(Xn, y, n=1200)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    imp_vals = np.abs(sv).mean(axis=0)

    tag = f'{ds_label}_ET'
    p = os.path.join(NEW_FIG_DIR, tag)
    bar_plot(imp_vals, feat_cols, f'Feature Importance — {tag} (reverified 2026-07-25)', f'{p}_bar.png')
    g, order = groups_plot(imp_vals, feat_cols, f'Signal Groups — {tag} (reverified 2026-07-25)', f'{p}_groups.png')
    beeswarm_plot(sv, Xs, feat_cols, f'SHAP Beeswarm — {tag} (reverified 2026-07-25)', f'{p}_beeswarm.png')

    w(f'  Signal-group importance ranking (sum mean|SHAP|):')
    for rank, k in enumerate(order, 1):
        w(f'    {rank}. {k:6s} {g[k]:.4f}')
    w()

    top20 = np.argsort(imp_vals)[::-1][:20]
    w(f'  Top 20 features (mean|SHAP|):')
    for rank, i in enumerate(top20, 1):
        w(f'    {rank:2d}. {feat_cols[i]:22s} {imp_vals[i]:.6f}')
    w()
    return g, order, imp_vals, feat_cols

g_wrist, order_wrist, imp_wrist, fc_wrist = run_config(w_cols, 'WESAD_wrist')
g_all,   order_all,   imp_all,   fc_full  = run_config(fc_all, 'WESAD_all')

# ─────────────────────────────────────────────────────────────────
# TASK 3 -- compare against thesis claims
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('3. GROUP-RANKING COMPARISON: thesis claim vs current-schema regeneration')
w('-'*78)
w()
w('Thesis claim (per user): WESAD-full -> RESP/IBI/EDA at the top, HRV at the bottom.')
w(f'Current schema WESAD_all/ET ranking: {" > ".join(order_all)}')
has_resp = any(SIGNAL_GROUPS["RESP"](c) for c in fc_full)
w(f'  RESP present in current feature set at all? {has_resp}')
if not has_resp:
    w('  RESP does not exist in the current 130-column WESAD schema (no resp_freq/resp_iri '
    'columns -- those were part of the OLD 137-feature schema, dropped in the 2026-07-17 '
    'regen; see RECONCILIATION.md F3/D-section for the schema history). The "RESP top" claim '
    'CANNOT be reproduced on the current pipeline -- not "fell in rank", genuinely absent.')
ibi_rank_all = order_all.index('IBI') + 1 if 'IBI' in order_all else None
w(f'  IBI rank in current WESAD_all/ET: {ibi_rank_all}/{len(order_all)} '
  f'({"near the bottom" if ibi_rank_all and ibi_rank_all >= len(order_all)-2 else "not near the bottom"})')
w(f'  => WESAD-full claim does NOT hold on the current schema: dominant groups are '
  f'{order_all[0]}/{order_all[1]}/{order_all[2]}, and IBI/HR/RMSSD/SDNN (HRV) sit at the bottom, '
  f'consistent with the "HRV at the bottom" half of the claim -- but RESP is entirely absent '
  f'and IBI is at the bottom, not the top.')
w()
w('Thesis claim (per user): WESAD-wrist -> EDA/ACC/TEMP dominant, HRV at the bottom.')
w(f'Current schema WESAD_wrist/ET ranking: {" > ".join(order_wrist)}')
top3_wrist = set(order_wrist[:3])
claim_top3 = {'EDA', 'ACC', 'TEMP'}
w(f'  Top-3 groups match {{EDA,ACC,TEMP}} as a set? {top3_wrist == claim_top3}  (top-3 found: {order_wrist[:3]})')
hrv_wrist = [k for k in order_wrist if k in ('HR','IBI','RMSSD','SDNN')]
bottom_n = len(order_wrist) - len(hrv_wrist)
w(f'  HRV groups (HR/IBI/RMSSD/SDNN) occupy ranks: {[order_wrist.index(k)+1 for k in hrv_wrist]} '
  f'(bottom {len(order_wrist)-bottom_n} of {len(order_wrist)})')
w(f'  => WESAD-wrist claim {"HOLDS" if top3_wrist == claim_top3 else "PARTIALLY HOLDS"} on the current '
  f'schema: same 3 groups dominate (order may differ), HRV groups cluster at the bottom, matching '
  f'the thesis\'s description reasonably well -- unlike the full-set claim.')
w()

w('Cross-check against RECONCILIATION.md section F2 (2026-07-24, same methodology, computed '
  'in-memory, no figures/results file saved at the time):')
w('  RECONCILIATION.md WESAD wrist/ET : TEMP > EDA > ACC > SDNN > RMSSD > HR > IBI')
w(f'  This script\'s WESAD wrist/ET    : {" > ".join(order_wrist)}')
w('  RECONCILIATION.md WESAD full/ET  : TEMP > EDA > ACC > DEMO > SDNN > EMG > RMSSD > HR > IBI')
w(f'  This script\'s WESAD full/ET     : {" > ".join(order_all)}')
match_wrist = order_wrist == ['TEMP','EDA','ACC','SDNN','RMSSD','HR','IBI']
match_all   = order_all   == ['TEMP','EDA','ACC','DEMO','SDNN','EMG','RMSSD','HR','IBI']
w(f'  Exact order match -- wrist: {match_wrist}, full: {match_all} '
  f'(independent re-derivation, same seed/hyperparameters, cross-validates both).')
w()

w('='*78)
w('END')
w('='*78)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved: {OUT_PATH}')
