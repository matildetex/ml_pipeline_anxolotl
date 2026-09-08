"""
Regenerate SHAP figures for Anxolotl + WDD with the CURRENT feature schema
(11 stats/channel, no abs_mean — matches the WESAD regeneration of 2026-07-16).

The old 14-jun figures (Anxolotl_40common/60all, WDD_60common/90all) were
produced by a prior script version with a single generic 'Ensemble' key and a
different (pre-unification) feature schema. They are superseded and removed.

Note: for Anxolotl, the old '40common' vs '60all' split no longer applies —
Anxolotl's full 55-feature set is now a strict subset of both WESAD's and
WDD's feature sets, so there is only one meaningful Anxolotl configuration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

from xai import run_dataset_tabular, run_autogluon_train
from common import load_wdd, load_anxolotl

print('Loading datasets…')
X_wdd, y_wdd, g_wdd, fc_wdd = load_wdd()
X_anx, y_anx, g_anx, fc_anx = load_anxolotl()

fc_55common = sorted(set(fc_wdd) & set(fc_anx))
assert fc_55common == sorted(fc_anx), "Anxolotl no longer a strict subset of WDD — re-check before proceeding"

print(f'Anxolotl: {len(fc_anx)} features (own = common with WDD/WESAD, no longer distinct)')
print(f'WDD     : {len(fc_wdd)} features total, {len(fc_55common)} shared with Anxolotl')

# ── Anxolotl — single configuration ────────────────────────────────────────
print('\n' + '='*60)
print('  Anxolotl — 55 features (own = full = common)')
print('='*60)
Xa_55, _ = run_dataset_tabular(X_anx, y_anx, g_anx, fc_anx, 'Anxolotl_55all', use_blnorm=True)
run_autogluon_train(Xa_55, y_anx, fc_anx, 'Anxolotl_55all', use_blnorm=True)

# ── WDD — shared-with-Anxolotl subset ──────────────────────────────────────
print('\n' + '='*60)
print('  WDD — 55 features shared with Anxolotl')
print('='*60)
Xd_55, _ = run_dataset_tabular(X_wdd, y_wdd, g_wdd, fc_55common, 'WDD_55common', use_blnorm=True)
run_autogluon_train(Xd_55, y_wdd, fc_55common, 'WDD_55common', use_blnorm=True)

# ── WDD — own full feature set ─────────────────────────────────────────────
print('\n' + '='*60)
print('  WDD — 83 features (own, full)')
print('='*60)
Xd_83, _ = run_dataset_tabular(X_wdd, y_wdd, g_wdd, fc_wdd, 'WDD_83all', use_blnorm=True)
run_autogluon_train(Xd_83, y_wdd, fc_wdd, 'WDD_83all', use_blnorm=True)

print('\nDone.')
