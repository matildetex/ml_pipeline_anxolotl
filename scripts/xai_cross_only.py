"""Re-runs only the cross-dataset signal-group comparison from xai.py.
Use when xai.py already produced all other figures but crashed at the
cross_dataset_groups call due to feature-dimension mismatch."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use('Agg')

from common import load_wesad_tabular, load_wdd, load_anxolotl
from xai import cross_dataset_groups, prep_bl, prep_std

print('Loading datasets...')
X_wesad, y_wesad, g_wesad, fc_wesad = load_wesad_tabular()
X_wdd,   y_wdd,   g_wdd,   fc_wdd   = load_wdd()
X_anx,   y_anx,   g_anx,   fc_anx   = load_anxolotl()

fc_40 = sorted(set(fc_wesad) & set(fc_anx))
y_w = np.array(y_wesad)
y_d = np.array(y_wdd)
y_a = np.array(y_anx)

print(f'Preparing data ({len(fc_40)} common features)...')
Xw_40, _, _ = prep_std(X_wesad, fc_40)
Xd_40, _  = prep_bl(X_wdd, y_d, g_wdd, fc_40)
Xa_40, _  = prep_bl(X_anx, y_a, g_anx, fc_40)

print('Running cross_dataset_groups...')
cross_dataset_groups(Xw_40, y_w, Xd_40, y_d, Xa_40, y_a, fc_40, '40common')

print('Done. Figures saved to figures/shap/')
