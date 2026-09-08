"""
Redraws wesad_wrist55_ET_blnorm_shap_groups.png with a FIXED group->color
mapping so it is visually comparable to figures/shap/reverified_2026-07-25/
WESAD_all_ET_groups.png (the full-137-feat panel, Figure 1.3a -- untouched).

Why a fix was needed: the existing groups_plot() convention used elsewhere in
this repo (scripts/wesad_shap_reverification.py, and the run_shap_blnorm.py
script that first drew this figure) assigns bar colors by RANK POSITION
(plt.cm.tab10(np.linspace(0, 1, len(order))) over the importance-sorted
`order` list), not by group identity. That means the same group (e.g. ACC)
gets a different color in every figure depending on where it lands in that
figure's own ranking -- confirmed by inspecting WESAD_all_ET_groups.png:
TEMP=blue, EDA=orange, ACC=green, DEMO=red, SDNN=brown/8c564b, EMG=pink,
RMSSD=gray, HR=olive, IBI=cyan (reproduced exactly via
plt.cm.tab10(np.linspace(0,1,9)) over that same 9-group order).

Fixed mapping (per explicit user instruction, overriding the earlier
WESAD_all_ET_groups.png-matched attempt):
  ACC   -> #8c564b (brown)
  SDNN  -> #7f7f7f (grey)
  RMSSD -> #bcbd22 (olive)
  IBI   -> #ff7f0e (orange)
  HR    -> #2ca02c (green)

Group values are NOT recomputed -- taken verbatim from
results/section1_2_wesad_wrist55_blnorm_2026-07-26/shap_stdout.log (run_shap_blnorm.py's
STEP 4 output), so this script only changes bar color/order-of-color-assignment,
nothing about the SHAP computation itself. Overwrites the PNG this same job
directory already produced; touches nothing outside this directory.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
FIG_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_blnorm_2026-07-26', 'figures')
OUT_PATH = os.path.join(FIG_DIR, 'wesad_wrist55_ET_blnorm_shap_groups.png')

# Verbatim from shap_stdout.log STEP 4.
g = {'ACC': 0.2615, 'SDNN': 0.1077, 'RMSSD': 0.0904, 'IBI': 0.0548, 'HR': 0.0516}
order = sorted(g, key=g.get, reverse=True)

FIXED_COLORS = {
    'ACC':   '#8c564b',
    'SDNN':  '#7f7f7f',
    'RMSSD': '#bcbd22',
    'IBI':   '#ff7f0e',
    'HR':    '#2ca02c',
}

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'figure.dpi': 150})
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(order, [g[k] for k in order], color=[FIXED_COLORS[k] for k in order], alpha=0.85)
ax.set_ylabel('Summed |SHAP| Importance')
ax.set_title('Signal-Group SHAP Importance — WESAD wrist-55, BL-norm, ExtraTrees\n'
             '(replaces Figure 1.3b)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {os.path.relpath(OUT_PATH, ROOT)}')
for k in order:
    print(f'  {k:6s} color={FIXED_COLORS[k]}  value={g[k]}')
