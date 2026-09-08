"""
Grouped bar chart: effect of adding ECG on Macro F1, WESAD-wrist vs Anxolotl.
Values taken verbatim from results_master.csv / existing result files (not
recomputed here):
  WESAD wrist-only : 0.6902 (results/wesad_55plus_ecg_loso.txt, 55-feat, ExtraTrees+BL-norm)
  WESAD wrist+ECG  : 0.8900 (results/wesad_55plus_ecg_loso.txt, 72-feat, ExtraTrees+BL-norm)
  Anxolotl Watch   : 0.9169 (results_master.csv row 2, ET-1000, canonical BL-norm leak-fixed LOSO)
  Anxolotl Watch+ECG: 0.9503 (results_master.csv row 19, ET-1000, wrist-55+ECG-17, canonical LOSO)
Purpose: show the ECG gain is large on WESAD-wrist but small on Anxolotl.
Saved under results/section1_2_wesad_wrist55_blnorm_2026-07-26/figures/, per
user's explicit choice of output directory -- does not touch anything else.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/matildxi/Secretária/Documentos/Tese/anxolotl'
FIG_DIR = os.path.join(ROOT, 'results', 'section1_2_wesad_wrist55_blnorm_2026-07-26', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PATH = os.path.join(FIG_DIR, 'ecg_effect_wesad_wrist_vs_anxolotl.png')

groups = ['WESAD wrist-only', 'Anxolotl']
no_ecg = [0.6902, 0.9169]
with_ecg = [0.8900, 0.9503]

x = np.arange(len(groups))
width = 0.32

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 15, 'figure.dpi': 150})
fig, ax = plt.subplots(figsize=(7, 5.5))

b1 = ax.bar(x - width/2, no_ecg, width, label='without ECG', color='tab:blue', alpha=0.85)
b2 = ax.bar(x + width/2, with_ecg, width, label='with ECG', color='tab:red', alpha=0.85)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=13)

for gi, (a, b) in enumerate(zip(no_ecg, with_ecg)):
    ax.annotate(f'+{b-a:.4f}', xy=(x[gi], max(a, b) + 0.055),
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='dimgray')

ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=15)
ax.set_ylabel('Macro F1', fontsize=16)
ax.set_ylim(0, 1.08)
ax.tick_params(axis='both', labelsize=15)
ax.legend(loc='lower right', fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {os.path.relpath(OUT_PATH, ROOT)}')
print(f'  WESAD wrist-only: {no_ecg[0]:.4f} -> {with_ecg[0]:.4f}  (delta +{with_ecg[0]-no_ecg[0]:.4f})')
print(f'  Anxolotl        : {no_ecg[1]:.4f} -> {with_ecg[1]:.4f}  (delta +{with_ecg[1]-no_ecg[1]:.4f})')
