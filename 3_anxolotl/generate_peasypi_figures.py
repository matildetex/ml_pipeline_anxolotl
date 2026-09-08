"""
Figures for peasypi CNN comparison.
Run AFTER peasypi_cnn.py has produced results/peasypi_cnn_anxolotl.csv.
Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
os.makedirs('figures', exist_ok=True)

# ── Published peasypi WESAD numbers ──────────────────────────────────────────
WESAD_CONFIGS = ['E4', 'Fitbit', 'WearOS\n(ACC+TEMP)', 'Tizen']
WESAD_F1      = [0.878, 0.894, 0.801, 0.730]
WESAD_SENSORS = {
    'E4':    'BVP + EDA + TEMP + ACC',
    'Fitbit':'ACC + TEMP + EDA',
    'WearOS\n(ACC+TEMP)': 'ACC + TEMP',
    'Tizen': 'ACC + TEMP + BVP',
}

# ── Load our Anxolotl results ─────────────────────────────────────────────────
results_path = 'results/peasypi_cnn_anxolotl.csv'
if not os.path.exists(results_path):
    print(f'ERROR: {results_path} not found. Run peasypi_cnn.py first.')
    exit(1)

df_res  = pd.read_csv(results_path)
anx_f1  = df_res['f1'].values
anx_auc = df_res['auc'].values
mean_f1  = anx_f1.mean()
mean_auc = anx_auc.mean()
subjects = df_res['subject'].values

print(f'Loaded {len(df_res)} subjects, mean F1={mean_f1:.3f}, mean AUC={mean_auc:.3f}')

# ── Colours ──────────────────────────────────────────────────────────────────
C_WESAD  = '#4C72B0'
C_ANX    = '#DD8452'
C_ET     = '#C44E52'
C_WEAROS = '#55A868'

ET500_F1 = 0.922  # ExtraTrees 500 + BL-norm (Heo classifiers)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — peasypi CNN: WESAD configs vs Anxolotl adaptation
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('peasypi CNN (Gil-Martin et al. 2022) — WESAD vs Anxolotl adaptation',
             fontsize=12)

# Left: WESAD configs vs ours
ax = axes[0]
x = np.arange(len(WESAD_CONFIGS) + 1)
labels = WESAD_CONFIGS + ['Anxolotl\n(5ch FFT-CNN)']
f1s    = WESAD_F1 + [mean_f1]
colors = [C_WESAD] * 4 + [C_ANX]
bars   = ax.bar(x, f1s, 0.6, color=colors, alpha=0.88)
for bar, v in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(ET500_F1, color=C_ET, linestyle='--', linewidth=1.5,
           label=f'ET-500 + BL-norm (F1={ET500_F1})')
ax.axvline(3.5, color='grey', linestyle=':', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0, 1.12)
ax.set_title('CNN approach: published WESAD vs Anxolotl')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
wesad_patch = mpatches.Patch(color=C_WESAD, alpha=0.8, label='WESAD (published, peasypi)')
anx_patch   = mpatches.Patch(color=C_ANX,   alpha=0.8, label='Anxolotl (this work)')
ax.legend(handles=[wesad_patch, anx_patch,
                   plt.Line2D([0],[0], color=C_ET, linestyle='--',
                              label=f'ET-500 + BL-norm (F1={ET500_F1})')],
          fontsize=9, loc='lower right')

# Right: per-subject F1 Anxolotl
ax = axes[1]
sort_idx = np.argsort(anx_f1)
ax.bar(np.arange(len(sort_idx)), anx_f1[sort_idx], 0.65, color=C_ANX, alpha=0.88)
ax.axhline(mean_f1, color=C_ANX, linestyle='--', linewidth=2,
           label=f'Mean F1={mean_f1:.3f}')
ax.axhline(ET500_F1, color=C_ET, linestyle='--', linewidth=1.5,
           label=f'ET-500 mean F1={ET500_F1}')
ax.set_xticks(np.arange(len(sort_idx)))
ax.set_xticklabels(subjects[sort_idx], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Binary F1 (stress class)')
ax.set_title('Per-subject F1 — peasypi CNN on Anxolotl (LOSO-CV)')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig6_peasypi_cnn.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/fig6_peasypi_cnn.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Overall methodology comparison (3 approaches)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))

methods = [
    'Heo et al.\nANX + StdScaler\n(Heo classifiers)',
    'Heo et al.\nANX + BL-norm\n(Heo classifiers)',
    'peasypi CNN\nANX + BL-norm\n(FFT-based)',
    'This work\nANX + BL-norm\n(ExtraTrees 500)',
]
f1_vals = [0.678, 0.777, mean_f1, 0.922]   # means across Heo classifiers + ET500
colors  = ['#9E9E9E', '#DD8452', '#4C72B0', '#C44E52']
note    = ['mean 7 clf', 'mean 7 clf', '37 subjects', '37 subjects']

bars = ax.bar(range(4), f1_vals, 0.55, color=colors, alpha=0.88)
for bar, v, n in zip(bars, f1_vals, note):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, 0.02,
            n, ha='center', va='bottom', fontsize=8, color='white', style='italic')

ax.set_xticks(range(4))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0, 1.15)
ax.set_title('Anxolotl stress detection — comparison of approaches\n'
             '(all: LOSO-CV, binary F1 @ 0.5 threshold)')
ax.grid(axis='y', alpha=0.3)

# Highlight our best method
bars[3].set_edgecolor('black')
bars[3].set_linewidth(2)

# Dataset region annotation
ax.text(1.5, 1.12, 'Anxolotl dataset — LOSO-CV',
        ha='center', fontsize=10, style='italic', color='#555')

plt.tight_layout()
plt.savefig('figures/fig7_all_methods_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved figures/fig7_all_methods_comparison.png')
print('\nAll peasypi figures done.')
print(f'\nKey numbers:')
print(f'  peasypi CNN on Anxolotl (mean F1): {mean_f1:.3f}')
print(f'  peasypi WearOS on WESAD (F1):      0.801  (published)')
print(f'  ExtraTrees 500 + BL-norm (F1):     0.922  (our best)')
