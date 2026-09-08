"""
Figures for Heo et al. (2021) pipeline validation and comparison.
Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})

CLFS = ['DT', 'RF', 'AB', 'KN', 'LDA', 'SVM', 'GB']
x = np.arange(len(CLFS))

# ── Data ──────────────────────────────────────────────────────────────────────

WESAD_RUN_F1  = [0.875, 0.869, 0.865, 0.884, 0.906, 0.890, 0.885]
WESAD_RUN_AUC = [0.911, 0.904, 0.898, 0.924, 0.932, 0.928, 0.912]
WESAD_REPO_F1 = [0.590, 0.585, 0.613, 0.667, 0.630, 0.625, 0.611]  # buggy repo files

ANX_STD_F1  = [0.648, 0.696, 0.704, 0.620, 0.672, 0.692, 0.713]
ANX_BL_F1   = [0.885, 0.844, 0.847, 0.572, 0.684, 0.718, 0.892]
ANX_STD_AUC = [0.597, 0.717, 0.735, 0.616, 0.699, 0.712, 0.763]
ANX_BL_AUC  = [0.873, 0.941, 0.934, 0.869, 0.732, 0.761, 0.960]

ET500_F1  = 0.922
ET500_AUC = 0.976

PAPER_F1  = 0.9336

SUBJECTS_WESAD = ['S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S13','S14','S15','S16','S17']
WESAD_PER_SUBJ_F1 = {
    'DT':  [0.889,0.850,1.000,0.941,0.941,1.000,0.897,0.750,0.571,0.780,0.903,0.884,0.828,1.000,0.895],
    'RF':  [0.889,0.850,1.000,0.889,0.839,1.000,0.846,0.774,0.556,0.875,0.903,0.974,0.733,1.000,0.900],
    'AB':  [0.783,0.850,1.000,1.000,0.909,1.000,0.846,0.733,0.556,0.941,0.824,0.974,0.774,0.944,0.837],
    'KN':  [0.929,0.900,1.000,0.762,0.848,1.000,0.929,0.774,1.000,0.973,0.867,0.824,0.688,0.944,0.818],
    'LDA': [0.923,0.818,1.000,1.000,0.875,1.000,0.889,0.714,0.974,1.000,0.938,0.973,0.667,1.000,0.818],
    'SVM': [0.963,0.878,1.000,0.800,0.909,1.000,0.929,0.710,1.000,1.000,0.909,0.813,0.667,0.973,0.800],
    'GB':  [0.923,0.878,1.000,1.000,0.909,1.000,0.750,0.800,0.571,0.889,0.839,0.950,0.867,0.973,0.923],
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_WESAD   = '#4C72B0'
C_ANX_STD = '#DD8452'
C_ANX_BL  = '#55A868'
C_ET      = '#C44E52'
C_REPO    = '#9E9E9E'
C_PAPER   = '#8172B3'

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Main comparison: WESAD (run) vs Anxolotl (StandardScaler vs BL-norm)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.5))
w = 0.22
bars1 = ax.bar(x - w,   WESAD_RUN_F1, w, label='WESAD — Heo et al. pipeline (our run)',   color=C_WESAD,   alpha=0.88)
bars2 = ax.bar(x,       ANX_STD_F1,   w, label='Anxolotl — StandardScaler (Heo et al.)',  color=C_ANX_STD, alpha=0.88)
bars3 = ax.bar(x + w,   ANX_BL_F1,    w, label='Anxolotl — BL-norm (this work)',          color=C_ANX_BL,  alpha=0.88)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)

ax.axhline(ET500_F1,  color=C_ET,    linestyle='--', linewidth=1.5, label=f'Anxolotl — ExtraTrees 500 + BL-norm (F1={ET500_F1})')
ax.axhline(PAPER_F1,  color=C_PAPER, linestyle=':',  linewidth=1.5, label=f'WESAD — Heo et al. paper claim (F1={PAPER_F1})')

ax.set_xticks(x); ax.set_xticklabels(CLFS)
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0, 1.08)
ax.set_title('Heo et al. (2021) classifiers — WESAD vs Anxolotl\n'
             'Same pipeline, two datasets, two normalizations')
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig1_main_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_main_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Repo bug disclosure: published files vs actual run
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bars_repo = ax.bar(x - 0.2, WESAD_REPO_F1, 0.38, label='Repo result files (published, buggy)', color=C_REPO,  alpha=0.88)
bars_run  = ax.bar(x + 0.2, WESAD_RUN_F1,  0.38, label='Actual run (our execution)',            color=C_WESAD, alpha=0.88)
for bars in [bars_repo, bars_run]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
ax.axhline(PAPER_F1, color=C_PAPER, linestyle=':', linewidth=1.5, label=f'Paper claim (F1={PAPER_F1})')
ax.set_xticks(x); ax.set_xticklabels(CLFS)
ax.set_ylabel('Binary F1')
ax.set_ylim(0, 1.08)
ax.set_title('WESAD — Published repo files vs actual pipeline run\n(repo files contain bugs: debug subset_ids, boolean labels)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
delta = [r - p for r, p in zip(WESAD_RUN_F1, WESAD_REPO_F1)]
bars = ax.bar(x, delta, 0.5, color=C_WESAD, alpha=0.88)
for bar, d in zip(bars, delta):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'+{d:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(CLFS)
ax.set_ylabel('ΔF1 (actual run − repo files)')
ax.set_title('F1 gain from running the correct pipeline')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig2_repo_bug.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_repo_bug.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Normalization effect: StandardScaler vs BL-norm (Anxolotl only)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bars_std = ax.bar(x - 0.2, ANX_STD_F1, 0.38, label='StandardScaler (Heo et al.)', color=C_ANX_STD, alpha=0.88)
bars_bl  = ax.bar(x + 0.2, ANX_BL_F1,  0.38, label='BL-norm (this work)',         color=C_ANX_BL,  alpha=0.88)
for bars in [bars_std, bars_bl]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
ax.axhline(ET500_F1, color=C_ET, linestyle='--', linewidth=1.5, label=f'ET-500 + BL-norm (F1={ET500_F1})')
ax.set_xticks(x); ax.set_xticklabels(CLFS)
ax.set_ylabel('Binary F1 (stress class)')
ax.set_ylim(0, 1.08)
ax.set_title('Anxolotl LOSO-CV — Effect of normalization\n(same classifiers, same data, same splits)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
delta_norm = [b - s for b, s in zip(ANX_BL_F1, ANX_STD_F1)]
colors = [C_ANX_BL if d >= 0 else C_ET for d in delta_norm]
bars = ax.bar(x, delta_norm, 0.5, color=colors, alpha=0.88)
for bar, d in zip(bars, delta_norm):
    ypos = bar.get_height() + 0.003 if d >= 0 else bar.get_height() - 0.018
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{d:+.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(CLFS)
ax.set_ylabel('ΔF1 (BL-norm − StandardScaler)')
ax.set_title('Gain from BL-norm vs StandardScaler\n(Anxolotl, per classifier)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_normalization_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_normalization_effect.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-subject F1 heatmap on WESAD (actual run)
# ══════════════════════════════════════════════════════════════════════════════
heatmap_data = pd.DataFrame(WESAD_PER_SUBJ_F1, index=SUBJECTS_WESAD)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0, vmax=1, ax=ax, linewidths=0.3,
            cbar_kws={'label': 'Binary F1 (stress class)'})
ax.set_title('Per-subject F1 — Heo et al. pipeline on WESAD (our run, LOSO-CV)', fontsize=12)
ax.set_xlabel('Classifier')
ax.set_ylabel('Subject')
plt.tight_layout()
plt.savefig('figures/fig4_wesad_per_subject.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4_wesad_per_subject.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Summary: all methods, both datasets
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

methods = ['WESAD\n(Heo repo\nfiles)', 'WESAD\n(Heo pipeline\nour run)',
           'Anxolotl\nStandardScaler', 'Anxolotl\nBL-norm (avg)', 'Anxolotl\nET-500 + BL-norm']
f1_vals  = [np.mean(WESAD_REPO_F1), np.mean(WESAD_RUN_F1),
            np.mean(ANX_STD_F1), np.mean(ANX_BL_F1), ET500_F1]
colors   = [C_REPO, C_WESAD, C_ANX_STD, C_ANX_BL, C_ET]
patterns = ['//','','//','','']

bars = ax.bar(range(5), f1_vals, 0.55, color=colors, alpha=0.88)
for i, (bar, v) in enumerate(zip(bars, f1_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(PAPER_F1, color=C_PAPER, linestyle=':', linewidth=2,
           label=f'Heo et al. paper claim (F1={PAPER_F1})')
ax.set_xticks(range(5))
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Mean Binary F1 (stress class)')
ax.set_ylim(0, 1.12)
ax.set_title('Summary — Heo et al. (2021) classifiers across conditions')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Annotate dataset regions
ax.axvspan(-0.4, 1.4, alpha=0.04, color=C_WESAD, label='_WESAD')
ax.axvspan(1.6, 4.4, alpha=0.04, color=C_ANX_BL, label='_Anxolotl')
ax.text(0.5, 1.09, 'WESAD', ha='center', fontsize=10, color=C_WESAD, fontweight='bold')
ax.text(3.0, 1.09, 'Anxolotl', ha='center', fontsize=10, color='#2e7d32', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig5_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5_summary.png')

print('\nAll figures saved to figures/')
