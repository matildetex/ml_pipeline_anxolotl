"""
Regenerates figures/fig_wesad_restricted_per_subject.png.

The original plotting script that produced this figure no longer exists in
the repo (documented in ANXOLOTL_AUDIT_FULL.txt:687). Reconstructed here from
the exact per-subject numbers in results/wesad_55plus_ecg_loso.txt (values
verified to match the original PNG bar-by-bar before this rewrite).

Change vs. the lost original: no title, larger fonts throughout (per user
request).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Per-fold MacroF1 from results/wesad_55plus_ecg_loso.txt, sorted by subject number.
DATA = {
    'S2':  (0.7571, 0.6731),
    'S3':  (0.3817, 0.9967),
    'S4':  (0.9768, 0.9501),
    'S5':  (0.9727, 1.0000),
    'S6':  (0.5265, 0.6288),
    'S7':  (0.8666, 0.9496),
    'S8':  (0.8445, 1.0000),
    'S9':  (0.4317, 0.5530),
    'S10': (0.4233, 0.8952),
    'S11': (0.6715, 1.0000),
    'S13': (0.6722, 0.9994),
    'S14': (0.8342, 0.9850),
    'S15': (0.7117, 0.7358),
    'S16': (0.6883, 1.0000),
    'S17': (0.5942, 0.9839),
}

subjects = list(DATA.keys())
f1_55 = np.array([DATA[s][0] for s in subjects])
f1_72 = np.array([DATA[s][1] for s in subjects])
delta = f1_72 - f1_55

FS = 15  # base font size (was ~10-11 in the original)
plt.rcParams.update({'font.size': FS})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 11), gridspec_kw={'height_ratios': [2.2, 1]}
)

x = np.arange(len(subjects))
w = 0.46
ax1.bar(x - w/2, f1_55, w, color='#5b8bb5', label='55-feat (wrist only)')
ax1.bar(x + w/2, f1_72, w, color='#2f5d45', label='72-feat (wrist + ECG)')

for xi, v in zip(x - w/2, f1_55):
    ax1.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=FS - 2)
for xi, v in zip(x + w/2, f1_72):
    ax1.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=FS - 2)

mean_55, mean_72 = f1_55.mean(), f1_72.mean()
ax1.axhline(mean_55, color='#5b8bb5', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axhline(mean_72, color='#2f5d45', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.text(x[-1] + 0.55, mean_55 + 0.02, f'mean={mean_55:.3f}',
          color='#5b8bb5', fontsize=FS - 2, ha='left')
ax1.text(x[-1] + 0.55, mean_72 + 0.02, f'mean={mean_72:.3f}',
          color='#2f5d45', fontsize=FS - 2, ha='left')

ax1.set_xticks(x)
ax1.set_xticklabels(subjects, fontsize=FS)
ax1.set_ylabel('Macro F1', fontsize=FS + 1)
ax1.set_xlim(x[0] - 0.7, x[-1] + 2.7)
ax1.set_ylim(0, 1.08)
ax1.legend(fontsize=FS - 1, loc='lower right')
ax1.tick_params(axis='both', labelsize=FS)

colors = ['#3fae5c' if d >= 0 else '#c0392b' for d in delta]
ax2.bar(x, delta, w, color=colors)
for xi, v in zip(x, delta):
    ax2.text(xi, v + (0.012 if v >= 0 else -0.055), f'{v:+.2f}',
              ha='center', va='bottom' if v >= 0 else 'top', fontsize=FS - 2)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(subjects, fontsize=FS)
ax2.set_ylabel('$\\Delta$F1\n(+ECG $-$ wrist)', fontsize=FS + 1)
ax2.set_xlim(x[0] - 0.7, x[-1] + 2.7)
ax2.set_ylim(delta.min() - 0.11, delta.max() + 0.08)
ax2.tick_params(axis='both', labelsize=FS)

handles = [plt.Rectangle((0, 0), 1, 1, color='#3fae5c', label='ECG improves'),
           plt.Rectangle((0, 0), 1, 1, color='#c0392b', label='ECG hurts')]
ax2.legend(handles=handles, fontsize=FS - 1, loc='lower right')

plt.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures',
                    'fig_wesad_restricted_per_subject.png')
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved {OUT}')
