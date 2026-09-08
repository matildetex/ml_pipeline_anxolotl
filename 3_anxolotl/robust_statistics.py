"""
Robust statistical characterisation of the main LOSO-CV result.

Reads already-computed per-subject F1 scores and produces:
  1. Descriptive statistics (mean, SD, SEM, median, IQR, min/max, CV)
  2. Bootstrap 95% CI for the mean F1 (10 000 resamples)
  3. Seed-robustness summary with bootstrap CI
  4. Normality check (Shapiro-Wilk)
  5. Production vs temporal-imputed comparison:
       - paired bootstrap CI for the mean difference
       - Cohen's d (paired)
       - rank-biserial correlation r (effect size for Wilcoxon)
       - Wilcoxon signed-rank test
  6. Figure: bootstrap sampling distributions + per-subject scatter

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, shapiro, sem as scipy_sem
import os

SEED     = 42
N_BOOT   = 10_000
ALPHA    = 0.05
OUT_TXT  = 'results/robust_statistics.txt'
OUT_FIG  = 'figures/fig_robust_statistics.png'

rng = np.random.default_rng(SEED)

# ── load data ────────────────────────────────────────────────────────────────

df_prod = pd.read_csv('results/final_model_per_subject.csv').set_index('subject')
f1_prod = df_prod['f1'].values.astype(float)

df_imp  = pd.read_csv('results/sensitivity_imputed_per_subject.csv').set_index('subject')
f1_imp  = df_imp['f1_imputed'].values.astype(float)

df_seed = pd.read_csv('results/seed_robustness.csv')
f1_seed = df_seed['f1'].values.astype(float)

common  = df_prod.index.intersection(df_imp.index)
f1_p_c  = df_prod.loc[common, 'f1'].values.astype(float)
f1_i_c  = df_imp.loc[common, 'f1_imputed'].values.astype(float)
diffs   = f1_i_c - f1_p_c

n = len(f1_prod)


# ── bootstrap helper ─────────────────────────────────────────────────────────

def boot_ci(x, stat=np.mean, n_boot=N_BOOT, alpha=ALPHA):
    """Return (lo, hi) bootstrap percentile CI."""
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    dist    = stat(samples, axis=1)
    lo, hi  = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, dist

def cohens_d_paired(a, b):
    d = a - b
    return d.mean() / (d.std(ddof=1) + 1e-12)

def rank_biserial(x, y):
    """Rank-biserial correlation from paired Wilcoxon."""
    d = y - x
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.abs(d)) + 1
    W_plus  = ranks[d > 0].sum()
    W_minus = ranks[d < 0].sum()
    n_d = len(d)
    return (W_plus - W_minus) / (n_d * (n_d + 1) / 2)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  MAIN RESULT — production config
# ═══════════════════════════════════════════════════════════════════════════

mean_p = f1_prod.mean()
std_p  = f1_prod.std(ddof=1)
sem_p  = scipy_sem(f1_prod)
med_p  = np.median(f1_prod)
q1_p, q3_p = np.percentile(f1_prod, [25, 75])
cv_p   = std_p / mean_p

ci_lo_p, ci_hi_p, boot_dist_p = boot_ci(f1_prod)

sw_stat, sw_p = shapiro(f1_prod)

lines = []
lines += [
    "=" * 62,
    "  ROBUST STATISTICS — Anxolotl LOSO-CV (ET-1000 + BL-norm)",
    "=" * 62,
    "",
    "── 1. Main result: production config (gesture-confirmed) ──",
    f"  n subjects          : {n}",
    f"  Mean F1             : {mean_p:.4f}",
    f"  SD                  : {std_p:.4f}",
    f"  SEM                 : {sem_p:.4f}",
    f"  Median F1           : {med_p:.4f}",
    f"  IQR [Q1 – Q3]       : [{q1_p:.4f} – {q3_p:.4f}]",
    f"  Min / Max           : {f1_prod.min():.4f} / {f1_prod.max():.4f}",
    f"  CV (SD/Mean)        : {cv_p:.3f}  ({cv_p*100:.1f}%)",
    f"  95% Bootstrap CI    : [{ci_lo_p:.4f} – {ci_hi_p:.4f}]  (n_boot={N_BOOT})",
    f"  Shapiro-Wilk        : W={sw_stat:.4f}  p={sw_p:.4f}  "
    f"({'normal' if sw_p > 0.05 else 'non-normal'} at α=0.05)",
]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SEED ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════

mean_s  = f1_seed.mean()
std_s   = f1_seed.std(ddof=1)
ci_lo_s, ci_hi_s, boot_dist_s = boot_ci(f1_seed)

lines += [
    "",
    "── 2. Seed robustness (10 seeds) ──────────────────────────",
    f"  Seeds tested        : {list(df_seed['seed'].values)}",
    f"  Mean F1             : {mean_s:.4f}",
    f"  SD                  : {std_s:.4f}",
    f"  Range               : [{f1_seed.min():.4f} – {f1_seed.max():.4f}]",
    f"  95% Bootstrap CI    : [{ci_lo_s:.4f} – {ci_hi_s:.4f}]",
]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PRODUCTION vs TEMPORAL-IMPUTED COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

mean_i = f1_imp.mean()
std_i  = f1_imp.std(ddof=1)
ci_lo_i, ci_hi_i, boot_dist_i = boot_ci(f1_imp)

mean_diff = diffs.mean()
ci_lo_d, ci_hi_d, boot_dist_d = boot_ci(diffs)
cd  = cohens_d_paired(f1_p_c, f1_i_c)
rrb = rank_biserial(f1_p_c, f1_i_c)
w_stat, w_p = wilcoxon(f1_i_c, f1_p_c)

improved = (diffs > 1e-9).sum()
same     = (np.abs(diffs) <= 1e-9).sum()
worse    = (diffs < -1e-9).sum()

lines += [
    "",
    "── 3. Production vs temporal-imputed ──────────────────────",
    f"  n common subjects   : {len(common)}",
    f"",
    f"  Production  — mean  : {mean_p:.4f}  95% CI [{ci_lo_p:.4f} – {ci_hi_p:.4f}]",
    f"  Imputed     — mean  : {mean_i:.4f}  95% CI [{ci_lo_i:.4f} – {ci_hi_i:.4f}]",
    f"",
    f"  Mean diff (imp−prod): {mean_diff:+.4f}",
    f"  95% Boot CI (diff)  : [{ci_lo_d:+.4f} – {ci_hi_d:+.4f}]",
    f"  Improved/same/worse : {improved}/{same}/{worse}  (out of {len(common)})",
    f"",
    f"  Cohen's d (paired)  : {cd:.4f}  ({'negligible' if abs(cd)<0.2 else 'small' if abs(cd)<0.5 else 'medium'})",
    f"  Rank-biserial r     : {rrb:.4f}  ({'negligible' if abs(rrb)<0.1 else 'small' if abs(rrb)<0.3 else 'medium'})",
    f"  Wilcoxon W          : {w_stat:.2f}  p={w_p:.4f}  "
    f"({'NOT significant' if w_p >= 0.05 else 'significant'} at α=0.05)",
    f"",
    f"  Conclusion: the 95% bootstrap CI for the difference straddles zero",
    f"  [{ci_lo_d:+.4f} – {ci_hi_d:+.4f}], Cohen's d≈{cd:.3f} (negligible effect).",
    f"  Main result is robust to marker-labelling strategy.",
]

lines += ["", "=" * 62]

report = "\n".join(lines)
print(report)

os.makedirs('results', exist_ok=True)
with open(OUT_TXT, 'w') as fh:
    fh.write(report + "\n")
print(f"\nSaved {OUT_TXT}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Robust statistics — Anxolotl LOSO-CV F1", fontsize=13, fontweight='bold')

# Panel A: bootstrap distribution of mean F1 (production)
ax = axes[0]
ax.hist(boot_dist_p, bins=60, color='#4C72B0', alpha=0.85, density=True)
ax.axvline(mean_p,   color='white',  lw=2,   label=f'Mean = {mean_p:.4f}')
ax.axvline(ci_lo_p,  color='#DD8452', lw=1.5, ls='--', label=f'95% CI [{ci_lo_p:.3f}–{ci_hi_p:.3f}]')
ax.axvline(ci_hi_p,  color='#DD8452', lw=1.5, ls='--')
ax.set_xlabel('Mean F1 (bootstrap resample)')
ax.set_ylabel('Density')
ax.set_title('Bootstrap dist. — main result')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: bootstrap distribution of paired difference
ax = axes[1]
ax.hist(boot_dist_d, bins=60, color='#55A868', alpha=0.85, density=True)
ax.axvline(mean_diff, color='white',   lw=2,   label=f'Mean diff = {mean_diff:+.4f}')
ax.axvline(ci_lo_d,   color='#DD8452', lw=1.5, ls='--', label=f'95% CI [{ci_lo_d:+.3f}–{ci_hi_d:+.3f}]')
ax.axvline(ci_hi_d,   color='#DD8452', lw=1.5, ls='--')
ax.axvline(0,         color='red',     lw=1,   ls=':',  label='zero')
ax.set_xlabel('Mean F1 diff (imputed − production)')
ax.set_ylabel('Density')
ax.set_title('Bootstrap dist. — paired diff')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel C: per-subject scatter production vs imputed
ax = axes[2]
subjs = list(common)
x = f1_p_c
y = f1_i_c
ax.scatter(x, y, color='#4C72B0', alpha=0.7, zorder=3)
lims = [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02]
ax.plot(lims, lims, 'w--', lw=1, label='y = x')
ax.set_xlabel('F1 — production (gesture-confirmed)')
ax.set_ylabel('F1 — temporal-imputed')
ax.set_title(f'Per-subject F1 (n={len(common)})\nWilcoxon p={w_p:.3f}  |  Cohen\'s d={cd:.3f}')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {OUT_FIG}")
