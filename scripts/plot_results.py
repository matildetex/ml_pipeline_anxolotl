"""
Generate all comparison plots from results/*.json and results/*.txt
Output: figures/*.png  (publication-quality, 300 dpi)

Usage:
    python3 scripts/plot_results.py
"""
import json, os, re, glob, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})
PALETTE   = sns.color_palette('tab10')
C_WRIST   = PALETTE[0]   # blue
C_ALL     = PALETTE[1]   # orange
C_GREY    = '#aaaaaa'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_json_results():
    """Returns {filename_stem: [result_dict, ...]}"""
    data = {}
    for jf in sorted(glob.glob(os.path.join(RES_DIR, '*.json'))):
        stem = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf) as f:
                d = json.load(f)
            data[stem] = d if isinstance(d, list) else [d]
        except Exception as e:
            print(f'  [WARN] could not load {jf}: {e}')
    return data


def parse_cross_dataset_txt(path):
    """Parse lines like:  name   F1=0.867  F1@opt=0.901  AUC=0.923  t=0.521"""
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            m = re.search(
                r'^\s+(.+?)\s+F1=([\d.]+)\s+F1@opt=([\d.]+)\s+AUC=([\d.]+)',
                line)
            if m:
                results.append({
                    'label': m.group(1).strip(),
                    'f1':    float(m.group(2)),
                    'f1opt': float(m.group(3)),
                    'auc':   float(m.group(4)),
                })
            # blnorm format
            m2 = re.search(r'(ExtraTrees\s+\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m2:
                results.append({
                    'label': m2.group(1).strip(),
                    'auc':   float(m2.group(2)),
                    'f1':    float(m2.group(3)),
                    'f1opt': float(m2.group(4)),
                    'mcc':   float(m2.group(5)),
                })
    return results


def _load_cd_json(path):
    if not os.path.exists(path): return []
    try:
        with open(path) as f: d = json.load(f)
        return d if isinstance(d, list) else [d]
    except Exception:
        return []

def get_cross_dataset():
    """Load cross-dataset results from JSON (preferred) or fall back to txt parsing."""
    wdd   = _load_cd_json(os.path.join(RES_DIR, 'cross_dataset_wdd.json'))
    wesad = _load_cd_json(os.path.join(RES_DIR, 'cross_dataset_wesad.json'))
    dann  = _load_cd_json(os.path.join(RES_DIR, 'cross_dataset_dann.json'))
    # fallback: parse txt if JSON not available yet
    if not wdd:   wdd   = parse_cross_dataset_txt(os.path.join(RES_DIR, 'cross_dataset_wdd.txt'))
    if not wesad: wesad = parse_cross_dataset_txt(os.path.join(RES_DIR, 'cross_dataset_wesad.txt'))
    if not dann:  dann  = parse_cross_dataset_txt(os.path.join(RES_DIR, 'cross_dataset_dann.txt'))
    return wdd, wesad, dann


def split_wrist_all(records):
    wrist = [r for r in records if r and 'wrist' in r.get('label','').lower()]
    all_f = [r for r in records if r and 'wrist' not in r.get('label','').lower()
             and 'all' in r.get('label','').lower()]
    return wrist, all_f


def short_name(label):
    """Shorten long model labels for axis ticks."""
    label = re.sub(r'\s*\(.*?\)', '', label).strip()
    subs = {
        'LogReg WESAD': 'LogReg', 'SVM WESAD': 'SVM',
        'Ensemble WESAD': 'Ensemble', '1D-CNN WESAD': '1D-CNN',
        'CNN-GRU WESAD': 'CNN-GRU', 'RNN (BiGRU) WESAD': 'RNN',
        'Transformer WESAD': 'Transformer', 'Mamba (SSM) WESAD': 'Mamba',
        'DANN WESAD': 'DANN', 'AutoGluon WESAD': 'AutoGluon',
        'ET  Anxolotl': 'ET', 'RF  Anxolotl': 'RF', 'EN  Anxolotl': 'Ensemble',
    }
    for k, v in subs.items():
        if label.startswith(k):
            return v
    return label[:20]


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  saved → {path}')


def grouped_bar(ax, labels, vals_a, vals_b, label_a='Wrist', label_b='All features',
                color_a=C_WRIST, color_b=C_ALL, ylabel='Macro F1',
                errs_a=None, errs_b=None):
    x   = np.arange(len(labels))
    w   = 0.35
    kw  = dict(capsize=3, error_kw={'elinewidth': 1})
    ax.bar(x - w/2, vals_a, w, label=label_a, color=color_a, alpha=0.85,
           yerr=errs_a, **kw)
    ax.bar(x + w/2, vals_b, w, label=label_b, color=color_b, alpha=0.85,
           yerr=errs_b, **kw)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.7)
    for xi, v in zip(x - w/2, vals_a):
        if not np.isnan(v): ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for xi, v in zip(x + w/2, vals_b):
        if not np.isnan(v): ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — WESAD LOSO: F1 comparison (wrist vs all)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wesad_comparison(data):
    wesad_files = [k for k in data if k.startswith('wesad_')]
    if not wesad_files:
        print('  [SKIP] no WESAD results found'); return

    rows_w, rows_a = [], []
    for stem in sorted(wesad_files):
        records = data[stem]
        for r in records:
            if r is None: continue
            if 'wrist' in r.get('label','').lower():
                rows_w.append(r)
            elif 'all' in r.get('label','').lower():
                rows_a.append(r)

    if not rows_w: print('  [SKIP] no wrist results'); return

    # align by model name
    names_w = [short_name(r['label']) for r in rows_w]
    names_a = [short_name(r['label']) for r in rows_a]
    labels  = names_w  # use wrist order as reference

    f1_w  = [r.get('f1', float('nan'))  for r in rows_w]
    f1_a  = [r.get('f1', float('nan'))  for r in rows_a] if rows_a else [float('nan')] * len(labels)
    auc_w = [r.get('auc', float('nan')) for r in rows_w]
    auc_a = [r.get('auc', float('nan')) for r in rows_a] if rows_a else [float('nan')] * len(labels)
    std_w = [r.get('f1_std', None) or 0 for r in rows_w]
    std_a = [r.get('f1_std', None) or 0 for r in rows_a] if rows_a else [0] * len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('WESAD LOSO — Wrist vs All Features', fontsize=14, fontweight='bold')

    grouped_bar(axes[0], labels, f1_w, f1_a, ylabel='Macro F1',
                errs_a=std_w, errs_b=std_a)
    axes[0].set_title('Macro F1')

    grouped_bar(axes[1], labels, auc_w, auc_a, ylabel='AUC-ROC')
    axes[1].set_title('AUC-ROC')

    plt.tight_layout()
    savefig('fig1_wesad_comparison.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Wrist vs All: Δ F1 per model
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wrist_vs_all_delta(data):
    wesad_files = [k for k in data if k.startswith('wesad_')]
    if not wesad_files: return

    deltas, names = [], []
    for stem in sorted(wesad_files):
        records = data[stem]
        r_w = next((r for r in records if r and 'wrist' in r.get('label','').lower()), None)
        r_a = next((r for r in records if r and 'all'   in r.get('label','').lower()), None)
        if r_w and r_a and r_w.get('f1') and r_a.get('f1'):
            deltas.append(r_a['f1'] - r_w['f1'])
            names.append(short_name(r_w['label']))

    if not deltas: return

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE[2] if d >= 0 else PALETTE[3] for d in deltas]
    x = np.arange(len(names))
    ax.bar(x, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('ΔF1  (All – Wrist)')
    ax.set_title('Gain from Adding Chest Sensors (WESAD LOSO)')
    pos = mpatches.Patch(color=PALETTE[2], alpha=0.85, label='All > Wrist')
    neg = mpatches.Patch(color=PALETTE[3], alpha=0.85, label='Wrist > All')
    ax.legend(handles=[pos, neg], framealpha=0.7)
    for xi, d in zip(x, deltas):
        ax.text(xi, d + (0.003 if d >= 0 else -0.008), f'{d:+.3f}',
                ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)
    plt.tight_layout()
    savefig('fig2_wrist_vs_all_delta.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Generic dataset LOSO comparison (WDD / Anxolotl)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dataset_comparison(data, stem, title, key_a, key_b, fname):
    """Generic grouped bar for any dataset with two feature-set variants."""
    if stem not in data:
        print(f'  [SKIP] no {stem} results'); return
    records = [r for r in data[stem] if r]
    rows_a = [r for r in records if key_a in r.get('label','')]
    rows_b = [r for r in records if key_b in r.get('label','')]
    if not rows_a and not rows_b:
        print(f'  [SKIP] no matching records in {stem}'); return

    # get unique model names from label prefix
    import re as _re
    def model_name(lbl):
        return _re.sub(r'\s*(WDD|Anxolotl)[^—]*—.*', '', lbl).strip()

    names = list(dict.fromkeys(model_name(r['label']) for r in rows_a or rows_b))
    def val(rows, name, key):
        r = next((x for x in rows if model_name(x['label']) == name), None)
        return r.get(key, float('nan')) if r else float('nan')

    f1_a  = [val(rows_a, n, 'f1')  for n in names]
    f1_b  = [val(rows_b, n, 'f1')  for n in names]
    auc_a = [val(rows_a, n, 'auc') for n in names]
    auc_b = [val(rows_b, n, 'auc') for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title} — Feature Sets Comparison', fontsize=14, fontweight='bold')
    grouped_bar(axes[0], names, f1_a, f1_b, label_a=key_a, label_b=key_b, ylabel='Macro F1')
    axes[0].set_title('Macro F1')
    grouped_bar(axes[1], names, auc_a, auc_b, label_a=key_a, label_b=key_b, ylabel='AUC-ROC')
    axes[1].set_title('AUC-ROC')
    plt.tight_layout()
    savefig(fname)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Anxolotl LOSO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_anxolotl(data):
    if 'anxolotl_loso' not in data:
        print('  [SKIP] no anxolotl results'); return
    # delegate to generic comparison
    plot_dataset_comparison(data, 'anxolotl_loso',
                            'Anxolotl LOSO (BL-norm)', '40 common', '60 all',
                            'fig4_anxolotl_loso.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Cross-dataset comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cross_dataset(data):
    wdd, wesad_cd, dann = get_cross_dataset()

    # Also grab best from JSON results if cross-dataset JSON exists
    # Best known baselines
    baselines = [
        {'label': 'LOSO ceiling\n(Anxolotl)', 'f1': 0.923, 'auc': 0.974, 'type': 'ceiling'},
    ]

    # Best results from parsed text files
    best_wdd = max(wdd,   key=lambda r: r['f1'], default=None) if wdd   else None
    best_wes = max(wesad_cd, key=lambda r: r['f1'], default=None) if wesad_cd else None
    best_dan = max(dann,  key=lambda r: r['f1'], default=None) if dann  else None

    entries = []
    if best_wdd: entries.append({**best_wdd, 'type': 'WDD→Anxolotl'})
    if best_wes: entries.append({**best_wes, 'type': 'WESAD→Anxolotl'})
    if best_dan: entries.append({**best_dan, 'type': 'DANN'})
    entries += baselines

    if len(entries) < 2:
        print('  [SKIP] insufficient cross-dataset results'); return

    labels  = [e['type'] for e in entries]
    f1_vals = [e.get('f1', float('nan'))  for e in entries]
    auc_vals= [e.get('auc', float('nan')) for e in entries]
    colors  = [PALETTE[4] if e['type'] != 'ceiling' else C_GREY for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cross-Dataset Generalisation (Train→Test on Anxolotl)', fontsize=14, fontweight='bold')

    for ax, vals, metric in [(axes[0], f1_vals, 'Macro F1'), (axes[1], auc_vals, 'AUC-ROC')]:
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel(metric); ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        # dashed line at LOSO ceiling
        ceil = next((e.get(metric.split('-')[0].lower().replace(' ', ''),
                           float('nan')) for e in entries if e['type'] == 'ceiling'), float('nan'))
        if not np.isnan(ceil):
            ax.axhline(ceil, color=C_GREY, linestyle='--', linewidth=1.2,
                       label='LOSO ceiling')
            ax.legend(framealpha=0.7)

    plt.tight_layout()
    savefig('fig5_cross_dataset.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — ROC curves (WESAD wrist best models)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(data, feature_key='wrist', title_suffix='Wrist Features',
                    fname='fig5_roc_wesad_wrist.png', stem=None):
    if stem:
        file_keys = [stem] if stem in data else []
    else:
        file_keys = sorted(k for k in data if k.startswith('wesad_'))

    plotted = False
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1],[0,1], 'k--', lw=0.8, label='Random')

    colors_iter = iter(plt.cm.tab10(np.linspace(0, 1, max(len(file_keys), 1))))
    for key in file_keys:
        records = data[key]
        r = next((x for x in records
                  if x and feature_key in x.get('label','').lower()
                  and x.get('fpr') and x.get('tpr')), None)
        if r is None: continue
        fpr, tpr = np.array(r['fpr']), np.array(r['tpr'])
        auc_val = r.get('auc', float('nan'))
        name = short_name(r['label'])
        c = next(colors_iter)
        ax.plot(fpr, tpr, lw=1.8, color=c,
                label=f'{name}  (AUC={auc_val:.3f})')
        plotted = True

    if not plotted:
        plt.close()
        print(f'  [SKIP] no ROC data for {feature_key}'); return

    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {title_suffix}')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    savefig(fname)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — ROC curves Anxolotl
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_anxolotl(data):
    if 'anxolotl_loso' not in data:
        print('  [SKIP] no anxolotl results'); return
    records = [r for r in data['anxolotl_loso']
               if r and r.get('fpr') and r.get('tpr')]
    if not records:
        print('  [SKIP] no ROC data for anxolotl'); return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'k--', lw=0.8, label='Random')

    for i, r in enumerate(records):
        fpr, tpr = np.array(r['fpr']), np.array(r['tpr'])
        name = short_name(r['label'])
        ax.plot(fpr, tpr, lw=2, color=PALETTE[i % 10],
                label=f'{name}  (AUC={r.get("auc", float("nan")):.3f})')

    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves — Anxolotl LOSO (BL-norm)')
    ax.legend(loc='lower right'); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    plt.tight_layout()
    savefig('fig7_roc_anxolotl.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Metrics heatmap (all models × all metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metrics_heatmap(data):
    metrics = ['f1', 'auc', 'f1opt', 'ba', 'mcc']
    metric_labels = ['Macro F1', 'AUC', 'F1@opt', 'Bal.Acc', 'MCC']

    rows = []
    for stem in sorted(data.keys()):
        for r in data[stem]:
            if r is None: continue
            rows.append({
                'Model': short_name(r.get('label', stem)),
                'Features': 'wrist' if 'wrist' in r.get('label','').lower() else 'all',
                **{m: r.get(m, float('nan')) for m in metrics}
            })

    if not rows:
        print('  [SKIP] no data for heatmap'); return

    import pandas as pd
    df = pd.DataFrame(rows)
    df['Model_feat'] = df['Model'] + '\n(' + df['Features'] + ')'
    df = df.set_index('Model_feat')[metrics]
    df.columns = metric_labels

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.45)))
    sns.heatmap(df.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={'shrink': 0.6, 'label': 'Score'})
    ax.set_title('Metrics Overview — All Models & Feature Sets', pad=12)
    ax.set_xlabel('')
    plt.tight_layout()
    savefig('fig8_metrics_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — F1 per fold (variability across subjects)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fold_variability(data):
    wesad_files = sorted(k for k in data if k.startswith('wesad_'))
    box_data, box_labels = [], []

    for stem in wesad_files:
        r = next((x for x in data[stem]
                  if x and x.get('fold_f1s') and 'wrist' in x.get('label','').lower()), None)
        if r:
            box_data.append(r['fold_f1s'])
            box_labels.append(short_name(r['label']))

    if not box_data:
        print('  [SKIP] no fold-level data'); return

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(box_data)))):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(box_labels)+1))
    ax.set_xticklabels(box_labels, rotation=30, ha='right')
    ax.set_ylabel('Macro F1 per Subject Fold')
    ax.set_title('Per-Subject F1 Variability — WESAD LOSO (Wrist Features)')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    savefig('fig9_fold_variability.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Confusion matrices grid (best model per category)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(data):
    # pick best wrist result per model file
    candidates = []
    for stem in sorted(data.keys()):
        r = next((x for x in data[stem]
                  if x and x.get('cm') and 'wrist' in x.get('label','').lower()), None)
        if r:
            candidates.append(r)

    if not candidates:
        print('  [SKIP] no confusion matrix data'); return

    n   = len(candidates)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle('Confusion Matrices — Wrist Features', fontsize=14, fontweight='bold')

    for ax, r in zip(axes, candidates):
        cm = np.array(r['cm'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['BL','ST'], yticklabels=['BL','ST'],
                    cbar=False, linewidths=0.5)
        ax.set_title(short_name(r['label']), fontsize=10)
        ax.set_xlabel('Pred'); ax.set_ylabel('True')

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    savefig('fig10_confusion_matrices.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Summary: top-line results across all datasets
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(data):
    wdd, wesad_cd, dann = get_cross_dataset()

    # Collect one best result per scenario
    entries = []

    # WESAD best model (wrist)
    best_wesad_f1, best_wesad = 0, None
    for stem in data:
        if not stem.startswith('wesad_'): continue
        for r in data[stem]:
            if r and 'wrist' in r.get('label','').lower():
                if r.get('f1', 0) > best_wesad_f1:
                    best_wesad_f1 = r['f1']
                    best_wesad = r
    if best_wesad:
        entries.append({'scenario': 'WESAD LOSO\n(best wrist)',
                        'f1': best_wesad['f1'], 'auc': best_wesad.get('auc', float('nan'))})

    # Anxolotl LOSO best
    if 'anxolotl_loso' in data:
        best_anx = max((r for r in data['anxolotl_loso'] if r and r.get('f1')),
                        key=lambda r: r['f1'], default=None)
        if best_anx:
            entries.append({'scenario': 'Anxolotl LOSO\n(best)',
                            'f1': best_anx['f1'], 'auc': best_anx.get('auc', float('nan'))})

    # Cross-dataset best
    all_cd = wdd + wesad_cd + dann
    if all_cd:
        best_cd = max(all_cd, key=lambda r: r['f1'])
        entries.append({'scenario': 'Cross-dataset\n(best)',
                        'f1': best_cd['f1'], 'auc': best_cd.get('auc', float('nan'))})

    if len(entries) < 2:
        print('  [SKIP] not enough data for summary'); return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[0], PALETTE[2], PALETTE[4]][:len(entries)]
    x = np.arange(len(entries))
    bars_f1 = ax.bar(x - 0.2, [e['f1']  for e in entries], 0.35,
                     label='Macro F1', color=colors, alpha=0.85)
    bars_auc = ax.bar(x + 0.2, [e.get('auc', float('nan')) for e in entries], 0.35,
                      label='AUC',      color=colors, alpha=0.5, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels([e['scenario'] for e in entries])
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.1)
    ax.set_title('Summary: Best Results by Evaluation Scenario')
    ax.legend(framealpha=0.7)
    for bar in list(bars_f1) + list(bars_auc):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    savefig('fig11_summary.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE E — Model ranking per dataset
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_ranking(data):
    """Horizontal bar chart: all models ranked by F1, one panel per dataset."""
    configs = [
        ('WESAD (wrist)',     [k for k in data if k.startswith('wesad_')], 'wrist'),
        ('WDD (60 common)',   ['wdd_classical'],  '60 common'),
        ('Anxolotl (40 common)', ['anxolotl_loso'], '40 common'),
    ]
    panels = []
    for title, stems, key in configs:
        m = collect_by_model(data, stems, key) if 'collect_by_model' in dir() else {}
        panels.append((title, m))

    # build panels after helpers defined — defer to after function definitions
    _ranking_configs = configs
    if not any(panels):
        print('  [SKIP] no data'); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('Model Ranking by Macro F1 — Per Dataset', fontsize=14, fontweight='bold')

    for ax, (ds_title, stems, key) in zip(axes, _ranking_configs):
        m = collect_by_model(data, stems, key)
        if not m:
            ax.set_title(f'{ds_title}\n(no data)'); continue
        names = sorted(m, key=lambda n: m[n].get('f1', 0) or 0)
        f1s   = [m[n].get('f1', float('nan')) or 0 for n in names]
        aucs  = [m[n].get('auc', float('nan')) or 0 for n in names]
        y = np.arange(len(names))
        ax.barh(y - 0.2, f1s,  0.35, label='Macro F1', color=PALETTE[0], alpha=0.85)
        ax.barh(y + 0.2, aucs, 0.35, label='AUC',      color=PALETTE[1], alpha=0.60)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Score'); ax.set_title(ds_title)
        ax.legend(fontsize=8, loc='lower right')
        for yi, v in enumerate(f1s):
            if not np.isnan(v): ax.text(v + 0.005, yi - 0.2, f'{v:.3f}', va='center', fontsize=7)

    plt.tight_layout()
    savefig('figE_model_ranking.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE F — F1 vs AUC scatter (all models × all datasets × feature sets)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_f1_vs_auc(data):
    ds_map = {
        'wesad': ('WESAD', PALETTE[0]),
        'wdd':   ('WDD',   PALETTE[2]),
        'anxolotl': ('Anxolotl', PALETTE[4]),
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0,1],[0,1],'k--', lw=0.6, alpha=0.4)

    for stem, records in data.items():
        ds_key = next((k for k in ds_map if stem.startswith(k)), None)
        if ds_key is None: continue
        ds_label, c = ds_map[ds_key]
        for r in records:
            if r is None: continue
            f1 = r.get('f1'); auc = r.get('auc')
            if f1 is None or auc is None: continue
            marker = 'o' if 'wrist' in r.get('label','').lower() or '40 common' in r.get('label','') or '60 common' in r.get('label','') else 's'
            ax.scatter(auc, f1, c=[c], marker=marker, s=60, alpha=0.75, edgecolors='white', lw=0.5)
            ax.annotate(model_base(r['label']), (auc, f1),
                        textcoords='offset points', xytext=(4, 2), fontsize=6, color=c, alpha=0.8)

    # legend patches
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in [(v[0], v[1]) for v in ds_map.values()]]
    handles += [plt.Line2D([0],[0], marker='o', color='grey', label='restricted feat', linestyle='None'),
                plt.Line2D([0],[0], marker='s', color='grey', label='all feat', linestyle='None')]
    ax.legend(handles=handles, fontsize=9, framealpha=0.8)
    ax.set_xlabel('AUC-ROC'); ax.set_ylabel('Macro F1')
    ax.set_title('F1 vs AUC — All Models & Datasets')
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    plt.tight_layout()
    savefig('figF_f1_vs_auc_scatter.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE G — Class-level metrics: Baseline vs Stress (precision & recall)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_class_level_metrics(data):
    """For each dataset, bar chart showing per-class precision and recall."""
    configs = [
        ('WESAD (wrist)',        [k for k in data if k.startswith('wesad_')], 'wrist'),
        ('WDD (60 common)',      ['wdd_classical'],  '60 common'),
        ('Anxolotl (40 common)', ['anxolotl_loso'],  '40 common'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Per-Class Precision & Recall — Baseline vs Stress', fontsize=14, fontweight='bold')

    for ax, (ds_title, stems, key) in zip(axes, configs):
        m = collect_by_model(data, stems, key)
        if not m:
            ax.set_title(f'{ds_title}\n(no data)'); continue

        names = sorted(m, key=lambda n: m[n].get('f1', 0) or 0, reverse=True)
        prec0 = [m[n].get('precision_0', float('nan')) for n in names]
        rec0  = [m[n].get('recall_0',    float('nan')) for n in names]
        prec1 = [m[n].get('precision_1', float('nan')) for n in names]
        rec1  = [m[n].get('recall_1',    float('nan')) for n in names]

        x = np.arange(len(names)); w = 0.2
        ax.bar(x - 1.5*w, prec0, w, label='Prec Baseline', color=PALETTE[0], alpha=0.85)
        ax.bar(x - 0.5*w, rec0,  w, label='Rec Baseline',  color=PALETTE[0], alpha=0.45)
        ax.bar(x + 0.5*w, prec1, w, label='Prec Stress',   color=PALETTE[3], alpha=0.85)
        ax.bar(x + 1.5*w, rec1,  w, label='Rec Stress',    color=PALETTE[3], alpha=0.45)

        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel('Score'); ax.set_title(ds_title)
        ax.legend(fontsize=8, framealpha=0.7)

    plt.tight_layout()
    savefig('figG_class_level_metrics.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE H — Per-fold F1 violin: variability across datasets for shared models
# ═══════════════════════════════════════════════════════════════════════════════

def plot_violin_across_datasets(data):
    """Violin plot: fold F1 distribution for shared models, colored by dataset."""
    configs = [
        ('WESAD',    [k for k in data if k.startswith('wesad_')], 'wrist',     PALETTE[0]),
        ('WDD',      ['wdd_classical'],  '60 common',  PALETTE[2]),
        ('Anxolotl', ['anxolotl_loso'],  '40 common',  PALETTE[4]),
    ]
    # collect fold_f1s per model per dataset
    ds_data = {}
    for ds_name, stems, key, c in configs:
        m = collect_by_model(data, stems, key)
        ds_data[ds_name] = {'models': m, 'color': c}

    shared = sorted(set.intersection(*[set(v['models'].keys()) for v in ds_data.values()
                                        if v['models']]) or set())
    if not shared:
        print('  [SKIP] no shared models for violin'); return

    fig, axes = plt.subplots(1, len(shared), figsize=(max(12, len(shared)*3), 6), sharey=True)
    if len(shared) == 1: axes = [axes]
    fig.suptitle('Per-Fold F1 Distribution — Shared Models', fontsize=14, fontweight='bold')

    for ax, model in zip(axes, shared):
        plot_data, colors, labels = [], [], []
        for ds_name, v in ds_data.items():
            r = v['models'].get(model)
            if r and r.get('fold_f1s'):
                plot_data.append(r['fold_f1s'])
                colors.append(v['color'])
                labels.append(ds_name)

        if not plot_data: ax.set_title(model); continue
        parts = ax.violinplot(plot_data, showmedians=True, showextrema=True)
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c); pc.set_alpha(0.65)
        for part in ['cmedians','cmins','cmaxes','cbars']:
            if part in parts: parts[part].set_color('black')
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylim(0, 1.05); ax.set_title(model)
        if ax == axes[0]: ax.set_ylabel('Macro F1 per Fold')

    plt.tight_layout()
    savefig('figH_violin_across_datasets.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I — Precision-Recall curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(data):
    """PR curves for all models, one panel per dataset (restricted features)."""
    configs = [
        ('WESAD (wrist)',        [k for k in data if k.startswith('wesad_')], 'wrist'),
        ('WDD (60 common)',      ['wdd_classical'],  '60 common'),
        ('Anxolotl (40 common)', ['anxolotl_loso'],  '40 common'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Precision-Recall Curves (Stress Class)', fontsize=14, fontweight='bold')

    for ax, (ds_title, stems, key) in zip(axes, configs):
        plotted = False
        m = collect_by_model(data, stems, key)
        colors_it = iter(plt.cm.tab10(np.linspace(0, 1, max(len(m), 1))))
        for model_name, r in sorted(m.items(), key=lambda x: x[1].get('ap', 0) or 0, reverse=True):
            pre = r.get('precision'); rec = r.get('recall')
            if not pre or not rec: continue
            ap = r.get('ap', float('nan'))
            c = next(colors_it)
            lbl = f'{model_name}  (AP={ap:.3f})' if not np.isnan(ap) else model_name
            ax.plot(rec, pre, lw=1.8, color=c, label=lbl)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, 'no PR data\n(re-run models)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='grey')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(ds_title)
        ax.legend(loc='lower left', fontsize=7, framealpha=0.7)

    plt.tight_layout()
    savefig('figI_precision_recall_curves.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE J — F1@0.5 vs F1@opt: gain from threshold optimisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_threshold_gain(data):
    """Bar chart showing F1@0.5 and F1@opt side by side per model × dataset."""
    configs = [
        ('WESAD',    [k for k in data if k.startswith('wesad_')], 'wrist'),
        ('WDD',      ['wdd_classical'],  '60 common'),
        ('Anxolotl', ['anxolotl_loso'],  '40 common'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('F1@0.5 vs F1@Optimal Threshold — Gain from Threshold Tuning',
                 fontsize=14, fontweight='bold')

    for ax, (ds_name, stems, key) in zip(axes, configs):
        m = collect_by_model(data, stems, key)
        names = sorted(m, key=lambda n: m[n].get('f1', 0) or 0, reverse=True)
        f1_default = [m[n].get('f1',    float('nan')) for n in names]
        f1_opt     = [m[n].get('f1opt', float('nan')) for n in names]
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, f1_default, w, label='F1@0.5',   color=PALETTE[0], alpha=0.85)
        ax.bar(x + w/2, f1_opt,     w, label='F1@opt',   color=PALETTE[1], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel('Macro F1'); ax.set_title(ds_name)
        ax.legend(fontsize=9)
        for xi, (v0, vo) in enumerate(zip(f1_default, f1_opt)):
            if not np.isnan(v0) and not np.isnan(vo) and vo > v0 + 0.005:
                ax.annotate(f'+{vo-v0:.3f}', (xi, max(v0, vo) + 0.01),
                            ha='center', fontsize=7, color='green')

    plt.tight_layout()
    savefig('figJ_threshold_gain.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE K — Per-subject F1 heatmap (models × subjects)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_subject_heatmap(data):
    """Heatmap: rows=models, cols=subject fold index, value=F1. One per dataset."""
    configs = [
        ('WESAD (wrist)',        [k for k in data if k.startswith('wesad_')], 'wrist'),
        ('WDD (60 common)',      ['wdd_classical'],  '60 common'),
        ('Anxolotl (40 common)', ['anxolotl_loso'],  '40 common'),
    ]
    import pandas as pd
    for ds_title, stems, key in configs:
        m = collect_by_model(data, stems, key)
        rows = {name: r['fold_f1s'] for name, r in m.items()
                if r and r.get('fold_f1s')}
        if not rows: continue

        n_folds = max(len(v) for v in rows.values())
        df = pd.DataFrame({name: vals + [float('nan')] * (n_folds - len(vals))
                           for name, vals in rows.items()}).T
        df.columns = [f'S{i+1:02d}' for i in range(n_folds)]

        fig, ax = plt.subplots(figsize=(max(12, n_folds * 0.7), max(4, len(rows) * 0.5)))
        sns.heatmap(df.astype(float), annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax, linewidths=0.3,
                    cbar_kws={'shrink': 0.6, 'label': 'Macro F1'})
        ax.set_title(f'Per-Subject F1 — {ds_title}', pad=10)
        ax.set_xlabel('Subject (fold)'); ax.set_ylabel('Model')
        plt.tight_layout()
        fname = f'figK_subject_heatmap_{ds_title.split()[0].lower()}.png'
        savefig(fname)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE L — ROC curves overlay: best model per dataset on same plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_best_per_dataset(data):
    """Single ROC plot with the best model from each dataset."""
    wdd_cd, wesad_cd, dann_cd = get_cross_dataset()

    configs = [
        ('WESAD (wrist)',        [k for k in data if k.startswith('wesad_')], 'wrist',      PALETTE[0]),
        ('WDD (60 common)',      ['wdd_classical'],  '60 common',  PALETTE[2]),
        ('Anxolotl (40 common)', ['anxolotl_loso'],  '40 common',  PALETTE[4]),
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1], 'k--', lw=0.8, label='Random (AUC=0.500)')

    for ds_title, stems, key, c in configs:
        m = collect_by_model(data, stems, key)
        best_r = max((r for r in m.values() if r and r.get('fpr') and r.get('tpr')),
                     key=lambda r: r.get('auc', 0) or 0, default=None)
        if best_r is None: continue
        fpr, tpr = np.array(best_r['fpr']), np.array(best_r['tpr'])
        auc = best_r.get('auc', float('nan'))
        name = model_base(best_r['label'])
        ax.plot(fpr, tpr, lw=2.2, color=c,
                label=f'{ds_title} — {name}  (AUC={auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Best ROC per Dataset (Restricted Features)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    savefig('figL_roc_best_per_dataset.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE M — Metrics radar chart per dataset (best model)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar(data):
    """Spider/radar chart: F1, AUC, F1@opt, BalAcc, MCC for best model per dataset."""
    import matplotlib.patches as mpatches
    configs = [
        ('WESAD (wrist)',        [k for k in data if k.startswith('wesad_')], 'wrist',  PALETTE[0]),
        ('WDD (60 common)',      ['wdd_classical'],  '60 common',  PALETTE[2]),
        ('Anxolotl (40 common)', ['anxolotl_loso'],  '40 common',  PALETTE[4]),
    ]
    metrics = ['f1', 'auc', 'f1opt', 'ba', 'mcc']
    labels  = ['Macro F1', 'AUC', 'F1@opt', 'Bal.Acc', 'MCC']
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], size=8)

    handles = []
    for ds_title, stems, key, c in configs:
        m = collect_by_model(data, stems, key)
        best_r = max((r for r in m.values() if r and r.get('f1')),
                     key=lambda r: r.get('f1', 0) or 0, default=None)
        if best_r is None: continue
        vals = [max(0, min(1, best_r.get(k, 0) or 0)) for k in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', lw=2, color=c, markersize=5)
        ax.fill(angles, vals, alpha=0.12, color=c)
        name = model_base(best_r['label'])
        handles.append(mpatches.Patch(color=c, label=f'{ds_title} — {name}'))

    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title('Best Model per Dataset — Metrics Radar', pad=20, fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig('figM_radar_best_per_dataset.png')


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS for cross-dataset model matching
# ═══════════════════════════════════════════════════════════════════════════════

# Maps label prefix → canonical model name
MODEL_ALIASES = {
    'LogReg': 'LogReg', 'SVM': 'SVM',
    'RF': 'RF', 'ET': 'ET',
    'Ensemble RF+ET': 'Ens RF+ET', 'Ensemble full': 'Ens full',
    'Ensemble': 'Ensemble',
    'DANN': 'DANN', 'AutoGluon': 'AutoGluon',
    '1D-CNN': '1D-CNN', 'CNN-GRU': 'CNN-GRU',
    'RNN (BiGRU)': 'RNN', 'Transformer': 'Transformer',
    'Mamba (SSM)': 'Mamba',
}

def model_base(label):
    """Extract canonical model name from a result label."""
    label = label.strip()
    for alias, canonical in sorted(MODEL_ALIASES.items(), key=lambda x: -len(x[0])):
        if label.startswith(alias):
            return canonical
    return label.split()[0]


def collect_by_model(data, stems, feat_key):
    """
    Returns {model_name: record} for given stems and feature key substring.
    feat_key: substring that must appear in label (e.g. 'wrist', '60 common', '40 common').
    """
    out = {}
    for stem in stems:
        for r in data.get(stem, []):
            if r and feat_key in r.get('label', ''):
                name = model_base(r['label'])
                if name not in out or (r.get('f1') or 0) > (out[name].get('f1') or 0):
                    out[name] = r
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Same model, three datasets (compatible feature sets)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_models_across_datasets(data):
    """
    For each model that appears in ≥2 datasets, grouped bar:
      X = model name
      Groups = WESAD (wrist) / WDD (60 common) / Anxolotl (40 common)
    """
    wesad_stems = [k for k in data if k.startswith('wesad_')]
    wesad = collect_by_model(data, wesad_stems,   'wrist')
    wdd   = collect_by_model(data, ['wdd_classical'], '60 common')
    anx   = collect_by_model(data, ['anxolotl_loso'], '40 common')

    # keep models present in at least 2 datasets
    all_models = sorted(set(wesad) | set(wdd) | set(anx))
    models = [m for m in all_models
              if sum([m in wesad, m in wdd, m in anx]) >= 2]
    if not models:
        print('  [SKIP] no common models across datasets'); return

    def vals(d, key='f1'):
        return [d.get(m, {}).get(key, float('nan')) for m in models]

    f1_w = vals(wesad); f1_d = vals(wdd); f1_a = vals(anx)
    auc_w = vals(wesad, 'auc'); auc_d = vals(wdd, 'auc'); auc_a = vals(anx, 'auc')

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(models) * 1.8), 6))
    fig.suptitle('Same Model — WESAD vs WDD vs Anxolotl\n'
                 '(WESAD wrist / WDD 60 common feat / Anxolotl 40 common feat)',
                 fontsize=13, fontweight='bold')

    colors = [PALETTE[0], PALETTE[2], PALETTE[4]]
    datasets = ['WESAD (wrist)', 'WDD (60 common)', 'Anxolotl (40 common)']
    x = np.arange(len(models)); w = 0.25

    for ax, f1_sets, title in [
        (axes[0], [f1_w, f1_d, f1_a], 'Macro F1'),
        (axes[1], [auc_w, auc_d, auc_a], 'AUC-ROC'),
    ]:
        for i, (vals_ds, ds_name, c) in enumerate(zip(f1_sets, datasets, colors)):
            xi = x + (i - 1) * w
            bars = ax.bar(xi, vals_ds, w, label=ds_name, color=c, alpha=0.85)
            for xj, v in zip(xi, vals_ds):
                if not np.isnan(v):
                    ax.text(xj, v + 0.005, f'{v:.2f}', ha='center', va='bottom',
                            fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha='right')
        ax.set_ylabel(title); ax.set_ylim(0, 1.12); ax.set_title(title)
        ax.legend(framealpha=0.8, fontsize=9)

    plt.tight_layout()
    savefig('figA_models_across_datasets.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Wrist vs All: per-dataset side-by-side
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wrist_vs_all_per_dataset(data):
    """
    Three sub-plots (one per dataset). Each: X=model, groups=wrist/all.
    WESAD: wrist 56 / all 137
    WDD:   60 common / 90 all
    Anxolotl: 40 common / 60 all
    """
    configs = [
        ('WESAD',     [k for k in data if k.startswith('wesad_')],
         'wrist', 'all', 'Wrist (56)', 'All (137)'),
        ('WDD',       ['wdd_classical'],
         '60 common', '90 all', '60 common feat', '90 all feat'),
        ('Anxolotl',  ['anxolotl_loso'],
         '40 common', '60 all', '40 common feat', '60 all feat'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Wrist vs All Features — Per Dataset', fontsize=14, fontweight='bold')

    for ax, (ds_name, stems, key_a, key_b, lbl_a, lbl_b) in zip(axes, configs):
        m_a = collect_by_model(data, stems, key_a)
        m_b = collect_by_model(data, stems, key_b)
        models = sorted(set(m_a) | set(m_b))
        if not models:
            ax.set_title(f'{ds_name} — no data'); continue

        f1_a = [m_a.get(m, {}).get('f1', float('nan')) for m in models]
        f1_b = [m_b.get(m, {}).get('f1', float('nan')) for m in models]
        std_a = [m_a.get(m, {}).get('f1_std') or 0 for m in models]
        std_b = [m_b.get(m, {}).get('f1_std') or 0 for m in models]

        grouped_bar(ax, models, f1_a, f1_b,
                    label_a=lbl_a, label_b=lbl_b,
                    errs_a=std_a, errs_b=std_b, ylabel='Macro F1')
        ax.set_title(f'{ds_name}')

    plt.tight_layout()
    savefig('figB_wrist_vs_all_per_dataset.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Pipeline progression: WESAD → WDD → Cross-dataset → Anxolotl
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_progression(data):
    """
    For the shared classical models, line chart showing F1 across the 4 stages:
    WESAD LOSO → WDD LOSO → Cross-dataset → Anxolotl LOSO
    Each line = one model. Shows domain shift degradation and recovery.
    """
    wdd_cd, wesad_cd, dann_cd = get_cross_dataset()

    wesad_m = collect_by_model(data, [k for k in data if k.startswith('wesad_')], 'wrist')
    wdd_m   = collect_by_model(data, ['wdd_classical'], '60 common')
    anx_m   = collect_by_model(data, ['anxolotl_loso'], '40 common')

    # best cross-dataset result (RF+ET WDD→Anxolotl)
    best_cd_f1 = max((r['f1'] for r in wdd_cd if r.get('f1')), default=float('nan'))

    # models present in at least WESAD + one other
    shared = [m for m in sorted(set(wesad_m) & (set(wdd_m) | set(anx_m)))
              if m not in ('1D-CNN','CNN-GRU','RNN','Transformer','Mamba')]
    if not shared:
        print('  [SKIP] no shared models for progression plot'); return

    stages = ['WESAD\n(LOSO)', 'WDD\n(LOSO)', 'Cross-dataset\n(WDD→Anxolotl)', 'Anxolotl\n(LOSO)']
    stage_x = [0, 1, 2, 3]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(shared)))

    for model, c in zip(shared, cmap):
        f1_wesad = wesad_m.get(model, {}).get('f1', float('nan'))
        f1_wdd   = wdd_m.get(model, {}).get('f1', float('nan'))
        f1_anx   = anx_m.get(model, {}).get('f1', float('nan'))

        # cross-dataset: use the best known or from parsed txt if RF/ET
        if model in ('RF', 'ET', 'Ensemble RF+ET'):
            f1_cd = best_cd_f1
        else:
            f1_cd = float('nan')

        ys = [f1_wesad, f1_wdd, f1_cd, f1_anx]
        valid = [(xi, y) for xi, y in zip(stage_x, ys) if not np.isnan(y)]
        if len(valid) < 2: continue
        xs_v, ys_v = zip(*valid)
        ax.plot(xs_v, ys_v, 'o-', color=c, lw=2, markersize=7, label=model)
        for xi, yi in valid:
            ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color=c)

    # shade cross-dataset gap region
    ax.axvspan(1.5, 2.5, alpha=0.07, color='red', label='_nolegend_')
    ax.text(2, 0.05, 'domain\nshift', ha='center', va='bottom', fontsize=9,
            color='red', alpha=0.7)

    ax.set_xticks(stage_x); ax.set_xticklabels(stages)
    ax.set_ylabel('Macro F1'); ax.set_ylim(0, 1.08)
    ax.set_title('Performance Progression Through the Pipeline\n'
                 '(compatible feature set: wrist / 60-common / 40-common)')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8)
    plt.tight_layout()
    savefig('figC_pipeline_progression.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Δ F1 wrist vs all: heatmap (model × dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wrist_vs_all_heatmap(data):
    """
    Rows = models, Cols = datasets
    Cell value = F1_all − F1_wrist  (positive = all features help)
    """
    import pandas as pd
    configs = [
        ('WESAD',    [k for k in data if k.startswith('wesad_')], 'wrist', 'all'),
        ('WDD',      ['wdd_classical'],  '60 common', '90 all'),
        ('Anxolotl', ['anxolotl_loso'],  '40 common', '60 all'),
    ]
    rows = {}
    for ds_name, stems, key_w, key_a in configs:
        m_w = collect_by_model(data, stems, key_w)
        m_a = collect_by_model(data, stems, key_a)
        for model in sorted(set(m_w) | set(m_a)):
            fw = m_w.get(model, {}).get('f1', float('nan'))
            fa = m_a.get(model, {}).get('f1', float('nan'))
            rows.setdefault(model, {})[ds_name] = fa - fw

    if not rows:
        print('  [SKIP] no data for wrist/all heatmap'); return

    df = pd.DataFrame(rows).T.reindex(columns=['WESAD', 'WDD', 'Anxolotl'])
    df = df.dropna(how='all').astype(float)
    if df.empty:
        print('  [SKIP] empty dataframe'); return

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.45)))
    sns.heatmap(df, annot=True, fmt='+.3f', cmap='RdYlGn', center=0,
                vmin=-0.15, vmax=0.15, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'ΔF1 (All − Wrist)'})
    ax.set_title('Impact of Adding Extra Features\n(ΔF1 = All Features − Wrist Only)',
                 pad=10)
    ax.set_xlabel('Dataset'); ax.set_ylabel('Model')
    plt.tight_layout()
    savefig('figD_delta_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Loading results from {RES_DIR} ...')
    data = load_json_results()
    if not data:
        print('No JSON result files found. Run run_all.sh first.')
    else:
        print(f'  Found: {list(data.keys())}')

    print(f'\nGenerating figures → {FIG_DIR}')

    print('\n[1] WESAD comparison (wrist vs all)')
    plot_wesad_comparison(data)

    print('\n[2] Wrist vs All Δ F1')
    plot_wrist_vs_all_delta(data)

    print('\n[3] WDD LOSO comparison')
    plot_dataset_comparison(data, 'wdd_classical', 'WDD LOSO', '60 common', '90 all',
                            'fig3_wdd_comparison.png')

    print('\n[4] Anxolotl LOSO')
    plot_anxolotl(data)

    print('\n[5] Cross-dataset comparison')
    plot_cross_dataset(data)

    print('\n[6] ROC curves — WESAD wrist')
    plot_roc_curves(data, feature_key='wrist',
                    title_suffix='Wrist Features', fname='fig6_roc_wesad_wrist.png')

    print('\n[7] ROC curves — WESAD all')
    plot_roc_curves(data, feature_key=' all',
                    title_suffix='All Features', fname='fig7_roc_wesad_all.png')

    print('\n[8] ROC curves — WDD')
    plot_roc_curves(data, feature_key='wdd',
                    title_suffix='WDD LOSO', fname='fig8_roc_wdd.png', stem='wdd_classical')

    print('\n[9] ROC curves — Anxolotl')
    plot_roc_anxolotl(data)

    print('\n[10] Metrics heatmap')
    plot_metrics_heatmap(data)

    print('\n[11] Fold variability')
    plot_fold_variability(data)

    print('\n[12] Confusion matrices')
    plot_confusion_matrices(data)

    print('\n[13] Summary overview')
    plot_summary(data)

    # ── Cross-dataset comparison plots ────────────────────────────────────────
    print('\n[A] Same model across datasets')
    plot_models_across_datasets(data)

    print('\n[B] Wrist vs All — per dataset side-by-side')
    plot_wrist_vs_all_per_dataset(data)

    print('\n[C] Pipeline progression WESAD→WDD→Cross-dataset→Anxolotl')
    plot_pipeline_progression(data)

    print('\n[D] ΔF1 heatmap (All − Wrist)')
    plot_wrist_vs_all_heatmap(data)

    # ── Additional plots ──────────────────────────────────────────────────────
    print('\n[E] Model ranking per dataset')
    plot_model_ranking(data)

    print('\n[F] F1 vs AUC scatter')
    plot_f1_vs_auc(data)

    print('\n[G] Per-class precision & recall')
    plot_class_level_metrics(data)

    print('\n[H] Per-fold F1 violin across datasets')
    plot_violin_across_datasets(data)

    print('\n[I] Precision-Recall curves')
    plot_pr_curves(data)

    print('\n[J] Threshold gain (F1@0.5 vs F1@opt)')
    plot_threshold_gain(data)

    print('\n[K] Per-subject F1 heatmap')
    plot_subject_heatmap(data)

    print('\n[L] Best ROC per dataset overlay')
    plot_roc_best_per_dataset(data)

    print('\n[M] Radar chart best model per dataset')
    plot_radar(data)

    print(f'\nDone. All figures in {FIG_DIR}')
