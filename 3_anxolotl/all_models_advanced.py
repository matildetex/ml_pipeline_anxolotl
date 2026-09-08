"""
Advanced model comparison with:
  1. Threshold optimization (per-fold F1-optimal threshold from train set)
  2. LightGBM with hyperparameter variations
  3. Subject calibration (baseline windows of test subject added to train)
  4. Watch + ECG features (17 features including nonlinear HRV)

Feature sets tested:
  A. Watch only               — 60 feat, fixed threshold
  B. Watch only               — 60 feat, optimal threshold
  C. Watch + ECG              — 77 feat, optimal threshold
  D. Watch + ECG + calibration — 77 feat, optimal threshold, subject calibration

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
from demographics_analysis import baseline_relative_scale

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

# ── Classifiers ───────────────────────────────────────────────────────────────

def make_classifiers():
    """Original 8 + best new models + LightGBM variants."""
    return {
        # Original Heo et al.
        'DT':         DecisionTreeClassifier(random_state=0),
        'RF':         RandomForestClassifier(max_depth=4, random_state=0),
        'AB':         AdaBoostClassifier(random_state=0),
        'KN':         KNeighborsClassifier(n_neighbors=9),
        'LDA':        LinearDiscriminantAnalysis(),
        'SVM':        SVC(probability=True),
        'GB':         GradientBoostingClassifier(random_state=0),
        # Best from previous runs
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                            class_weight='balanced', random_state=42),
        'ET-1000':    ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                                            min_samples_leaf=1, class_weight='balanced',
                                            random_state=42, n_jobs=-1),
        'RF-1000':    RandomForestClassifier(n_estimators=1000, max_features=0.5,
                                              min_samples_leaf=1, class_weight='balanced',
                                              random_state=42, n_jobs=-1),
        'HGB':        HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                                      learning_rate=0.1, random_state=42),
        'Ensemble':   VotingClassifier(voting='soft', n_jobs=-1, estimators=[
                          ('et',  ExtraTreesClassifier(n_estimators=300, max_features=0.5,
                                                        class_weight='balanced', random_state=42, n_jobs=-1)),
                          ('rf',  RandomForestClassifier(n_estimators=300, max_features=0.5,
                                                          class_weight='balanced', random_state=42, n_jobs=-1)),
                          ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                                   learning_rate=0.1, random_state=42)),
                      ]),
        # LightGBM variants
        'LGBM-base':  lgb.LGBMClassifier(n_estimators=500, num_leaves=31,
                                          learning_rate=0.05, class_weight='balanced',
                                          random_state=42, n_jobs=-1, verbose=-1),
        'LGBM-deep':  lgb.LGBMClassifier(n_estimators=1000, num_leaves=63,
                                          learning_rate=0.03, min_child_samples=5,
                                          subsample=0.8, colsample_bytree=0.8,
                                          class_weight='balanced', random_state=42,
                                          n_jobs=-1, verbose=-1),
        'LGBM-dart':  lgb.LGBMClassifier(n_estimators=500, num_leaves=31,
                                          boosting_type='dart', drop_rate=0.1,
                                          learning_rate=0.1, class_weight='balanced',
                                          random_state=42, n_jobs=-1, verbose=-1),
    }

# ── Threshold optimization ────────────────────────────────────────────────────

def optimal_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on the given set."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, y_prob >= t, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ── LOSO-CV core ─────────────────────────────────────────────────────────────

def run_loso(X, y, groups, label='', use_opt_thresh=False, use_calibration=False):
    clfs     = make_classifiers()
    subjects = np.unique(groups)
    results  = {name: {'f1': [], 'auc': []} for name in clfs}
    imp = SimpleImputer(strategy='median')

    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        # BL-norm
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_tr, y_tr, groups[train_mask]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_te, y_te, groups[test_mask]))

        # Impute NaN (ECG windows where BIOPAC was unavailable)
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)

        # Subject calibration: use FIRST HALF of test subject's baseline windows
        # to personalise the model; evaluate on second half + all stress windows.
        # Avoids data leakage: calibration windows are excluded from evaluation.
        if use_calibration:
            base_idx = np.where(y_te == 0)[0]
            if len(base_idx) >= 4:
                n_cal  = len(base_idx) // 2
                cal_idx   = base_idx[:n_cal]       # first half → calibration
                eval_idx  = np.concatenate([
                    base_idx[n_cal:],              # second half of baseline
                    np.where(y_te == 1)[0],        # all stress windows
                ])
                X_tr_n = np.vstack([X_tr_n, X_te_n[cal_idx]])
                y_tr   = np.concatenate([y_tr, y_te[cal_idx]])
                # Restrict test set to non-calibration windows
                X_te_n = X_te_n[eval_idx]
                y_te   = y_te[eval_idx]

        for name, clf in clfs.items():
            try:
                clf.fit(X_tr_n, y_tr)
                y_prob = clf.predict_proba(X_te_n)[:, 1]
                auc    = roc_auc_score(y_te, y_prob)

                if use_opt_thresh:
                    y_prob_tr = clf.predict_proba(X_tr_n)[:, 1]
                    thresh = optimal_threshold(y_tr, y_prob_tr)
                    y_pred = (y_prob >= thresh).astype(int)
                else:
                    y_pred = (y_prob >= 0.5).astype(int)
            except Exception:
                y_pred = np.zeros(len(y_te), dtype=int)
                auc    = float('nan')

            results[name]['f1'].append(
                f1_score(y_te, y_pred, average='binary', zero_division=0)
            )
            results[name]['auc'].append(auc)

    summary = {name: {'f1': np.mean(v['f1']), 'auc': np.nanmean(v['auc'])}
               for name, v in results.items()}

    print(f'\n{label}')
    print(f'  {"Model":<14} {"F1":>7}  {"AUC":>7}')
    print(f'  {"-"*32}')
    for name, m in sorted(summary.items(), key=lambda x: -x[1]['f1']):
        print(f'  {name:<14} {m["f1"]:>7.3f}  {m["auc"]:>7.3f}')
    return summary

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')
    with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
        meta = pickle.load(f)
    X_watch    = np.array(meta['X'], dtype=float)
    y_all      = np.array(meta['y'])
    groups_all = np.array(meta['groups'])

    with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl', 'rb') as f:
        ecg_data = pickle.load(f)
    X_ecg_raw = ecg_data['X_ecg'].astype(float)
    ecg_cols  = ecg_data['ecg_cols']

    # Clip physiological outliers
    X_ecg = X_ecg_raw.copy()
    rmssd_idx = ecg_cols.index('ecg_rmssd') if 'ecg_rmssd' in ecg_cols else 2
    std_idx   = ecg_cols.index('ecg_rr_std') if 'ecg_rr_std' in ecg_cols else 1
    lfhf_idx  = ecg_cols.index('ecg_lf_hf')  if 'ecg_lf_hf'  in ecg_cols else 7
    X_ecg[X_ecg[:, rmssd_idx] > 300, :] = np.nan
    X_ecg[X_ecg[:, std_idx]   > 300, :] = np.nan
    X_ecg[X_ecg[:, lfhf_idx]  > 50, lfhf_idx] = np.nan

    X_watch_ecg = np.hstack([X_watch, X_ecg])

    print(f'Watch: {X_watch.shape}')
    print(f'Watch+ECG: {X_watch_ecg.shape}  '
          f'(valid ECG: {(~np.isnan(X_ecg[:,0])).sum()}/{len(X_ecg)})')
    print(f'ECG cols ({len(ecg_cols)}): {ecg_cols}')

    print('\n' + '='*60)
    print('LOSO-CV — 4 configurations')
    print('='*60)

    # A: watch, fixed threshold (baseline)
    res_A = run_loso(X_watch,     y_all, groups_all,
                     '── A: Watch, thresh=0.5 ──',
                     use_opt_thresh=False, use_calibration=False)

    # B: watch, optimal threshold
    res_B = run_loso(X_watch,     y_all, groups_all,
                     '── B: Watch, optimal threshold ──',
                     use_opt_thresh=True,  use_calibration=False)

    # C: watch+ECG, optimal threshold
    res_C = run_loso(X_watch_ecg, y_all, groups_all,
                     '── C: Watch+ECG, optimal threshold ──',
                     use_opt_thresh=True,  use_calibration=False)

    # D: watch+ECG, optimal threshold + subject calibration
    res_D = run_loso(X_watch_ecg, y_all, groups_all,
                     '── D: Watch+ECG, optimal threshold + calibration ──',
                     use_opt_thresh=True,  use_calibration=True)

    # ── Results table ─────────────────────────────────────────────────────────
    clf_names = list(make_classifiers().keys())
    rows = []
    for name in clf_names:
        rows.append({
            'model':   name,
            'F1_A':    res_A[name]['f1'],
            'F1_B':    res_B[name]['f1'],
            'F1_C':    res_C[name]['f1'],
            'F1_D':    res_D[name]['f1'],
            'dB_A':    res_B[name]['f1'] - res_A[name]['f1'],
            'dC_A':    res_C[name]['f1'] - res_A[name]['f1'],
            'dD_A':    res_D[name]['f1'] - res_A[name]['f1'],
        })
    df_res = pd.DataFrame(rows).sort_values('F1_A', ascending=False)
    df_res.to_csv('results/all_models_advanced.csv', index=False)

    print('\n' + '='*75)
    print('SUMMARY — sorted by Watch-only F1 (A)')
    print('='*75)
    print(f'{"Model":<14} {"A(base)":>9} {"B(+thr)":>9} {"C(+ECG)":>9} {"D(+cal)":>9}'
          f'  {"ΔB":>7} {"ΔC":>7} {"ΔD":>7}')
    print('-'*75)
    for _, r in df_res.iterrows():
        print(f'{r["model"]:<14} {r["F1_A"]:>9.3f} {r["F1_B"]:>9.3f}'
              f' {r["F1_C"]:>9.3f} {r["F1_D"]:>9.3f}'
              f'  {r["dB_A"]:>+7.3f} {r["dC_A"]:>+7.3f} {r["dD_A"]:>+7.3f}')

    best_D = df_res.sort_values('F1_D', ascending=False).iloc[0]
    print(f'\nBest overall (D): {best_D["model"]}  F1={best_D["F1_D"]:.3f}')
    print(f'Gain over watch-only (A): {best_D["F1_D"] - best_D["F1_A"]:+.3f}')

    # ── Figure: progression bar chart ─────────────────────────────────────────
    top_models = df_res.sort_values('F1_A', ascending=False).head(10)['model'].tolist()
    x = np.arange(len(top_models))
    w = 0.20
    C = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']
    labels_bar = ['A: Watch, thr=0.5', 'B: Watch, opt-thr',
                  'C: +ECG, opt-thr',  'D: +ECG, opt-thr, calib']

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (res, col, lbl) in enumerate(zip([res_A, res_B, res_C, res_D], C, labels_bar)):
        vals = [res[m]['f1'] for m in top_models]
        ax.bar(x + (i - 1.5)*w, vals, w, label=lbl, color=col, alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(top_models, rotation=25, ha='right')
    ax.set_ylabel('Binary F1 (stress class)')
    ax.set_ylim(0, 1.15)
    ax.set_title('Progression: threshold opt. → ECG features → subject calibration\n'
                 '(Anxolotl LOSO-CV, BL-norm, top-10 models)')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig16_advanced_progression.png', dpi=150, bbox_inches='tight')
    plt.close(); print('\nSaved fig16_advanced_progression.png')

    # ── Heatmap: all configs × top models ────────────────────────────────────
    heat = pd.DataFrame({
        'A: Watch\nthr=0.5':    {m: res_A[m]['f1'] for m in top_models},
        'B: Watch\nopt-thr':    {m: res_B[m]['f1'] for m in top_models},
        'C: +ECG\nopt-thr':     {m: res_C[m]['f1'] for m in top_models},
        'D: +ECG+calib\nopt-thr':{m: res_D[m]['f1'] for m in top_models},
    })
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.7, vmax=1.0, ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Binary F1'})
    ax.set_title('F1 heatmap — progressive improvements\n(Anxolotl LOSO-CV, BL-norm)', pad=10)
    plt.tight_layout()
    plt.savefig('figures/fig17_advanced_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Saved fig17_advanced_heatmap.png')

    print('\nSaved results/all_models_advanced.csv')
