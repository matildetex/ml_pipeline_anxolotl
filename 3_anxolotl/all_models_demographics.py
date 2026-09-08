"""
Full model comparison across 3 feature sets (watch-only, +demographics, +EADS-21).
Extends demographics_analysis.py with all available sklearn models.

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    VotingClassifier, HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from demographics_analysis import (
    load_demographics, baseline_relative_scale,
    SESSION_TO_FORM, EADS_COLS, DEPRESSION_IDX, ANXIETY_IDX, STRESS_IDX,
)

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})

# ── Classifiers ───────────────────────────────────────────────────────────────

def make_all_classifiers():
    from sklearn import tree, svm as sk_svm
    return {
        # ── Original Heo et al. classifiers (exact params, exact names) ──────
        'DT':         DecisionTreeClassifier(random_state=0),
        'RF':         RandomForestClassifier(max_depth=4, random_state=0),
        'AB':         AdaBoostClassifier(random_state=0),
        'KN':         KNeighborsClassifier(n_neighbors=9),
        'LDA':        LinearDiscriminantAnalysis(),
        'SVM':        SVC(probability=True),
        'GB':         GradientBoostingClassifier(random_state=0),
        # ── Original best model (exact params) ───────────────────────────────
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                            class_weight='balanced', random_state=42),
        # ── New / extended models ─────────────────────────────────────────────
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
        'LR-L1':      LogisticRegression(C=0.001, penalty='l1', solver='saga',
                                          max_iter=2000, random_state=42),
        'LR-L2':      LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
        'SVM-Bal':    SVC(probability=True, class_weight='balanced', random_state=42),
        'SVM-Poly':   SVC(C=0.1, kernel='poly', degree=3, gamma='auto',
                          class_weight='balanced', probability=True, random_state=42),
        'MLP':        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                     random_state=42, early_stopping=True),
    }

# ── LOSO-CV ───────────────────────────────────────────────────────────────────

def run_loso(X, y, groups, label=''):
    clfs = make_all_classifiers()
    subjects = np.unique(groups)
    results = {name: {'f1': [], 'auc': []} for name in clfs}

    imp = SimpleImputer(strategy='mean')

    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        X_train_n = np.nan_to_num(
            baseline_relative_scale(X_train, y_train, groups[train_mask])
        )
        X_test_n  = np.nan_to_num(
            baseline_relative_scale(X_test,  y_test,  groups[test_mask])
        )

        for name, clf in clfs.items():
            try:
                clf.fit(X_train_n, y_train)
                y_pred = clf.predict(X_test_n)
                y_prob = clf.predict_proba(X_test_n)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                y_pred = np.zeros(len(y_test), dtype=int)
                auc = float('nan')
            results[name]['f1'].append(
                f1_score(y_test, y_pred, average='binary', zero_division=0)
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
    XLSX = 'Data_tese.xlsx'

    # Load wearable data
    print('Loading wearable data...')
    with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
        meta = pickle.load(f)
    X_all      = np.array(meta['X'], dtype=float)
    y_all      = np.array(meta['y'])
    groups_all = np.array(meta['groups'])
    feat_cols  = meta['feature_cols']

    # Load demographics
    print('Loading demographics...')
    df_demo = load_demographics(XLSX)
    demo_cols  = ['age', 'gender', 'height', 'weight', 'bmi',
                  'respiratory', 'medication', 'exercises', 'exercise_freq',
                  'athlete', 'stimulant']
    eads_cols_ = ['eads_depression', 'eads_anxiety', 'eads_stress', 'eads_total', 'vas_stress']

    def expand(cols):
        rows = []
        for g in groups_all:
            rows.append(df_demo.loc[g, cols].values.astype(float)
                        if g in df_demo.index else np.zeros(len(cols)))
        return np.array(rows)

    X_watch      = X_all
    X_watch_demo = np.hstack([X_all, expand(demo_cols)])
    X_watch_eads = np.hstack([X_all, expand(eads_cols_)])

    print(f'Feature sets: watch={X_watch.shape[1]}, '
          f'+demo={X_watch_demo.shape[1]}, +eads={X_watch_eads.shape[1]}')
    print(f'Models: {len(make_all_classifiers())}')

    # Run LOSO-CV
    print('\n' + '='*60)
    print('LOSO-CV — BL-norm — all models')
    print('='*60)
    res_watch = run_loso(X_watch,      y_all, groups_all, '── Watch only ──')
    res_demo  = run_loso(X_watch_demo, y_all, groups_all, '── Watch + Demographics ──')
    res_eads  = run_loso(X_watch_eads, y_all, groups_all, '── Watch + EADS-21/VAS ──')

    # Build results table
    clf_names = list(make_all_classifiers().keys())
    rows = []
    for name in clf_names:
        rows.append({
            'model':         name,
            'F1_watch':      res_watch[name]['f1'],
            'F1_demo':       res_demo[name]['f1'],
            'F1_eads':       res_eads[name]['f1'],
            'AUC_watch':     res_watch[name]['auc'],
            'AUC_demo':      res_demo[name]['auc'],
            'AUC_eads':      res_eads[name]['auc'],
            'dF1_demo':      res_demo[name]['f1']  - res_watch[name]['f1'],
            'dF1_eads':      res_eads[name]['f1']  - res_watch[name]['f1'],
        })
    df_res = pd.DataFrame(rows).sort_values('F1_watch', ascending=False)
    df_res.to_csv('results/all_models_demographics.csv', index=False)

    print('\n' + '='*60)
    print('FINAL TABLE — sorted by Watch-only F1')
    print('='*60)
    print(f'{"Model":<14} {"F1 Watch":>9} {"F1 +Demo":>9} {"F1 +EADS":>9}'
          f'  {"Δ Demo":>7} {"Δ EADS":>7}')
    print('-'*60)
    for _, r in df_res.iterrows():
        print(f'{r["model"]:<14} {r["F1_watch"]:>9.3f} {r["F1_demo"]:>9.3f}'
              f' {r["F1_eads"]:>9.3f}  {r["dF1_demo"]:>+7.3f} {r["dF1_eads"]:>+7.3f}')

    # ── Figures ───────────────────────────────────────────────────────────────
    C_W = '#4C72B0'; C_D = '#55A868'; C_E = '#DD8452'
    models_sorted = df_res['model'].tolist()
    x = np.arange(len(models_sorted))
    w = 0.26

    # Fig A — all models, 3 feature sets
    fig, ax = plt.subplots(figsize=(18, 6))
    f1_w = [res_watch[m]['f1'] for m in models_sorted]
    f1_d = [res_demo[m]['f1']  for m in models_sorted]
    f1_e = [res_eads[m]['f1']  for m in models_sorted]
    ax.bar(x - w, f1_w, w, label='Watch only',          color=C_W, alpha=0.88)
    ax.bar(x,     f1_d, w, label='Watch + Demographics', color=C_D, alpha=0.88)
    ax.bar(x + w, f1_e, w, label='Watch + EADS-21/VAS', color=C_E, alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(models_sorted, rotation=30, ha='right')
    ax.set_ylabel('Binary F1 (stress class)')
    ax.set_ylim(0, 1.1)
    ax.set_title('All models — 3 feature sets (Anxolotl LOSO-CV, BL-norm)')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig11_all_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print('\nSaved fig11_all_models_comparison.png')

    # Fig B — heatmap: models × feature sets
    heat_data = pd.DataFrame({
        'Watch only':           {m: res_watch[m]['f1'] for m in models_sorted},
        'Watch + Demographics': {m: res_demo[m]['f1']  for m in models_sorted},
        'Watch + EADS-21/VAS':  {m: res_eads[m]['f1']  for m in models_sorted},
    })
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Binary F1'})
    ax.set_title('F1 heatmap — all models × feature sets\n(Anxolotl LOSO-CV, BL-norm)', pad=10)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig('figures/fig12_heatmap_all_models.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Saved fig12_heatmap_all_models.png')

    # Fig C — delta heatmap (gain from adding features)
    delta_data = pd.DataFrame({
        'Δ Demographics': {m: res_demo[m]['f1']  - res_watch[m]['f1'] for m in models_sorted},
        'Δ EADS-21/VAS':  {m: res_eads[m]['f1']  - res_watch[m]['f1'] for m in models_sorted},
    })
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(delta_data, annot=True, fmt='+.3f', cmap='RdBu_r',
                center=0, vmin=-0.05, vmax=0.05, ax=ax, linewidths=0.3,
                cbar_kws={'label': 'ΔF1 vs watch-only'})
    ax.set_title('ΔF1 from adding features\n(positive = improvement)', pad=10)
    plt.tight_layout()
    plt.savefig('figures/fig13_delta_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Saved fig13_delta_heatmap.png')

    print(f'\nSaved results/all_models_demographics.csv')
    print(f'\nBest model overall: {df_res.iloc[0]["model"]} — F1={df_res.iloc[0]["F1_watch"]:.3f}')
